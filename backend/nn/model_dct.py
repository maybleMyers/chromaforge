from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt

from .chroma_layers_dct import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    SingleStreamBlock,
    timestep_embedding,
    Approximator,
    distribute_modulations,
    NerfEmbedder,
    NerfFinalLayer,
    NerfFinalLayerConv,
    NerfGLUBlock
)
from backend.operations import using_forge_operations
from backend import memory_management

# Removed chroma_dct_memory_estimation function
# ChromaDCT now uses the same memory estimation as regular Chroma
# This fixes the issue where ChromaDCT was keeping too much in CPU swap


@dataclass
class ChromaParams:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    approximator_in_dim: int
    approximator_depth: int
    approximator_hidden_size: int
    patch_size: int
    nerf_hidden_size: int
    nerf_mlp_ratio: int
    nerf_depth: int
    nerf_max_freqs: int
    _use_compiled: bool


chroma_params = ChromaParams(
    in_channels=3,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
    approximator_in_dim=64,
    approximator_depth=5,
    approximator_hidden_size=5120,
    patch_size=16,
    nerf_hidden_size=64,
    nerf_mlp_ratio=4,
    nerf_depth=4,
    nerf_max_freqs=8,
    _use_compiled=False,
)


def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=8):
    """
    Modifies attention mask to allow attention to a few extra padding tokens.

    Args:
        mask: Original attention mask (1 for tokens to attend to, 0 for masked tokens)
        max_seq_length: Maximum sequence length of the model
        num_extra_padding: Number of padding tokens to unmask

    Returns:
        Modified mask
    """
    # Get the actual sequence length from the mask
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]

    modified_mask = mask.clone()

    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())

        # Only add extra padding tokens if there's room
        if current_seq_len < max_seq_length:
            # Calculate how many padding tokens we can unmask
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)

            # Unmask the specified number of padding tokens right after the sequence
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1

    return modified_mask


class Chroma(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: ChromaParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        # self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        # patchify ops
        # Use forge operations for memory management compatibility
        with using_forge_operations(device=memory_management.cpu, dtype=memory_management.unet_dtype(), manual_cast_enabled=True):
            self.img_in_patch = nn.Conv2d(
                params.in_channels,
                params.hidden_size,
                kernel_size=params.patch_size,
                stride=params.patch_size,
                bias=True
            )
            nn.init.zeros_(self.img_in_patch.weight)
            nn.init.zeros_(self.img_in_patch.bias)
        # TODO: need proper mapping for this approximator output!
        # currently the mapping is hardcoded in distribute_modulations function
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            self.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
        )
        # Use forge operations for memory management compatibility
        with using_forge_operations(device=memory_management.cpu, dtype=memory_management.unet_dtype(), manual_cast_enabled=True):
            self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        # self.final_layer = LastLayer(
        #     self.hidden_size,
        #     1,
        #     self.out_channels,
        #     use_compiled=params._use_compiled,
        # )

        # pixel channel concat with DCT 
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs
        )

        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                use_compiled=params._use_compiled
            ) for _ in range(params.nerf_depth)
        ])
        # self.nerf_final_layer = NerfFinalLayer(
        #     params.nerf_hidden_size,
        #     out_channels=params.in_channels,
        #     use_compiled=params._use_compiled
        # )
        self.nerf_final_layer_conv = NerfFinalLayerConv(
            params.nerf_hidden_size,
            out_channels=params.in_channels,
            use_compiled=params._use_compiled
        )
        # TODO: move this hardcoded value to config
        # single layer has 3 modulation vectors
        # double layer has 6 modulation vectors for each expert
        # final layer has 2 modulation vectors
        self.mod_index_length = 3 * params.depth_single_blocks + 2 * 6 * params.depth + 2
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks = params.depth
        # self.mod_index = torch.tensor(list(range(self.mod_index_length)), device=0)
        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length))),
            persistent=False,
        )
        self.approximator_in_dim = params.approximator_in_dim


    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device








    def inner_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        guidance: Tensor,
        attn_padding: int = 1,
    ) -> Tensor:
        if img.ndim != 4:
            raise ValueError("Input img tensor must be in [B, C, H, W] format.")
        if txt.ndim != 3:
            raise ValueError("Input txt tensors must have 3 dimensions.")
        B, C, H, W = img.shape

        # gemini gogogo idk how to unfold and pack the patch properly :P
        # Store the raw pixel values of each patch for the NeRF head later.
        # unfold creates patches: [B, C * P * P, NumPatches]
        nerf_pixels = nn.functional.unfold(img, kernel_size=self.params.patch_size, stride=self.params.patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2) # -> [B, NumPatches, C * P * P]
        
        # partchify ops
        img = self.img_in_patch(img) # -> [B, Hidden, H/P, W/P]
        num_patches = img.shape[2] * img.shape[3]
        # flatten into a sequence for the transformer.
        img = img.flatten(2).transpose(1, 2) # -> [B, NumPatches, Hidden]

        txt = self.txt_in(txt)

        # TODO:
        # need to fix grad accumulation issue here for now it's in no grad mode
        # besides, i don't want to wash out the PFP that's trained on this model weights anyway
        # the fan out operation here is deleting the backward graph
        # alternatively doing forward pass for every block manually is doable but slow
        # custom backward probably be better
        with torch.no_grad():
            distill_timestep = timestep_embedding(timesteps, self.approximator_in_dim//4)
            # TODO: need to add toggle to omit this from schnell but that's not a priority
            distil_guidance = timestep_embedding(guidance, self.approximator_in_dim//4)
            # get all modulation index - ensure it's on the same device as input tensors
            mod_index_device = self.mod_index.to(timesteps.device)
            modulation_index = timestep_embedding(mod_index_device, self.approximator_in_dim//2)
            # we need to broadcast the modulation index here so each batch has all of the index
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
            # and we need to broadcast timestep and guidance along too
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .repeat(1, self.mod_index_length, 1)
            )
            # then and only then we could concatenate it together
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
            input_vec = input_vec.to(img.dtype)
            mod_vectors = self.distilled_guidance_layer(input_vec)
        mod_vectors_dict = distribute_modulations(mod_vectors, self.depth_single_blocks, self.depth_double_blocks)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # compute mask
        # assume max seq length from the batched input

        max_len = txt.shape[1]

        # mask
        with torch.no_grad():
            txt_mask_w_padding = modify_mask_to_attend_padding(
                txt_mask, max_len, attn_padding
            )
            txt_img_mask = torch.cat(
                [
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img.shape[1]], device=txt_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = (
                txt_img_mask[None, None, ...]
                .repeat(txt.shape[0], self.num_heads, 1, 1)
                .int()
                .bool()
            )
            # txt_mask_w_padding[txt_mask_w_padding==False] = True

        for i, block in enumerate(self.double_blocks):
            # the guidance replaced by FFN output
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]

            double_mod = [img_mod, txt_mod]

            # just in case in different GPU for simple pipeline parallel
            if self.training:
                img, txt = ckpt.checkpoint(
                    block, img, txt, pe, double_mod, txt_img_mask
                )
            else:
                img, txt = block(
                    img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask
                )

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            if self.training:
                img = ckpt.checkpoint(block, img, pe, single_mod, txt_img_mask)
            else:
                img = block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)
        img = img[:, txt.shape[1] :, ...]

        # final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        # img = self.final_layer(
        #     img, distill_vec=final_mod
        # )  # (N, T, patch_size ** 2 * out_channels)

        # aliasing
        nerf_hidden = img
        # reshape for per-patch processing
        nerf_hidden = nerf_hidden.reshape(B * num_patches, self.params.hidden_size)
        nerf_pixels = nerf_pixels.reshape(B * num_patches, C, self.params.patch_size**2).transpose(1, 2)

        # get DCT-encoded pixel embeddings [pixel-dct]
        img_dct = self.nerf_image_embedder(nerf_pixels)

        # pass through the dynamic MLP blocks (the NeRF)
        for i, block in enumerate(self.nerf_blocks):
            if self.training:
                img_dct = ckpt.checkpoint(block, img_dct, nerf_hidden)
            else:
                img_dct = block(img_dct, nerf_hidden)

        # final projection to get the output pixel values
        # img_dct = self.nerf_final_layer(img_dct) # -> [B*NumPatches, P*P, C]
        img_dct = self.nerf_final_layer_conv.norm(img_dct)
        
        # gemini gogogo idk how to fold this properly :P
        # Reassemble the patches into the final image.
        img_dct = img_dct.transpose(1, 2) # -> [B*NumPatches, C, P*P]
        # Reshape to combine with batch dimension for fold
        img_dct = img_dct.reshape(B, num_patches, -1) # -> [B, NumPatches, C*P*P]
        img_dct = img_dct.transpose(1, 2) # -> [B, C*P*P, NumPatches]
        img_dct = nn.functional.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=self.params.patch_size,
            stride=self.params.patch_size
        ) # [B, Hidden, H, W]
        img_dct = self.nerf_final_layer_conv.conv(img_dct)

        return img_dct

    def forward(self, x, timestep, context, **kwargs):
        """
        Forge-compatible forward method that adapts to the expected interface.
        
        Args:
            x: Input image tensor [B, C, H, W] 
            timestep: Timestep tensor [B] or scalar
            context: Text conditioning tensor [B, seq_len, dim]
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Output tensor [B, C, H, W]
        """
        bs, c, h, w = x.shape
        input_device = x.device
        input_dtype = x.dtype
        
        # Convert timestep to correct format
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).repeat(bs)
        elif timestep.shape[0] == 1 and bs > 1:
            timestep = timestep.repeat(bs)
        
        # Create image position IDs (similar to regular Chroma)
        patch_size = self.params.patch_size  # Use the actual patch size from params
        h_patches = h // patch_size
        w_patches = w // patch_size
        
        img_ids = torch.zeros((h_patches, w_patches, 3), device=input_device, dtype=input_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_patches - 1, steps=h_patches, device=input_device, dtype=input_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_patches - 1, steps=w_patches, device=input_device, dtype=input_dtype)[None, :]
        img_ids = img_ids[None, :].repeat(bs, 1, 1, 1)
        img_ids = img_ids.reshape(bs, h_patches * w_patches, 3)
        
        # Create text position IDs (all zeros)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
        
        # Create text mask (assume all tokens are valid)
        txt_mask = torch.ones((bs, context.shape[1]), device=input_device, dtype=torch.bool)
        
        # Create guidance (set to 0 for now, might need adjustment)
        guidance = torch.zeros((bs,), device=input_device, dtype=input_dtype)
        
        # Call the inner forward method
        result = self.inner_forward(
            img=x,
            img_ids=img_ids, 
            txt=context,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=timestep,
            guidance=guidance,
            attn_padding=1
        )
        
        return result


class IntegratedChromaDCTTransformer2DModel(Chroma):
    """
    Integrated wrapper for ChromaDCT model to match the memory management
    of IntegratedChromaTransformer2DModel.

    This inherits directly from Chroma to ensure state dict keys match properly.
    """

    def __init__(self, **config):
        # Use default ChromaDCT params (defined at the top of this file)
        # The config passed from Flux config doesn't have the right parameters
        params = chroma_params  # Use the predefined ChromaDCT parameters

        # Call parent Chroma init directly
        super().__init__(params)

        # The model is self, not a separate attribute
        # This ensures state dict keys match without "model." prefix
