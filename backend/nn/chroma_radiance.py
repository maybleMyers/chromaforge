# implementation of Chroma Radiance for Forge, based on the radiance model changes
# References: 
# - Original radiance implementation: flow/src/models/chroma/model_dct.py
# - Forge Chroma implementation: backend/nn/chroma.py

from dataclasses import dataclass
from functools import lru_cache
import math
import torch
from torch import nn
from einops import rearrange, repeat
from backend.attention import attention_function
from backend.utils import fp16_fix, tensor2parameter
from backend.nn.flux import attention, rope, timestep_embedding, EmbedND, MLPEmbedder, RMSNorm, QKNorm, SelfAttention
from backend.nn.chroma import Approximator, ModulationOut, DoubleStreamBlock, SingleStreamBlock, LastLayer


class NerfEmbedder(nn.Module):
    """
    An embedder module that combines input features with a 2D positional
    encoding that mimics the Discrete Cosine Transform (DCT).
    """
    def __init__(self, in_channels, hidden_size_input, max_freqs, dtype=None, device=None, operations=None):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        
        if operations is None:
            self.embedder = nn.Sequential(
                nn.Linear(in_channels + max_freqs**2, hidden_size_input)
            )
        else:
            self.embedder = nn.Sequential(
                operations.Linear(in_channels + max_freqs**2, hidden_size_input, dtype=dtype, device=device)
            )
    
    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size, device, dtype):
        """Generate and cache 2D DCT-like positional embeddings."""
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        
        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        
        coeffs = (1 + freqs_x * freqs_y) ** -1
        
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        
        return dct

    def forward(self, inputs):
        """Forward pass for the embedder."""
        B, P2, C = inputs.shape
        patch_size = int(P2 ** 0.5)
        
        dct = self.fetch_pos(patch_size, inputs.device, inputs.dtype)
        dct = dct.repeat(B, 1, 1)
        
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        
        return inputs


class NerfGLUBlock(nn.Module):
    """A NerfBlock using a Gated Linear Unit (GLU) like MLP."""
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        if operations is None:
            self.param_generator = nn.Linear(hidden_size_s, total_params)
        else:
            self.param_generator = operations.Linear(hidden_size_s, total_params, dtype=dtype, device=device)
        self.norm = RMSNorm(hidden_size_x)
        self.mlp_ratio = mlp_ratio

    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params = self.param_generator(s)
        
        fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)
        
        fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2 = fc2_params.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)
        
        fc1_gate = torch.nn.functional.normalize(fc1_gate, dim=-2)
        fc1_value = torch.nn.functional.normalize(fc1_value, dim=-2)
        fc2 = torch.nn.functional.normalize(fc2, dim=-2)
        
        res_x = x
        x = self.norm(x)
        
        x = torch.bmm(torch.nn.functional.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)
        x = x + res_x
        return x


class NerfFinalLayer(nn.Module):
    """Final layer for NeRF processing."""
    def __init__(self, hidden_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        if operations is None:
            self.linear = nn.Linear(hidden_size, out_channels)
        else:
            self.linear = operations.Linear(hidden_size, out_channels, dtype=dtype, device=device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


class NerfFinalLayerConv(nn.Module):
    """Convolutional final layer for NeRF processing."""
    def __init__(self, hidden_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        if operations is None:
            self.conv = nn.Conv2d(
                in_channels=hidden_size,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        else:
            self.conv = operations.Conv2d(
                in_channels=hidden_size,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                dtype=dtype,
                device=device
            )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x shape: [N, C, H, W]
        x_permuted = x.permute(0, 2, 3, 1)  # [N, H, W, C]
        x_norm = self.norm(x_permuted)
        x_norm_permuted = x_norm.permute(0, 3, 1, 2)  # [N, C, H, W]
        x = self.conv(x_norm_permuted)
        return x


@dataclass
class ChromaRadianceParams:
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
    nerf_tile_size: int
    nerf_final_head_type: str = "linear"  # "linear" or "conv"
    _use_compiled: bool = False


def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=8):
    """Modifies attention mask to allow attention to a few extra padding tokens."""
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]
    modified_mask = mask.clone()
    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())
        if current_seq_len < max_seq_length:
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1
    return modified_mask


class ChromaRadianceTransformer2DModel(nn.Module):
    """Transformer model for radiance flow matching operating directly in latent space."""
    
    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = ChromaRadianceParams(**kwargs)
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
        self.approximator_in_dim = params.approximator_in_dim

        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )

        # Patchify operations
        self.img_in_patch = nn.Conv2d(
            params.in_channels,
            params.hidden_size,
            kernel_size=params.patch_size,
            stride=params.patch_size,
            bias=True
        )
        nn.init.zeros_(self.img_in_patch.weight)
        nn.init.zeros_(self.img_in_patch.bias)

        # Distilled guidance layer
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            self.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
            operations=operations,
        )

        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
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
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        # NeRF components for direct pixel output
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                dtype=dtype,
                device=device,
                operations=operations
            ) for _ in range(params.nerf_depth)
        ])

        # Choose final layer type based on configuration
        if params.nerf_final_head_type == "conv":
            self.nerf_final_layer_conv = NerfFinalLayerConv(
                params.nerf_hidden_size,
                out_channels=params.in_channels,
                dtype=dtype,
                device=device,
                operations=operations
            )
            self.nerf_final_layer = None
        else:
            self.nerf_final_layer = NerfFinalLayer(
                params.nerf_hidden_size,
                out_channels=params.in_channels,
                dtype=dtype,
                device=device,
                operations=operations
            )
            self.nerf_final_layer_conv = None

        # Modulation parameters
        self.mod_index_length = 3 * params.depth_single_blocks + 2 * 6 * params.depth + 2
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks = params.depth

        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length))),
            persistent=False,
        )

    def forward_tiled_nerf(
        self,
        nerf_hidden: torch.Tensor,
        nerf_pixels: torch.Tensor,
        B: int,
        C: int,
        num_patches: int,
        tile_size: int = 16
    ) -> torch.Tensor:
        """
        Processes the NeRF head in tiles to save memory.
        nerf_hidden has shape [B, L, D]
        nerf_pixels has shape [B, L, C * P * P]
        """
        output_tiles = []
        # Iterate over the patches in tiles. The dimension L (num_patches) is at index 1.
        for i in range(0, num_patches, tile_size):
            end = min(i + tile_size, num_patches)

            # Slice the current tile from the input tensors
            nerf_hidden_tile = nerf_hidden[:, i:end, :]
            nerf_pixels_tile = nerf_pixels[:, i:end, :]

            # Get the actual number of patches in this tile (can be smaller for the last tile)
            num_patches_tile = nerf_hidden_tile.shape[1]

            # Reshape the tile for per-patch processing
            # [B, NumPatches_tile, D] -> [B * NumPatches_tile, D]
            nerf_hidden_tile = nerf_hidden_tile.reshape(B * num_patches_tile, self.params.hidden_size)
            # [B, NumPatches_tile, C*P*P] -> [B*NumPatches_tile, C, P*P] -> [B*NumPatches_tile, P*P, C]
            nerf_pixels_tile = nerf_pixels_tile.reshape(B * num_patches_tile, C, self.params.patch_size**2).transpose(1, 2)

            # get DCT-encoded pixel embeddings [pixel-dct]
            img_dct_tile = self.nerf_image_embedder(nerf_pixels_tile)

            # pass through the dynamic MLP blocks (the NeRF)
            for block in self.nerf_blocks:
                img_dct_tile = block(img_dct_tile, nerf_hidden_tile)

            output_tiles.append(img_dct_tile)

        # Concatenate the processed tiles along the patch dimension
        return torch.cat(output_tiles, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def distribute_modulations(tensor, single_block_count: int = 38, double_blocks_count: int = 19):
        """Distributes slices of the tensor into the block_dict as ModulationOut objects."""
        batch_size, vectors, dim = tensor.shape
        block_dict = {}
        
        for i in range(single_block_count):
            key = f"single_blocks.{i}.modulation.lin"
            block_dict[key] = None
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.img_mod.lin"
            block_dict[key] = None
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.txt_mod.lin"
            block_dict[key] = None
        block_dict["final_layer.adaLN_modulation.1"] = None
        
        idx = 0
        for key in block_dict.keys():
            if "single_blocks" in key:
                block_dict[key] = ModulationOut(
                    shift=tensor[:, idx:idx+1, :],
                    scale=tensor[:, idx+1:idx+2, :],
                    gate=tensor[:, idx+2:idx+3, :]
                )
                idx += 3
            elif "img_mod" in key:
                double_block = []
                for _ in range(2):
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3
                block_dict[key] = double_block
            elif "txt_mod" in key:
                double_block = []
                for _ in range(2):
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3
                block_dict[key] = double_block
            elif "final_layer" in key:
                block_dict[key] = [
                    tensor[:, idx:idx+1, :],
                    tensor[:, idx+1:idx+2, :],
                ]
                idx += 2
        return block_dict

    def forward(self, x, timestep, context, attn_padding=1, **kwargs):
        if x.ndim == 3:
            # Input is already flattened sequence [B, H*W, C] from diffusion engine
            B, HW, C = x.shape
            H = W = int((HW) ** 0.5)  # Assume square latent
            # Unflatten to spatial format for processing
            x = x.transpose(1, 2).reshape(B, C, H, W)  # -> [B, C, H, W]
        elif x.ndim != 4:
            raise ValueError("Input x tensor must be in [B, C, H, W] or [B, H*W, C] format.")
        
        if context.ndim != 3:
            raise ValueError("Input context tensor must have 3 dimensions.")
        
        B, C, H, W = x.shape
        
        # Store raw pixel values for NeRF head (now working with 64-channel latent)
        nerf_pixels = nn.functional.unfold(x, kernel_size=self.params.patch_size, stride=self.params.patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2)  # -> [B, NumPatches, C * P * P]
        
        # Patchify operations 
        img = self.img_in_patch(x)  # -> [B, Hidden, H/P, W/P]
        num_patches = img.shape[2] * img.shape[3]
        img = img.flatten(2).transpose(1, 2)  # -> [B, NumPatches, Hidden]
        
        txt = self.txt_in(context)
        
        # Generate modulations
        with torch.no_grad():
            distill_timestep = timestep_embedding(timestep, self.approximator_in_dim//4)
            distil_guidance = timestep_embedding(torch.zeros_like(timestep), self.approximator_in_dim//4)
            
            modulation_index = timestep_embedding(self.mod_index.to(timestep.device), self.approximator_in_dim//2)
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
            
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .repeat(1, self.mod_index_length, 1)
            )
            
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))

        mod_vectors_dict = self.distribute_modulations(mod_vectors, self.depth_single_blocks, self.depth_double_blocks)

        # Prepare positional encodings
        img_ids = torch.zeros((img.shape[0], img.shape[1], 3), device=x.device, dtype=x.dtype)
        txt_ids = torch.zeros((txt.shape[0], txt.shape[1], 3), device=x.device, dtype=x.dtype)
        
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # Compute attention mask (placeholder - you may need to implement proper masking)
        max_len = txt.shape[1]
        txt_mask = torch.ones([txt.shape[0], txt.shape[1]], device=txt.device)
        
        with torch.no_grad():
            txt_mask_w_padding = modify_mask_to_attend_padding(txt_mask, max_len, attn_padding)
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

        # Double blocks
        for i, block in enumerate(self.double_blocks):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]
            img, txt = block(img=img, txt=txt, mod=double_mod, pe=pe)

        # Single blocks
        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            img = block(img, mod=single_mod, pe=pe)

        img = img[:, txt.shape[1]:, ...]

        # NeRF processing with tiling
        img_dct = self.forward_tiled_nerf(img, nerf_pixels, B, C, num_patches, tile_size=self.params.nerf_tile_size)

        # Final projection
        if self.nerf_final_layer_conv is not None:
            # Convolutional final layer path
            img_dct = self.nerf_final_layer_conv.norm(img_dct)
            img_dct = img_dct.transpose(1, 2)  # -> [B*NumPatches, C, P*P]
            img_dct = img_dct.reshape(B, num_patches, -1)  # -> [B, NumPatches, C*P*P]
            img_dct = img_dct.transpose(1, 2)  # -> [B, C*P*P, NumPatches]
            img_dct = nn.functional.fold(
                img_dct,
                output_size=(H, W),
                kernel_size=self.params.patch_size,
                stride=self.params.patch_size
            )  # [B, Hidden, H, W]
            img_dct = self.nerf_final_layer_conv.conv(img_dct)
        else:
            # Linear final layer path
            img_dct = self.nerf_final_layer(img_dct)  # -> [B*NumPatches, P*P, C]
            img_dct = img_dct.transpose(1, 2)  # -> [B*NumPatches, C, P*P]
            img_dct = img_dct.reshape(B, num_patches, -1)  # -> [B, NumPatches, C*P*P]
            img_dct = img_dct.transpose(1, 2)  # -> [B, C*P*P, NumPatches]
            img_dct = nn.functional.fold(
                img_dct,
                output_size=(H, W),
                kernel_size=self.params.patch_size,
                stride=self.params.patch_size
            )

        # ChromaRadiance works directly with RGB spatial format [B, C, H, W]
        return img_dct


class IntegratedChromaRadianceTransformer2DModel(nn.Module):
    """Integrated wrapper for ChromaRadiance that works with Forge's loading system."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Create the actual ChromaRadiance model using current Forge operations context
        self.model = ChromaRadianceTransformer2DModel(**kwargs)
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def load_state_dict(self, state_dict, strict=True):
        """
        Custom state dict loading to handle the model. prefix mapping.
        The safetensors file has keys like 'txt_in.weight' but PyTorch expects 'model.txt_in.weight'
        because of the self.model wrapper.
        """
        # Create a new state dict with proper prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            # Add 'model.' prefix to all parameters for the wrapped model
            new_key = f'model.{key}'
            new_state_dict[new_key] = value
            
        return super().load_state_dict(new_state_dict, strict=strict)
        
    @property 
    def device(self):
        return self.model.device