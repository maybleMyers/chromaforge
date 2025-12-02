# Z-Image ControlNet components ported from VideoX-Fun
# These provide control block classes for Z-Image ControlNet integration

import types
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.utils import is_torch_version
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention.
    """
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0+."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
        ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(nn.Module):
    """Base transformer block for Z-Image"""
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            )
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    """Control transformer block that generates hints - matches VideoX-Fun"""
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        block_id: int = 0
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id

        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x=None, **kwargs):
        """
        Forward pass for control block.
        c: control context [bsz, seq_len, dim]
        x: main unified tensor [bsz, seq_len, dim] - passed as keyword arg
        kwargs: attn_mask, freqs_cis, adaln_input
        """
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


def add_zimage_control_components(transformer, control_state_dict, device='cpu', dtype=torch.bfloat16):
    """
    Dynamically add control components to an existing Z-Image transformer.
    This patches the transformer to support control_context in forward().
    """
    import types

    # Get transformer config
    dim = 3840  # Z-Image default
    n_heads = 30
    n_kv_heads = 30
    norm_eps = 1e-5
    qk_norm = True
    in_channels = 16
    n_refiner_layers = 2

    # Try to get actual values from transformer
    if hasattr(transformer, 'dim'):
        dim = transformer.dim
    if hasattr(transformer, 'n_heads'):
        n_heads = transformer.n_heads
    if hasattr(transformer, 'in_channels'):
        in_channels = transformer.in_channels

    # Control layer positions from model config: [0, 5, 10, 15, 20, 25] = 6 layers
    # This matches Z-Image-Turbo-Fun-Controlnet-Union config
    control_layers_places = [0, 5, 10, 15, 20, 25]

    # Add control attributes
    transformer.control_layers_places = control_layers_places
    transformer.control_layers_mapping = {i: n for n, i in enumerate(control_layers_places)}
    transformer.control_in_dim = in_channels

    # Create control layers - these generate the hints
    transformer.control_layers = nn.ModuleList([
        ZImageControlTransformerBlock(
            i, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
            modulation=True, block_id=idx
        )
        for idx, i in enumerate(control_layers_places)
    ]).to(device=device, dtype=dtype)

    # Create control embedders (same structure as main embedder)
    # patch_size=2, f_patch_size=1, so input dim = 1 * 2 * 2 * in_channels = 4 * 16 = 64
    transformer.control_all_x_embedder = nn.ModuleDict({
        "2-1": nn.Linear(4 * in_channels, dim, bias=True)
    }).to(device=device, dtype=dtype)

    # Create control noise refiner (same structure as main noise_refiner)
    transformer.control_noise_refiner = nn.ModuleList([
        ZImageTransformerBlock(
            1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True
        )
        for layer_id in range(n_refiner_layers)
    ]).to(device=device, dtype=dtype)

    # Patch each main layer to accept hints
    for idx, layer in enumerate(transformer.layers):
        block_id = transformer.control_layers_mapping.get(idx, None)

        # Save original forward
        layer._original_forward = layer.forward

        # Create patched forward that accepts hints
        def make_patched_forward(orig_forward, block_id):
            def patched_forward(self, x, attn_mask=None, freqs_cis=None, adaln_input=None,
                              hints=None, context_scale=1.0, **kwargs):
                # Call original forward
                result = orig_forward(x, attn_mask, freqs_cis, adaln_input)
                # Apply hints if available and this is a control layer
                if hints is not None and block_id is not None and block_id < len(hints):
                    result = result + hints[block_id] * context_scale
                return result
            return patched_forward

        layer.forward = types.MethodType(
            make_patched_forward(layer._original_forward, block_id), layer
        )

    # Add forward_control method - THIS IS THE KEY METHOD
    transformer.forward_control = types.MethodType(_forward_control_method, transformer)

    # Load the control weights
    missing, unexpected = transformer.load_state_dict(control_state_dict, strict=False)
    print(f"Z-Image ControlNet loaded - missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing and len(missing) < 20:
        print(f"Missing keys: {missing}")

    transformer._control_layers_loaded = True

    return transformer


def _forward_control_method(
    self,
    x,           # unified tensor from main forward [bsz, unified_max_seqlen, dim]
    cap_feats,   # PADDED cap_feats tensor [bsz, cap_max_seqlen, dim] (after context_refiner)
    control_context,  # List of raw VAE latents [C, F, H, W] per batch item
    kwargs,      # dict with attn_mask, freqs_cis, adaln_input from main forward
    t=None,      # timestep embedding
    patch_size=2,
    f_patch_size=1,
):
    """
    Generate control hints from control context.
    This matches VideoX-Fun's forward_control method exactly.
    """
    if control_context is None:
        return None

    bsz = len(control_context)
    device = control_context[0].device
    pH = pW = patch_size
    pF = f_patch_size

    # Get cap_padding_len from cap_feats (it's a padded tensor now)
    cap_padding_len = cap_feats[0].size(0) if cap_feats.dim() == 3 else cap_feats.size(1)

    # Patchify control context (inline implementation of VideoX-Fun's patchify method)
    control_context_patches = []
    x_size = []
    x_pos_ids = []
    x_inner_pad_mask = []

    for i, ctrl in enumerate(control_context):
        C, F, H, W = ctrl.size()
        x_size.append((F, H, W))
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

        # Reshape to patches: "c f pf h ph w pw -> (f h w) (pf ph pw c)"
        ctrl = ctrl.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        ctrl = ctrl.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

        ctrl_ori_len = len(ctrl)
        ctrl_padding_len = (-ctrl_ori_len) % SEQ_MULTI_OF

        # Position IDs
        ctrl_ori_pos_ids = create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens),
            start=(cap_padding_len + 1, 0, 0),
            device=device,
        ).flatten(0, 2)
        ctrl_padding_pos_ids = create_coordinate_grid(
            size=(1, 1, 1),
            start=(0, 0, 0),
            device=device,
        ).flatten(0, 2).repeat(ctrl_padding_len, 1)
        ctrl_padded_pos_ids = torch.cat([ctrl_ori_pos_ids, ctrl_padding_pos_ids], dim=0)
        x_pos_ids.append(ctrl_padded_pos_ids)

        # Pad mask
        x_inner_pad_mask.append(
            torch.cat([
                torch.zeros((ctrl_ori_len,), dtype=torch.bool, device=device),
                torch.ones((ctrl_padding_len,), dtype=torch.bool, device=device),
            ], dim=0)
        )

        # Padded feature
        ctrl_padded = torch.cat([ctrl, ctrl[-1:].repeat(ctrl_padding_len, 1)], dim=0)
        control_context_patches.append(ctrl_padded)

    # control_context embed & refine
    x_item_seqlens = [len(_) for _ in control_context_patches]
    x_max_item_seqlen = max(x_item_seqlens)

    control_context_cat = torch.cat(control_context_patches, dim=0)
    control_context_cat = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context_cat)

    # Match timestep embedding dtype
    adaln_input = t.type_as(control_context_cat)

    # Apply pad token (use main x_pad_token, same as VideoX-Fun)
    control_context_cat[torch.cat(x_inner_pad_mask)] = self.x_pad_token

    control_context_split = list(control_context_cat.split(x_item_seqlens, dim=0))

    # Get freqs_cis using main rope_embedder (same as VideoX-Fun)
    x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

    control_context_padded = pad_sequence(control_context_split, batch_first=True, padding_value=0.0)
    x_freqs_cis_padded = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    # Refine control context through control_noise_refiner
    for layer in self.control_noise_refiner:
        control_context_padded = layer(control_context_padded, x_attn_mask, x_freqs_cis_padded, adaln_input)

    # Unify control context with cap_feats (same as VideoX-Fun lines 266-274)
    cap_item_seqlens = [len(_) for _ in cap_feats]  # iterate over batch dim of tensor
    control_context_unified = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        control_context_unified.append(torch.cat([control_context_padded[i][:x_len], cap_feats[i][:cap_len]]))
    control_context_unified = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)
    c = control_context_unified

    # Pass through control_layers with MAIN forward's kwargs (this is critical!)
    # VideoX-Fun: new_kwargs = dict(x=x); new_kwargs.update(kwargs)
    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)

    for layer in self.control_layers:
        c = layer(c, **new_kwargs)

    # Extract hints (all but last element of stacked tensor)
    hints = list(torch.unbind(c))[:-1]

    return hints


def create_coordinate_grid(size, start=None, device=None):
    """Create coordinate grid for position IDs"""
    if start is None:
        start = (0 for _ in size)

    axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
    grids = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grids, dim=-1)
