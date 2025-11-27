# Implementation of Z-Image transformer for Forge, based on Chroma pattern

from dataclasses import dataclass
import math
import torch
from torch import nn
from backend.attention import attention_function
from backend.utils import fp16_fix, tensor2parameter
from backend.nn.flux import attention, rope, timestep_embedding, EmbedND, MLPEmbedder, RMSNorm, QKNorm, SelfAttention

@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img, txt, mod, pe):
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = mod
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        B, L, _ = img_qkv.shape
        H = self.num_heads
        D = img_qkv.shape[-1] // (3 * H)
        img_q, img_k, img_v = img_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        B, L, _ = txt_qkv.shape
        txt_q, txt_k, txt_v = txt_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = fp16_fix(txt)
        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x, mod, pe):
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        del x_mod

        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        del q, k, v, pe
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        del attn, mlp

        x = x + mod.gate * output
        x = fp16_fix(x)
        return x

class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, mod):
        shift, scale = mod
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class IntegratedZImageTransformer2DModel(nn.Module):
    def __init__(self, in_channels: int = 16, vec_in_dim: int = 768, context_in_dim: int = 4096,
                 hidden_size: int = 3072, mlp_ratio: float = 4.0, num_heads: int = 24,
                 depth: int = 19, depth_single_blocks: int = 38, axes_dim: list = None,
                 theta: int = 10000, qkv_bias: bool = True, guidance_embed: bool = True, **kwargs):
        super().__init__()

        if axes_dim is None:
            axes_dim = [16, 56, 56]

        self.in_channels = in_channels
        self.out_channels = in_channels

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.guidance_embed = guidance_embed

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        if guidance_embed:
            self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        else:
            self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(self, img, img_ids, txt, txt_ids, timesteps, y=None, guidance=None, **kwargs):
        # Process timestep and guidance embeddings
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        # Process image and text inputs
        img = self.img_in(img)
        txt = self.txt_in(txt)

        # Compute positional embeddings
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # Process through double stream blocks
        for block in self.double_blocks:
            # Create modulations (simplified - real implementation would generate from vec)
            img_mod = (ModulationOut(shift=vec.unsqueeze(1), scale=vec.unsqueeze(1), gate=vec.unsqueeze(1)),
                      ModulationOut(shift=vec.unsqueeze(1), scale=vec.unsqueeze(1), gate=vec.unsqueeze(1)))
            txt_mod = (ModulationOut(shift=vec.unsqueeze(1), scale=vec.unsqueeze(1), gate=vec.unsqueeze(1)),
                      ModulationOut(shift=vec.unsqueeze(1), scale=vec.unsqueeze(1), gate=vec.unsqueeze(1)))
            img, txt = block(img, txt, (img_mod, txt_mod), pe=pe)

        # Concatenate image and text for single stream blocks
        x = torch.cat((txt, img), dim=1)

        # Process through single stream blocks
        for block in self.single_blocks:
            mod = ModulationOut(shift=vec.unsqueeze(1), scale=vec.unsqueeze(1), gate=vec.unsqueeze(1))
            x = block(x, mod, pe=pe)

        # Extract image part
        img = x[:, txt.shape[1]:, :]

        # Final layer
        img = self.final_layer(img, (vec.unsqueeze(1), vec.unsqueeze(1)))

        return img
