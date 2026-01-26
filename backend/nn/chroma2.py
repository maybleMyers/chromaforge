from dataclasses import dataclass

import math
import torch

from torch import nn
from einops import rearrange, repeat
from backend.attention import attention_function
from backend.utils import fp16_fix, tensor2parameter
from backend.nn.flux import rope, timestep_embedding, EmbedND


if hasattr(torch, 'rms_norm'):
    functional_rms_norm = torch.rms_norm
else:
    def functional_rms_norm(x, normalized_shape, weight, eps):
        if x.dtype in [torch.bfloat16, torch.float32]:
            n = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps) * weight
        else:
            n = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps).to(x.dtype) * weight
        return x * n


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = None
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6
        self.normalized_shape = [dim]

    def forward(self, x):
        if self.scale.dtype != x.dtype:
            self.scale = tensor2parameter(self.scale.to(dtype=x.dtype))
        return functional_rms_norm(x, self.normalized_shape, self.scale, self.eps)


def tensor2parameter(x):
    return nn.Parameter(x, requires_grad=False)


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        del v
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(k), k.to(q)


def attention(q, k, v, pe):
    q, k = apply_rope(q, k, pe)
    x = attention_function(q, k, v, q.shape[1], skip_reshape=True)
    return x


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    del xq_, xk_
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, bias=False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x):
        x = self.silu(self.in_layer(x))
        return self.out_layer(x)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class SharedModulation(nn.Module):
    def __init__(self, dim, n_mods=6):
        super().__init__()
        self.lin = nn.Linear(dim, n_mods * dim, bias=False)
        self.n_mods = n_mods
        self.dim = dim

    def forward(self, vec):
        out = self.lin(vec)
        out_chunks = out.chunk(self.n_mods, dim=-1)
        if self.n_mods == 6:
            return [
                ModulationOut(shift=out_chunks[0], scale=out_chunks[1], gate=out_chunks[2]),
                ModulationOut(shift=out_chunks[3], scale=out_chunks[4], gate=out_chunks[5])
            ]
        elif self.n_mods == 3:
            return ModulationOut(shift=out_chunks[0], scale=out_chunks[1], gate=out_chunks[2])
        else:
            return out_chunks


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        H = self.num_heads
        D = qkv.shape[-1] // (3 * H)
        q, k, v = qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        return q, k, v


class Flux2DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=3.0, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = mlp_hidden_dim

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            nn.Identity(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            nn.Identity(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def _swiglu(self, mlp, x):
        gate_up = mlp[0](x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = nn.functional.silu(gate) * up
        x = mlp[2](x)
        return x

    def forward(self, img, txt, img_mod, txt_mod, pe):
        img_mod1, img_mod2 = img_mod
        txt_mod1, txt_mod2 = txt_mod

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale.unsqueeze(1)) * img_modulated + img_mod1.shift.unsqueeze(1)
        img_q, img_k, img_v = self.img_attn(img_modulated)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale.unsqueeze(1)) * txt_modulated + txt_mod1.shift.unsqueeze(1)
        txt_q, txt_k, txt_v = self.txt_attn(txt_modulated)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate.unsqueeze(1) * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate.unsqueeze(1) * self._swiglu(
            self.img_mlp,
            (1 + img_mod2.scale.unsqueeze(1)) * self.img_norm2(img) + img_mod2.shift.unsqueeze(1)
        )

        txt = txt + txt_mod1.gate.unsqueeze(1) * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate.unsqueeze(1) * self._swiglu(
            self.txt_mlp,
            (1 + txt_mod2.scale.unsqueeze(1)) * self.txt_norm2(txt) + txt_mod2.shift.unsqueeze(1)
        )

        txt = fp16_fix(txt)
        return img, txt


class Flux2SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=3.0, qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, mod, pe):
        x_mod = (1 + mod.scale.unsqueeze(1)) * self.pre_norm(x) + mod.shift.unsqueeze(1)

        out = self.linear1(x_mod)
        qkv = out[..., :3 * self.hidden_size]
        gate = out[..., 3 * self.hidden_size:3 * self.hidden_size + self.mlp_hidden_dim]
        up = out[..., 3 * self.hidden_size + self.mlp_hidden_dim:]

        del x_mod, out

        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        del qkv

        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        del q, k, v

        mlp = nn.functional.silu(gate) * up
        del gate, up

        output = self.linear2(torch.cat((attn, mlp), dim=-1))
        del attn, mlp

        x = x + mod.gate.unsqueeze(1) * output
        x = fp16_fix(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=False)
        )

    def forward(self, x, vec):
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        x = (1 + scale.unsqueeze(1)) * self.norm_final(x) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class IntegratedChroma2Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 5,
        num_single_layers: int = 20,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        joint_attention_dim: int = 7680,
        axes_dim: list = None,
        theta: int = 10000,
        **kwargs
    ):
        super().__init__()

        if axes_dim is None:
            axes_dim = [32, 32, 32, 32]

        hidden_size = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers

        pe_dim = hidden_size // num_attention_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.img_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.txt_in = nn.Linear(joint_attention_dim, hidden_size, bias=False)

        self.time_in = MLPEmbedder(256, hidden_size)

        self.double_stream_modulation_img = SharedModulation(hidden_size, n_mods=6)
        self.double_stream_modulation_txt = SharedModulation(hidden_size, n_mods=6)
        self.single_stream_modulation = SharedModulation(hidden_size, n_mods=3)

        self.double_blocks = nn.ModuleList([
            Flux2DoubleStreamBlock(hidden_size, num_attention_heads, mlp_ratio=3.0, qkv_bias=False)
            for _ in range(num_layers)
        ])

        self.single_blocks = nn.ModuleList([
            Flux2SingleStreamBlock(hidden_size, num_attention_heads, mlp_ratio=3.0)
            for _ in range(num_single_layers)
        ])

        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)

    def inner_forward(self, img, img_ids, txt, txt_ids, timesteps):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        txt = self.txt_in(txt)

        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

        double_img_mod = self.double_stream_modulation_img(vec)
        double_txt_mod = self.double_stream_modulation_txt(vec)
        single_mod = self.single_stream_modulation(vec)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        del ids

        for block in self.double_blocks:
            img, txt = block(img, txt, double_img_mod, double_txt_mod, pe)

        img = torch.cat((txt, img), 1)

        for block in self.single_blocks:
            img = block(img, single_mod, pe)

        del pe
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)
        return img

    def forward(self, x, timestep, context, **kwargs):
        bs, c, h, w = x.shape
        input_device = x.device
        input_dtype = x.dtype
        patch_size = 2

        pad_h = (patch_size - x.shape[-2] % patch_size) % patch_size
        pad_w = (patch_size - x.shape[-1] % patch_size) % patch_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        del x

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)

        out = self.inner_forward(img, img_ids, context, txt_ids, timestep)
        del img, img_ids, txt_ids, timestep, context

        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        return out
