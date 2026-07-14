# Krea 2 (K2) single-stream MMDiT, vendored from https://github.com/krea-ai/krea-2 (mmdit.py).
# Adapted for Forge: torch.compile decorators removed, default SDPA backend (no forced
# CUDNN kernel), plus a converter for the Diffusers checkpoint layout
# (Krea2Transformer2DModel) and an adapter around the Qwen-Image VAE.

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> Tensor:
    if gqa and k.shape[1] != q.shape[1]:
        # enable_gqa is not supported by every SDPA backend / torch build, so expand
        # the kv heads manually to stay compatible.
        repeats = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return rearrange(x, "B H L D -> B L (H D)")


def _mask(mask: Tensor) -> Tensor:
    """Expand a (B, L) key-padding mask into a (B, 1, L, L) attention mask."""
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


# Krea 2 large-wide config, matches both OSS checkpoints (Raw / Turbo).
KREA2_CONFIG = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        # .to(device): Forge's CPU-swap partial load only moves modules exposing a
        # `weight` attribute to GPU, so this parameter can be left on CPU.
        out = vec + rearrange(self.lin.to(device=vec.device), "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        # .to(device): see SimpleModulation — may be left on CPU by partial load.
        out = vec + self.lin.to(device=vec.device)
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        # .to(device): see SimpleModulation — may be left on CPU by partial load.
        t = F.rms_norm(
            t, (self.features,), eps=self.eps,
            weight=(self.scale.to(device=x.device, dtype=torch.float32) + 1.0),
        )
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(
        self, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate))

        return out


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))

        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth — that's fixed at 2 + 2 blocks.
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)
        x = x.squeeze(-1)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)

        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift, freqs, mask
        )
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)

        return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(
            config.features, axes, theta=config.theta, ntk=1.0
        )
        self.first = nn.Linear(
            config.channels * config.patch**2, config.features, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        txtmask = _mask(mask[:, : context.shape[1]])

        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        # Pad combined sequence to a multiple of 256 to stabilize kernel shapes.
        fulllen = combined.shape[1]
        _padlen = (-fulllen) % 256
        if _padlen > 0:
            combined = F.pad(combined, (0, 0, 0, _padlen))
            mask = F.pad(mask, (0, _padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, _padlen))

        mask = _mask(mask)

        freqs = self.posemb(pos)

        for block in self.blocks:
            combined = block(combined, tvec, freqs, mask)

        final = self.last(combined, t)
        output = final[:, txtlen : txtlen + imglen, :]

        return output


def prepare_image_tokens(img: Tensor, txtlen: int, patch: int, txtmask: Tensor):
    """Patchify the latent and build the combined text+image position / mask tensors.

    Returns (img_tokens, pos, mask). Same as `prepare` in the reference sampling.py.
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def is_diffusers_krea2_state_dict(sd) -> bool:
    return 'time_mod_proj.weight' in sd or any(k.startswith('transformer_blocks.') for k in sd)


def _convert_krea2_block_key(key: str) -> str:
    """Rename Diffusers attention/ff/norm sub-keys to the reference block naming."""
    replacements = [
        ('attn.to_q.', 'attn.wq.'),
        ('attn.to_k.', 'attn.wk.'),
        ('attn.to_v.', 'attn.wv.'),
        ('attn.to_gate.', 'attn.gate.'),
        ('attn.to_out.0.', 'attn.wo.'),
        ('attn.norm_q.weight', 'attn.qknorm.qnorm.scale'),
        ('attn.norm_k.weight', 'attn.qknorm.knorm.scale'),
        ('norm1.weight', 'prenorm.scale'),
        ('norm2.weight', 'postnorm.scale'),
        ('ff.gate.', 'mlp.gate.'),
        ('ff.up.', 'mlp.up.'),
        ('ff.down.', 'mlp.down.'),
    ]
    for src, dst in replacements:
        key = key.replace(src, dst)
    return key


def convert_diffusers_krea2_state_dict(sd):
    """Convert a Diffusers Krea2Transformer2DModel checkpoint to the reference
    SingleStreamDiT naming. The Diffusers model is a pure rename of the reference
    implementation (same zero-centered RMSNorm scales and additive scale-shift
    tables), so only key names and one reshape change."""
    out = {}
    for key, value in sd.items():
        new_key = key

        if key.startswith('img_in.'):
            new_key = key.replace('img_in.', 'first.', 1)
        elif key.startswith('time_embed.linear_1.'):
            new_key = key.replace('time_embed.linear_1.', 'tmlp.0.', 1)
        elif key.startswith('time_embed.linear_2.'):
            new_key = key.replace('time_embed.linear_2.', 'tmlp.2.', 1)
        elif key.startswith('time_mod_proj.'):
            new_key = key.replace('time_mod_proj.', 'tproj.1.', 1)
        elif key == 'txt_in.norm.weight':
            new_key = 'txtmlp.0.scale'
        elif key.startswith('txt_in.linear_1.'):
            new_key = key.replace('txt_in.linear_1.', 'txtmlp.1.', 1)
        elif key.startswith('txt_in.linear_2.'):
            new_key = key.replace('txt_in.linear_2.', 'txtmlp.3.', 1)
        elif key.startswith('text_fusion.'):
            new_key = _convert_krea2_block_key(key.replace('text_fusion.', 'txtfusion.', 1))
        elif key.startswith('transformer_blocks.'):
            new_key = key.replace('transformer_blocks.', 'blocks.', 1)
            if new_key.endswith('.scale_shift_table'):
                # (6, dim) table -> flattened (6 * dim) modulation parameter
                new_key = new_key.replace('.scale_shift_table', '.mod.lin')
                value = value.reshape(-1)
            else:
                new_key = _convert_krea2_block_key(new_key)
        elif key == 'final_layer.scale_shift_table':
            new_key = 'last.modulation.lin'
        elif key == 'final_layer.norm.weight':
            new_key = 'last.norm.scale'
        elif key.startswith('final_layer.linear.'):
            new_key = key.replace('final_layer.linear.', 'last.linear.', 1)

        out[new_key] = value
    return out


# Inverse of the _convert_krea2_block_key / convert_diffusers_krea2_state_dict
# renames, for mapping Diffusers-named LoRA keys onto the reference model.
_KREA2_REFERENCE_TO_DIFFUSERS = [
    ('first.', 'img_in.'),
    ('tmlp.0.', 'time_embed.linear_1.'),
    ('tmlp.2.', 'time_embed.linear_2.'),
    ('tproj.1.', 'time_mod_proj.'),
    ('txtmlp.1.', 'txt_in.linear_1.'),
    ('txtmlp.3.', 'txt_in.linear_2.'),
    ('txtfusion.', 'text_fusion.'),
    ('blocks.', 'transformer_blocks.'),
    ('last.linear.', 'final_layer.linear.'),
    ('attn.wq.', 'attn.to_q.'),
    ('attn.wk.', 'attn.to_k.'),
    ('attn.wv.', 'attn.to_v.'),
    ('attn.gate.', 'attn.to_gate.'),
    ('attn.wo.', 'attn.to_out.0.'),
    ('mlp.gate.', 'ff.gate.'),
    ('mlp.up.', 'ff.up.'),
    ('mlp.down.', 'ff.down.'),
]


def krea2_lora_key_map(sdk, key_map):
    """Build LoRA-file-key -> model-key mappings for Krea 2.

    The transformer sits behind Krea2TransformerWrapper, so model state dict
    keys are 'diffusion_model.mmdit.blocks...' while LoRA files address the
    bare reference names ('blocks.N.attn.wq') or the Diffusers renames
    ('transformer_blocks.N.attn.to_q'); register both spellings under the
    common LoRA prefix conventions."""
    for k in sdk:
        if not (k.startswith('diffusion_model.') and k.endswith('.weight')):
            continue
        inner = k[len('diffusion_model.'):-len('.weight')]
        if inner.startswith('mmdit.'):
            inner = inner[len('mmdit.'):]
        # The rename table matches with trailing dots, so convert before
        # stripping the '.weight' suffix.
        diffusers = inner + '.weight'
        for ref, diff in _KREA2_REFERENCE_TO_DIFFUSERS:
            diffusers = diffusers.replace(ref, diff)
        diffusers = diffusers[:-len('.weight')]
        for name in {inner, diffusers}:
            key_map['diffusion_model.{}'.format(name)] = k
            key_map['transformer.{}'.format(name)] = k  # diffusers/PEFT
            key_map['lora_unet_{}'.format(name.replace('.', '_'))] = k  # kohya
            key_map['lycoris_{}'.format(name.replace('.', '_'))] = k
    return key_map


class Krea2QwenVAE(torch.nn.Module):
    """Wraps the Qwen-Image VAE (AutoencoderKLQwenImage, f8 / 16 latent channels)
    behind the 4D encode/decode interface that Forge's VAE patcher expects, with
    the per-channel latent normalization the Krea 2 transformer was trained on."""

    def __init__(self, config: dict):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        config = {k: v for k, v in config.items() if not k.startswith('_')}
        self.ae = AutoencoderKLQwenImage.from_config(config)
        self.register_buffer('latents_mean', torch.tensor(config['latents_mean']).view(1, -1, 1, 1, 1))
        self.register_buffer('latents_std', torch.tensor(config['latents_std']).view(1, -1, 1, 1, 1))

        # Shim for Forge's VAE class: downscale_ratio = 2 ** (len(down_block_types) - 1) = 8
        self.config = SimpleNamespace(
            down_block_types=(None, None, None, None),
            latent_channels=config.get('z_dim', 16),
        )

    def encode(self, x: Tensor) -> Tensor:
        # x: (B, 3, H, W) in [-1, 1]
        x = x.unsqueeze(2)  # add frame dimension: (B, 3, 1, H, W)
        z = self.ae.encode(x).latent_dist.mode()
        z = (z - self.latents_mean.to(z)) / self.latents_std.to(z)
        return z.squeeze(2)

    def decode(self, z: Tensor) -> Tensor:
        z = z.unsqueeze(2)
        z = z * self.latents_std.to(z) + self.latents_mean.to(z)
        x = self.ae.decode(z).sample
        return x.squeeze(2)  # (B, 3, H, W) in [-1, 1]
