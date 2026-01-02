"""
Latent Operations

Provides SLERP, FFT blending, orthogonal noise generation, and other
latent manipulation utilities adapted from RES4LYF.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.linalg import norm
import math


def slerp_tensor(val: float, low: Tensor, high: Tensor) -> Tensor:
    """
    Spherical linear interpolation between two tensors.

    Preserves the magnitude while interpolating direction on a hypersphere.

    Args:
        val: Interpolation factor (0=low, 1=high)
        low: Starting tensor
        high: Ending tensor

    Returns:
        Interpolated tensor
    """
    if low.ndim == 4 and low.shape[-3] > 1:
        dim = -3
    elif low.ndim == 5 and low.shape[-3] > 1:
        dim = -4
    elif low.ndim == 2:
        dim = (-2, -1)
    else:
        dim = -1

    low_norm = low / (torch.norm(low, dim=dim, keepdim=True) + 1e-8)
    high_norm = high / (torch.norm(high, dim=dim, keepdim=True) + 1e-8)

    dot = (low_norm * high_norm).sum(dim=dim, keepdim=True).clamp(-1.0, 1.0)

    near = dot > 0.9995
    opposite = dot < -0.9995
    condition = torch.logical_or(near, opposite)

    omega = torch.acos(dot)
    so = torch.sin(omega)

    factor_low = torch.sin((1 - val) * omega) / (so + 1e-8)
    factor_high = torch.sin(val * omega) / (so + 1e-8)

    res = factor_low * low + factor_high * high
    res = torch.where(condition, low * (1 - val) + high * val, res)
    return res


def slerp(v0: Tensor, v1: Tensor, t: float, dot_threshold: float = 0.9995) -> Tensor:
    """
    Spherical linear interpolation with fallback to lerp for edge cases.

    Args:
        v0: Starting vector
        v1: Final vector
        t: Interpolation factor (0-1)
        dot_threshold: Threshold for considering vectors as colinear

    Returns:
        Interpolated vector
    """
    assert v0.shape == v1.shape, "shapes must match"

    v0_norm = norm(v0, dim=-1)
    v1_norm = norm(v1, dim=-1)

    v0_normed = v0 / (v0_norm.unsqueeze(-1) + 1e-8)
    v1_normed = v1 / (v1_norm.unsqueeze(-1) + 1e-8)

    dot = (v0_normed * v1_normed).sum(-1)
    dot_mag = dot.abs()

    gotta_lerp = dot_mag.isnan() | (dot_mag > dot_threshold)

    if gotta_lerp.all():
        return torch.lerp(v0, v1, t)

    theta_0 = dot.arccos().unsqueeze(-1)
    sin_theta_0 = theta_0.sin()
    theta_t = theta_0 * t
    sin_theta_t = theta_t.sin()

    s0 = (theta_0 - theta_t).sin() / (sin_theta_0 + 1e-8)
    s1 = sin_theta_t / (sin_theta_0 + 1e-8)
    slerped = s0 * v0 + s1 * v1

    out = torch.where(gotta_lerp.unsqueeze(-1), torch.lerp(v0, v1, t), slerped)
    return out


def mix_latent_fft(
    lat0: Tensor,
    lat1: Tensor,
    phase_weight: float = 0.5,
    magnitude_weight: float = 0.5
) -> Tensor:
    """
    Mix two latents in FFT space by blending phase and magnitude separately.

    Args:
        lat0: First latent tensor
        lat1: Second latent tensor
        phase_weight: Weight for phase interpolation (0=lat0, 1=lat1)
        magnitude_weight: Weight for magnitude interpolation (0=lat0, 1=lat1)

    Returns:
        Blended latent tensor
    """
    fft0 = torch.fft.fftn(lat0, dim=(-2, -1))
    fft1 = torch.fft.fftn(lat1, dim=(-2, -1))

    mag0 = torch.abs(fft0)
    mag1 = torch.abs(fft1)
    phase0 = torch.angle(fft0)
    phase1 = torch.angle(fft1)

    blended_mag = mag0 * (1 - magnitude_weight) + mag1 * magnitude_weight

    phase_diff = phase1 - phase0
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    blended_phase = phase0 + phase_weight * phase_diff

    blended_fft = blended_mag * torch.exp(1j * blended_phase)
    result = torch.fft.ifftn(blended_fft, dim=(-2, -1)).real

    return result


def orthogonal_noise(noise: Tensor, *refs: Tensor) -> Tensor:
    """
    Make noise orthogonal to reference tensors using Gram-Schmidt.

    Args:
        noise: The noise tensor to orthogonalize
        *refs: Reference tensors to orthogonalize against

    Returns:
        Orthogonalized noise tensor
    """
    if noise.ndim == 4:
        b, c, h, w = noise.shape
    elif noise.ndim == 5:
        b, c, t, h, w = noise.shape
    else:
        return noise

    result = noise.clone()
    result_flat = result.view(b, c, -1)

    for ref in refs:
        ref_flat = ref.view(b, c, -1).clone()
        ref_flat = ref_flat / (ref_flat.norm(dim=-1, keepdim=True) + 1e-8)

        proj_coeff = torch.sum(result_flat * ref_flat, dim=-1, keepdim=True)
        projection = proj_coeff * ref_flat

        result_flat = result_flat - projection

    return result_flat.view_as(noise)


def latent_normalize_channels(x: Tensor) -> Tensor:
    """
    Normalize latent tensor per-channel to zero mean and unit variance.

    Args:
        x: Input latent tensor (B, C, H, W) or (B, C, T, H, W)

    Returns:
        Channel-normalized tensor
    """
    if x.ndim == 4:
        dims = (-2, -1)
    elif x.ndim == 5:
        dims = (-3, -2, -1)
    else:
        dims = -1

    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)

    return (x - mean) / (std + 1e-8)


def normalize_latent(
    target: Tensor,
    source: Tensor = None,
    mean: bool = True,
    std: bool = True,
    channelwise: bool = True
) -> Tensor:
    """
    Normalize target latent to match source latent statistics.

    Args:
        target: Target tensor to normalize
        source: Source tensor to match (if None, normalizes to zero mean/unit std)
        mean: Whether to match mean
        std: Whether to match standard deviation
        channelwise: Whether to normalize per channel

    Returns:
        Normalized tensor
    """
    result = target.clone()

    if channelwise and target.ndim >= 4:
        for b in range(result.shape[0]):
            for c in range(result.shape[1]):
                if mean and std:
                    result[b, c] = (target[b, c] - target[b, c].mean()) / (target[b, c].std() + 1e-8)
                    if source is not None:
                        result[b, c] = result[b, c] * source[b, c].std() + source[b, c].mean()
                elif mean:
                    result[b, c] = target[b, c] - target[b, c].mean()
                    if source is not None:
                        result[b, c] = result[b, c] + source[b, c].mean()
                elif std:
                    result[b, c] = target[b, c] / (target[b, c].std() + 1e-8)
                    if source is not None:
                        result[b, c] = result[b, c] * source[b, c].std()
    else:
        if mean and std:
            result = (target - target.mean()) / (target.std() + 1e-8)
            if source is not None:
                result = result * source.std() + source.mean()
        elif mean:
            result = target - target.mean()
            if source is not None:
                result = result + source.mean()
        elif std:
            result = target / (target.std() + 1e-8)
            if source is not None:
                result = result * source.std()

    return result


def get_cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
    """Compute cosine similarity between two tensors."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    return F.cosine_similarity(x_flat.unsqueeze(0), y_flat.unsqueeze(0))


def lerp_latent(a: Tensor, b: Tensor, t: float) -> Tensor:
    """Simple linear interpolation between tensors."""
    return a * (1 - t) + b * t


def blend_latents_weighted(
    latents: list,
    weights: list
) -> Tensor:
    """
    Blend multiple latents with specified weights.

    Args:
        latents: List of latent tensors
        weights: List of weights (will be normalized to sum to 1)

    Returns:
        Weighted blend of latents
    """
    weights = torch.tensor(weights, dtype=latents[0].dtype, device=latents[0].device)
    weights = weights / weights.sum()

    result = torch.zeros_like(latents[0])
    for latent, weight in zip(latents, weights):
        result = result + latent * weight

    return result
