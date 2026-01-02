"""
Advanced Schedulers for Diffusion Sampling

Provides beta distribution-based schedulers and sigma rescaling utilities
adapted from RES4LYF for use with Z-Image DCT and other diffusion models.
"""

import torch
import numpy as np
from scipy import stats


def beta57_scheduler(n, sigma_min, sigma_max, inner_model=None, device='cpu'):
    """
    Beta57 preset scheduler with fixed alpha=0.5, beta=0.7.

    This is a commonly used preset from RES4LYF that provides good results
    for flow-matching models. The beta distribution creates a non-uniform
    timestep distribution that can improve sampling quality.

    Args:
        n: Number of steps
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        inner_model: Optional model wrapper (for compatibility, not used)
        device: Target device for the output tensor

    Returns:
        Tensor of sigma values with shape (n+1,), ending with 0.0
    """
    alpha = 0.5
    beta = 0.7

    # Generate timesteps using inverse CDF of beta distribution
    timesteps = 1 - np.linspace(0, 1, n)
    timesteps = [stats.beta.ppf(x, alpha, beta) for x in timesteps]

    # Map timesteps to sigma range
    sigmas = [sigma_min + (x * (sigma_max - sigma_min)) for x in timesteps]
    sigmas += [0.0]  # Append final sigma

    return torch.FloatTensor(sigmas).to(device)


def beta_scheduler_custom(n, sigma_min, sigma_max, alpha=0.5, beta=0.7, device='cpu'):
    """
    Custom beta scheduler with configurable alpha and beta parameters.

    Args:
        n: Number of steps
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        device: Target device for the output tensor

    Returns:
        Tensor of sigma values with shape (n+1,), ending with 0.0
    """
    timesteps = 1 - np.linspace(0, 1, n)
    timesteps = [stats.beta.ppf(x, alpha, beta) for x in timesteps]
    sigmas = [sigma_min + (x * (sigma_max - sigma_min)) for x in timesteps]
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)


def rescale_sigmas(sigmas, start=1.0, end=0.0):
    """
    Rescale sigmas to a new range (ComfyUI style).

    Maps the sigma schedule to a new range defined by start and end,
    where the values represent fractions of the original sigma_max:
    - start=1.0, end=0.0: No change (full range, max to 0)
    - start=0.5, end=0.0: Use lower half of noise range (0.5*max to 0)
    - start=0.225, end=0.0: Start at 22.5% of max noise (0.225*max to 0)
    - start=1.0, end=0.1: Full start but don't fully denoise (max to 0.1*max)

    This is useful for:
    - Reducing initial noise for smoother results (lower start)
    - Partial denoising / img2img-like effects (higher end)
    - Fine-tuning the denoise range for specific models

    Args:
        sigmas: Input sigma tensor
        start: Fraction of sigma_max for first sigma (default 1.0)
        end: Fraction of sigma_max for last non-zero sigma (default 0.0)

    Returns:
        Rescaled sigma tensor
    """
    if sigmas.numel() == 0:
        return sigmas

    n = len(sigmas)
    if n <= 1:
        return sigmas

    # Get the original range (excluding the final 0)
    sig_max = sigmas[0].item()  # First sigma is typically the max
    sig_min = sigmas[-1].item()  # Last sigma is typically 0

    # Calculate new range based on original max
    new_max = sig_max * start
    new_min = sig_max * end

    # Linearly interpolate sigmas to new range
    # Normalize original sigmas to 0-1
    if sig_max - sig_min < 1e-8:
        return sigmas

    t = (sigmas - sig_min) / (sig_max - sig_min)

    # Map to new range
    rescaled = new_min + t * (new_max - new_min)

    return rescaled


def rescale_sigmas_factor(sigmas, factor=0.988):
    """
    Convenience function to rescale sigmas by a factor.

    Equivalent to rescale_sigmas(sigmas, factor, 0.0).

    Args:
        sigmas: Input sigma tensor
        factor: Rescale factor (default 0.988)

    Returns:
        Rescaled sigma tensor
    """
    return rescale_sigmas(sigmas, start=factor, end=0.0)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """
    Generate exponentially spaced sigmas (for reference/compatibility).

    Args:
        n: Number of steps
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        device: Target device

    Returns:
        Tensor of sigma values
    """
    sigmas = torch.linspace(
        np.log(sigma_max),
        np.log(sigma_min),
        n,
        device=device
    ).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_linear(n, sigma_min, sigma_max, device='cpu'):
    """
    Generate linearly spaced sigmas (for reference/compatibility).

    Args:
        n: Number of steps
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        device: Target device

    Returns:
        Tensor of sigma values
    """
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return torch.cat([sigmas, sigmas.new_zeros([1])])
