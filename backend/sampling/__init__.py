"""
Advanced Sampling Module for Chromaforge

Provides advanced noise generators, schedulers, samplers, and latent operations
adapted from RES4LYF for use with Z-Image DCT and other diffusion models.
"""

from .advanced_noise import (
    NoiseGenerator,
    FractalNoiseGenerator,
    PyramidNoiseGenerator,
    NOISE_PRESETS,
)

from .advanced_schedulers import (
    beta57_scheduler,
    rescale_sigmas,
)

from .advanced_samplers import (
    sample_res_2m,
    sample_res_2s,
    sample_res_2m_sde,
    sample_res_2s_sde,
    sample_er_sde,
)

from .apg import (
    MomentumBuffer,
    APGContext,
    apg_guidance,
    project,
    get_apg_context,
    set_apg_context,
    create_apg_context,
)

from .guidance import (
    SimpleLatentGuide,
    GuidanceManager,
    LatentBlender,
    create_guidance_callback,
)

from .latent_ops import (
    slerp_tensor,
    slerp,
    mix_latent_fft,
    orthogonal_noise,
    latent_normalize_channels,
    normalize_latent,
    get_cosine_similarity,
    lerp_latent,
    blend_latents_weighted,
)

__all__ = [
    # Noise generators
    'NoiseGenerator',
    'FractalNoiseGenerator',
    'PyramidNoiseGenerator',
    'NOISE_PRESETS',
    # Schedulers
    'beta57_scheduler',
    'rescale_sigmas',
    # Samplers
    'sample_res_2m',
    'sample_res_2s',
    'sample_res_2m_sde',
    'sample_res_2s_sde',
    'sample_er_sde',
    # APG (Adaptive Projected Guidance)
    'MomentumBuffer',
    'APGContext',
    'apg_guidance',
    'project',
    'get_apg_context',
    'set_apg_context',
    'create_apg_context',
    # Guidance
    'SimpleLatentGuide',
    'GuidanceManager',
    'LatentBlender',
    'create_guidance_callback',
    # Latent operations
    'slerp_tensor',
    'slerp',
    'mix_latent_fft',
    'orthogonal_noise',
    'latent_normalize_channels',
    'normalize_latent',
    'get_cosine_similarity',
    'lerp_latent',
    'blend_latents_weighted',
]
