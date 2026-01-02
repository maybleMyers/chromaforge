"""
Advanced Noise Generators for Diffusion Sampling

Provides power-spectral noise generators (fractal, pyramid) adapted from RES4LYF.
These allow for structured noise with different frequency characteristics.

Fractal noise alpha values:
  - brown = 2.0 (low frequency dominance)
  - pink = 1.0 (balanced)
  - white = 0.0 (equal power across frequencies)
  - blue = -1.0 (high frequency bias)
  - violet = -2.0 (strong high frequency bias)
"""

import torch
import torch.nn as nn
from functools import partial


def noise_generator_factory(cls, **fixed_params):
    """Factory function to create noise generator instances with fixed parameters."""
    def create_instance(**kwargs):
        params = {**fixed_params, **kwargs}
        return cls(**params)
    return create_instance


class NoiseGenerator:
    """Base class for noise generators."""

    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None,
                 seed=42, generator=None, sigma_min=None, sigma_max=None):
        self.seed = seed

        if x is not None:
            self.x = x
            self.size = x.shape
            self.dtype = x.dtype
            self.layout = x.layout
            self.device = x.device
        else:
            self.size = size
            self.dtype = dtype or torch.float32
            self.layout = layout or torch.strided
            self.device = device or torch.device('cpu')
            self.x = torch.zeros(self.size, dtype=self.dtype, layout=self.layout, device=self.device)

        # Allow overriding parameters if specified
        if size is not None:
            self.size = size
        if dtype is not None:
            self.dtype = dtype
        if layout is not None:
            self.layout = layout
        if device is not None:
            self.device = device

        self.sigma_max = sigma_max.to(self.device) if isinstance(sigma_max, torch.Tensor) else sigma_max
        self.sigma_min = sigma_min.to(self.device) if isinstance(sigma_min, torch.Tensor) else sigma_min

        self.last_seed = seed

        if generator is None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.generator = generator

    def __call__(self, **kwargs):
        raise NotImplementedError("Subclasses must implement __call__")

    def update(self, **kwargs):
        """Update generator parameters."""
        updated_values = []
        for attribute_name, value in kwargs.items():
            if value is not None:
                setattr(self, attribute_name, value)
            updated_values.append(getattr(self, attribute_name))
        return tuple(updated_values)


class GaussianNoiseGenerator(NoiseGenerator):
    """Standard Gaussian noise generator."""

    def __call__(self, **kwargs):
        self.last_seed += 1
        noise = torch.randn(
            self.size,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            generator=self.generator
        )
        return noise


class FractalNoiseGenerator(NoiseGenerator):
    """
    Fractal noise generator using FFT-based power spectral filtering.

    The alpha parameter controls the spectral slope:
      - alpha=2.0: Brown noise (1/f^2, low frequency dominance)
      - alpha=1.0: Pink noise (1/f, balanced)
      - alpha=0.0: White noise (flat spectrum)
      - alpha=-1.0: Blue noise (f, high frequency bias)
      - alpha=-2.0: Violet noise (f^2, strong high frequency)
    """

    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None,
                 seed=42, generator=None, sigma_min=None, sigma_max=None,
                 alpha=0.0, k=1.0, scale=0.1):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.alpha = alpha
        self.k = k
        self.scale = scale

    def __call__(self, *, alpha=None, k=None, scale=None, **kwargs):
        if alpha is not None:
            self.alpha = alpha
        if k is not None:
            self.k = k
        if scale is not None:
            self.scale = scale

        self.last_seed += 1

        # Handle both 4D (B, C, H, W) and 5D (B, C, T, H, W) tensors
        if len(self.size) == 5:
            b, c, t, h, w = self.size
        else:
            b, c, h, w = self.size

        # Generate base Gaussian noise
        noise = torch.normal(
            mean=0.0, std=1.0,
            size=self.size,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            generator=self.generator
        )

        # Create frequency grids
        y_freq = torch.fft.fftfreq(h, 1/h, device=self.device)
        x_freq = torch.fft.fftfreq(w, 1/w, device=self.device)

        if len(self.size) == 5:
            t_freq = torch.fft.fftfreq(t, 1/t, device=self.device)
            freq = torch.sqrt(
                t_freq[:, None, None]**2 +
                y_freq[None, :, None]**2 +
                x_freq[None, None, :]**2
            ).clamp(min=1e-10)
        else:
            freq = torch.sqrt(y_freq[:, None]**2 + x_freq[None, :]**2).clamp(min=1e-10)

        # Compute spectral density (power-law filtering)
        spectral_density = self.k / torch.pow(freq, self.alpha * self.scale)
        spectral_density[..., 0, 0] = 0  # Zero DC component

        # Apply spectral filtering in frequency domain
        noise_fft = torch.fft.fftn(noise)
        modified_fft = noise_fft * spectral_density
        noise = torch.fft.ifftn(modified_fft).real

        # Normalize to unit variance
        return noise / torch.std(noise)


class PyramidNoiseGenerator(NoiseGenerator):
    """
    Multi-octave pyramid noise generator.

    Creates noise by summing upsampled noise at multiple scales,
    with a discount factor controlling the contribution of each octave.
    """

    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None,
                 seed=42, generator=None, sigma_min=None, sigma_max=None,
                 discount=0.8, mode='nearest-exact'):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.discount = discount
        self.mode = mode

    def __call__(self, *, discount=None, mode=None, **kwargs):
        if discount is not None:
            self.discount = discount
        if mode is not None:
            self.mode = mode

        self.last_seed += 1

        x = torch.zeros(self.size, dtype=self.dtype, layout=self.layout, device=self.device)

        if len(self.size) == 5:
            b, c, t, h, w = self.size
            orig_h, orig_w, orig_t = h, w, t
        else:
            b, c, h, w = self.size
            orig_h, orig_w = h, w

        r = 1
        for i in range(5):
            r *= 2

            if len(self.size) == 5:
                scaled_size = (b, c, t * r, h * r, w * r)
                orig_size = (orig_t, orig_h, orig_w)
            else:
                scaled_size = (b, c, h * r, w * r)
                orig_size = (orig_h, orig_w)

            # Generate noise at higher resolution and downsample
            octave_noise = torch.normal(
                mean=0, std=0.5 ** i,
                size=scaled_size,
                dtype=self.dtype,
                layout=self.layout,
                device=self.device,
                generator=self.generator
            )

            # Interpolate to original size and add with discount
            x += torch.nn.functional.interpolate(
                octave_noise,
                size=orig_size,
                mode=self.mode
            ) * self.discount ** i

        return x / x.std()


class HiresPyramidNoiseGenerator(NoiseGenerator):
    """
    High-resolution pyramid noise generator.

    Similar to PyramidNoiseGenerator but starts from current resolution
    and adds detail at higher scales.
    """

    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None,
                 seed=42, generator=None, sigma_min=None, sigma_max=None,
                 discount=0.7, mode='nearest-exact'):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.discount = discount
        self.mode = mode

    def __call__(self, *, discount=None, mode=None, **kwargs):
        if discount is not None:
            self.discount = discount
        if mode is not None:
            self.mode = mode

        self.last_seed += 1

        if len(self.size) == 5:
            b, c, t, h, w = self.size
            orig_h, orig_w, orig_t = h, w, t
            u = nn.Upsample(size=(orig_t, orig_h, orig_w), mode=self.mode).to(self.device)
        else:
            b, c, h, w = self.size
            orig_h, orig_w = h, w
            orig_t = t = 1
            u = nn.Upsample(size=(orig_h, orig_w), mode=self.mode).to(self.device)

        # Start with base noise
        noise = ((torch.rand(
            size=self.size,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            generator=self.generator
        ) - 0.5) * 2 * 1.73)

        for i in range(4):
            r = torch.rand(1, device=self.device, generator=self.generator).item() * 2 + 2
            h_new = min(orig_h * 15, int(h * (r ** i)))
            w_new = min(orig_w * 15, int(w * (r ** i)))

            if len(self.size) == 5:
                t_new = min(orig_t * 15, int(t * (r ** i)))
                new_noise = torch.randn(
                    (b, c, t_new, h_new, w_new),
                    dtype=self.dtype, layout=self.layout,
                    device=self.device, generator=self.generator
                )
            else:
                new_noise = torch.randn(
                    (b, c, h_new, w_new),
                    dtype=self.dtype, layout=self.layout,
                    device=self.device, generator=self.generator
                )

            upsampled_noise = u(new_noise)
            noise += upsampled_noise * self.discount ** i

            if h_new >= orig_h * 15 or w_new >= orig_w * 15:
                break

        return noise / noise.std()


# Preset noise generators with fixed parameters
NOISE_PRESETS = {
    # Standard
    'gaussian': GaussianNoiseGenerator,

    # Fractal noise by color
    'brown': noise_generator_factory(FractalNoiseGenerator, alpha=2.0),
    'pink': noise_generator_factory(FractalNoiseGenerator, alpha=1.0),
    'white': noise_generator_factory(FractalNoiseGenerator, alpha=0.0),
    'blue': noise_generator_factory(FractalNoiseGenerator, alpha=-1.0),
    'violet': noise_generator_factory(FractalNoiseGenerator, alpha=-2.0),

    # Pyramid noise
    'pyramid': noise_generator_factory(PyramidNoiseGenerator, mode='bilinear'),
    'pyramid-bicubic': noise_generator_factory(PyramidNoiseGenerator, mode='bicubic'),
    'pyramid-nearest': noise_generator_factory(PyramidNoiseGenerator, mode='nearest'),
    'hires-pyramid': noise_generator_factory(HiresPyramidNoiseGenerator, mode='bilinear'),
    'hires-pyramid-bicubic': noise_generator_factory(HiresPyramidNoiseGenerator, mode='bicubic'),
}

# List of available noise types for UI
NOISE_TYPES = tuple(NOISE_PRESETS.keys())
