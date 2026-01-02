"""
Adaptive Projected Guidance (APG)

Implementation based on arXiv:2410.02416 "Eliminating Oversaturation and Artifacts
of High Guidance Scales in Diffusion Models"

APG decomposes the CFG update into parallel and orthogonal components relative to
the conditional prediction. The parallel component primarily causes oversaturation,
so reducing its influence via the eta parameter diminishes oversaturation without
compromising quality.

Key parameters:
- eta: Controls influence of parallel component (1.0 = standard CFG, 0.0 = fully orthogonal)
- momentum: Running average momentum for smoother guidance (-0.5 default)
- threshold: Norm threshold for update scaling (0 = disabled)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class MomentumBuffer:
    """Running average buffer for momentum-based APG smoothing."""

    def __init__(self, momentum: float = -0.5):
        """
        Initialize momentum buffer.

        Args:
            momentum: Momentum value for exponential moving average.
                     Negative values create a "look-ahead" effect.
        """
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor) -> None:
        """
        Update the running average with new value.

        Args:
            update_value: New update tensor to incorporate
        """
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

    def reset(self) -> None:
        """Reset the running average to zero."""
        self.running_average = 0


def project(
    v0: torch.Tensor,
    v1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project v0 onto v1, returning parallel and orthogonal components.

    Uses Gram-Schmidt orthogonalization to decompose v0 into:
    - v0_parallel: Component parallel to v1 (causes oversaturation)
    - v0_orthogonal: Component orthogonal to v1 (enhances quality)

    Args:
        v0: Vector to project [B, C, H, W]
        v1: Reference vector [B, C, H, W]

    Returns:
        Tuple of (parallel_component, orthogonal_component)
    """
    device = v0.device
    dtype = v0.dtype

    # Move to CPU for XPU devices to avoid precision issues
    if device.type == "xpu":
        v0, v1 = v0.to("cpu"), v1.to("cpu")

    # Use double precision for accurate projection
    v0_d, v1_d = v0.double(), v1.double()

    # Normalize v1 across spatial dimensions
    v1_norm = F.normalize(v1_d, dim=[-1, -2, -3])

    # Compute parallel component via dot product
    dot_product = (v0_d * v1_norm).sum(dim=[-1, -2, -3], keepdim=True)
    v0_parallel = dot_product * v1_norm

    # Orthogonal component is the remainder
    v0_orthogonal = v0_d - v0_parallel

    return v0_parallel.to(device, dtype=dtype), v0_orthogonal.to(device, dtype=dtype)


def apg_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    eta: float = 1.0,
    momentum_buffer: Optional[MomentumBuffer] = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Apply Adaptive Projected Guidance to conditional and unconditional predictions.

    This function replaces standard CFG:
        Standard CFG: pred_cond + (guidance_scale - 1) * (pred_cond - pred_uncond)

    With APG:
        1. Compute difference: diff = pred_cond - pred_uncond
        2. Apply momentum smoothing if buffer provided
        3. Apply threshold scaling if threshold > 0
        4. Project diff onto pred_cond to get parallel/orthogonal components
        5. Combine: normalized_update = orthogonal + eta * parallel
        6. Return: pred_cond + (guidance_scale - 1) * normalized_update

    Args:
        pred_cond: Conditional model prediction [B, C, H, W]
        pred_uncond: Unconditional model prediction [B, C, H, W]
        guidance_scale: CFG scale (typically 1-30)
        eta: Parallel component weight (1.0 = standard CFG, 0.0 = fully orthogonal)
        momentum_buffer: Optional momentum buffer for smoothing
        threshold: Norm threshold for update scaling (0 = disabled)

    Returns:
        Guided prediction tensor
    """
    # Compute CFG difference
    diff = pred_cond - pred_uncond

    # Apply momentum smoothing
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    # Apply threshold-based scaling
    if threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, threshold / (diff_norm + 1e-8))
        diff = diff * scale_factor

    # Project onto parallel and orthogonal components
    diff_parallel, diff_orthogonal = project(diff, pred_cond)

    # Combine with eta weighting
    # eta = 1.0: standard CFG (full parallel component)
    # eta = 0.0: fully orthogonal (no oversaturation)
    normalized_update = diff_orthogonal + eta * diff_parallel

    # Apply guidance
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update

    return pred_guided


class APGContext:
    """
    Context manager for APG state during sampling.

    Manages the momentum buffer across sampling steps.
    """

    def __init__(
        self,
        enabled: bool = False,
        eta: float = 1.0,
        momentum: float = -0.5,
        threshold: float = 0.0,
    ):
        """
        Initialize APG context.

        Args:
            enabled: Whether APG is enabled
            eta: Parallel component weight
            momentum: Momentum for buffer (-0.5 recommended)
            threshold: Norm threshold (0 = disabled)
        """
        self.enabled = enabled
        self.eta = eta
        self.momentum = momentum
        self.threshold = threshold
        self.buffer: Optional[MomentumBuffer] = None

    def start(self) -> None:
        """Initialize buffer for new sampling run."""
        if self.enabled:
            self.buffer = MomentumBuffer(self.momentum)

    def end(self) -> None:
        """Clean up after sampling run."""
        self.buffer = None

    def apply(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Apply APG or standard CFG based on enabled state.

        Args:
            pred_cond: Conditional prediction
            pred_uncond: Unconditional prediction
            guidance_scale: CFG scale

        Returns:
            Guided prediction
        """
        if not self.enabled or self.eta >= 1.0:
            # Standard CFG
            return pred_cond + (guidance_scale - 1) * (pred_cond - pred_uncond)

        return apg_guidance(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guidance_scale=guidance_scale,
            eta=self.eta,
            momentum_buffer=self.buffer,
            threshold=self.threshold,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False


# Global APG context for use in CFG denoiser
_apg_context: Optional[APGContext] = None


def get_apg_context() -> Optional[APGContext]:
    """Get the current global APG context."""
    return _apg_context


def set_apg_context(context: Optional[APGContext]) -> None:
    """Set the global APG context."""
    global _apg_context
    _apg_context = context


def create_apg_context(
    enabled: bool = False,
    eta: float = 1.0,
    momentum: float = -0.5,
    threshold: float = 0.0,
) -> APGContext:
    """
    Create and set a new APG context.

    Args:
        enabled: Whether APG is enabled
        eta: Parallel component weight (1.0 = standard CFG)
        momentum: Buffer momentum (-0.5 recommended)
        threshold: Norm threshold (0 = disabled)

    Returns:
        The created APGContext
    """
    context = APGContext(
        enabled=enabled,
        eta=eta,
        momentum=momentum,
        threshold=threshold,
    )
    set_apg_context(context)
    return context
