"""
Latent Guidance System

Provides latent guidance with weight scheduling and momentum for steering
diffusion sampling toward target latents. Adapted from RES4LYF LatentGuide.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
import math


class SimpleLatentGuide:
    """
    Simplified latent guidance with weight scheduling and momentum.

    Steers the sampling process toward a target latent using configurable
    scheduling and momentum for smooth guidance application.
    """

    def __init__(
        self,
        target_latent: torch.Tensor,
        weight: float = 0.5,
        start_step: float = 0.0,
        end_step: float = 1.0,
        momentum: float = 0.0,
        cutoff: float = 1.0,
        scheduler: str = 'linear',
    ):
        """
        Initialize the latent guide.

        Args:
            target_latent: The target latent to guide toward
            weight: Base guidance weight (0-1)
            start_step: Normalized step to start guidance (0-1)
            end_step: Normalized step to end guidance (0-1)
            momentum: Momentum for smoothing corrections (0-1)
            cutoff: Cosine similarity cutoff (stop guiding when similarity > cutoff)
            scheduler: Weight scheduling type ('linear', 'cosine', 'constant')
        """
        self.target = target_latent
        self.weight = weight
        self.start_step = start_step
        self.end_step = end_step
        self.momentum = momentum
        self.cutoff = cutoff
        self.scheduler = scheduler
        self.prev_correction = None

    def get_scheduled_weight(self, step_ratio: float) -> float:
        """Calculate weight based on scheduler type and current step."""
        if step_ratio < self.start_step or step_ratio > self.end_step:
            return 0.0

        t = (step_ratio - self.start_step) / (self.end_step - self.start_step + 1e-8)

        if self.scheduler == 'constant':
            return self.weight
        elif self.scheduler == 'cosine':
            return self.weight * (1 + math.cos(t * math.pi)) / 2
        else:
            return self.weight * (1 - t)

    def compute_similarity(self, x: torch.Tensor) -> float:
        """Compute cosine similarity between x and target."""
        x_flat = x.flatten()
        t_flat = self.target.flatten()
        return F.cosine_similarity(
            x_flat.unsqueeze(0),
            t_flat.unsqueeze(0)
        ).item()

    def apply(self, x: torch.Tensor, step_ratio: float) -> torch.Tensor:
        """Apply guidance correction to latent."""
        current_weight = self.get_scheduled_weight(step_ratio)
        if current_weight == 0:
            return x

        if self.cutoff < 1.0:
            sim = self.compute_similarity(x)
            if sim > self.cutoff:
                return x

        correction = (self.target - x) * current_weight

        if self.momentum > 0 and self.prev_correction is not None:
            correction = correction * (1 - self.momentum) + self.prev_correction * self.momentum

        self.prev_correction = correction.clone()
        return x + correction

    def reset(self):
        """Reset momentum state."""
        self.prev_correction = None


class GuidanceManager:
    """Manages multiple guides during sampling."""

    def __init__(self):
        self.guides: List[SimpleLatentGuide] = []

    def add_guide(self, guide: SimpleLatentGuide):
        """Add a guide to the manager."""
        self.guides.append(guide)

    def clear(self):
        """Clear all guides."""
        self.guides = []

    def reset_all(self):
        """Reset momentum state for all guides."""
        for guide in self.guides:
            guide.reset()

    def apply_all(self, x: torch.Tensor, step_ratio: float) -> torch.Tensor:
        """Apply all guides sequentially."""
        for guide in self.guides:
            x = guide.apply(x, step_ratio)
        return x


class LatentBlender:
    """
    Blends latents during sampling using various interpolation methods.
    """

    @staticmethod
    def linear_blend(x: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        """Simple linear interpolation."""
        return x * (1 - weight) + target * weight

    @staticmethod
    def slerp_blend(x: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        """Spherical linear interpolation."""
        x_norm = F.normalize(x.flatten(), dim=0)
        t_norm = F.normalize(target.flatten(), dim=0)

        dot = torch.dot(x_norm, t_norm).clamp(-1, 1)
        theta = torch.acos(dot)

        if theta.abs() < 1e-6:
            return x * (1 - weight) + target * weight

        sin_theta = torch.sin(theta)
        a = torch.sin((1 - weight) * theta) / sin_theta
        b = torch.sin(weight * theta) / sin_theta

        result = a * x + b * target
        return result

    @staticmethod
    def normalized_blend(x: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        """Blend with magnitude preservation."""
        orig_norm = x.norm()
        blended = x * (1 - weight) + target * weight
        return blended * (orig_norm / (blended.norm() + 1e-8))


def create_guidance_callback(
    guides: List[SimpleLatentGuide],
    total_steps: int
):
    """
    Create a callback function that applies guidance during sampling.

    Returns a callback compatible with k-diffusion samplers.
    """
    manager = GuidanceManager()
    for guide in guides:
        manager.add_guide(guide)

    def guidance_callback(d):
        step = d['i']
        x = d['x']
        step_ratio = step / max(total_steps - 1, 1)
        d['x'] = manager.apply_all(x, step_ratio)

    return guidance_callback
