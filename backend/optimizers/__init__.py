"""
ChromaForge Optimizer Extensions

This module provides enhanced optimizers with RamTorch integration for memory-efficient training.
"""

try:
    from .ramtorch_optimizer import (
        RamTorchOptimizerWrapper,
        create_ramtorch_optimizer,
        create_ramtorch_adamw,
        create_ramtorch_sgd,
        create_ramtorch_adam,
        is_ramtorch_optimizer,
        get_optimizer_memory_stats
    )
    RAMTORCH_OPTIMIZERS_AVAILABLE = True
except ImportError:
    RAMTORCH_OPTIMIZERS_AVAILABLE = False
    RamTorchOptimizerWrapper = None

__all__ = [
    'RamTorchOptimizerWrapper',
    'create_ramtorch_optimizer',
    'create_ramtorch_adamw',
    'create_ramtorch_sgd',
    'create_ramtorch_adam',
    'is_ramtorch_optimizer',
    'get_optimizer_memory_stats',
    'RAMTORCH_OPTIMIZERS_AVAILABLE'
]