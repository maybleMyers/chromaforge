"""
RamTorch Optimizer Integration for ChromaForge

Provides ZeRO-1 optimizer state sharding and integration with existing ChromaForge optimizers.
"""

import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Union, Iterator
from torch.optim import Optimizer

try:
    from backend.ramtorch_integration import ChromaZeROOptimizer
    from RamTorch.ramtorch.zero1 import create_zero_param_groups, broadcast_zero_params
    RAMTORCH_AVAILABLE = True
except ImportError:
    RAMTORCH_AVAILABLE = False
    ChromaZeROOptimizer = None


class RamTorchOptimizerWrapper:
    """
    Wrapper that adds RamTorch ZeRO-1 functionality to any PyTorch optimizer.

    This allows using RamTorch's distributed optimizer state sharding with
    existing optimizers like AdamW, SGD, etc.
    """

    def __init__(self, optimizer: Optimizer, model_parameters: Iterator[torch.nn.Parameter],
                 enable_zero: bool = True, enable_checkpointing: bool = False):
        """
        Initialize RamTorch optimizer wrapper.

        Args:
            optimizer: Base PyTorch optimizer to wrap
            model_parameters: Iterator of model parameters
            enable_zero: Enable ZeRO-1 optimizer state sharding
            enable_checkpointing: Enable gradient checkpointing (future feature)
        """
        if not RAMTORCH_AVAILABLE:
            raise ImportError("RamTorch not available - cannot create RamTorch optimizer")

        self.base_optimizer = optimizer
        self.model_params = list(model_parameters)
        self.enable_zero = enable_zero
        self.enable_checkpointing = enable_checkpointing

        # ZeRO-1 state
        self.zero_optimizer = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1

        # Initialize distributed training if available
        self._setup_distributed()

        # Initialize ZeRO-1 if enabled and distributed
        if self.enable_zero and self.is_distributed:
            self._setup_zero()

        print(f"RamTorchOptimizer initialized:")
        print(f"  Base optimizer: {type(self.base_optimizer).__name__}")
        print(f"  Parameters: {len(self.model_params)}")
        print(f"  ZeRO-1 enabled: {self.enable_zero and self.is_distributed}")
        print(f"  Distributed: {self.is_distributed} (rank {self.rank}/{self.world_size})")

    def _setup_distributed(self):
        """Setup distributed training environment"""
        if dist.is_available() and dist.is_initialized():
            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(f"Distributed training detected: rank {self.rank}/{self.world_size}")
        else:
            print("Single-node training (distributed not available/initialized)")

    def _setup_zero(self):
        """Setup ZeRO-1 optimizer state sharding"""
        if not self.is_distributed:
            print("Cannot setup ZeRO-1: distributed training not available")
            return

        try:
            self.zero_optimizer = ChromaZeROOptimizer(
                optimizer=self.base_optimizer,
                model_params=self.model_params
            )
            print(f"ZeRO-1 optimizer sharding enabled across {self.world_size} processes")
        except Exception as e:
            print(f"Failed to setup ZeRO-1: {e}")
            self.enable_zero = False

    def step(self, closure: Optional[callable] = None):
        """
        Perform optimizer step with optional ZeRO-1 parameter broadcasting.

        Args:
            closure: Optional closure for optimizer step

        Returns:
            Result from base optimizer step
        """
        if self.zero_optimizer is not None:
            # Use ZeRO-1 optimizer which handles parameter broadcasting
            return self.zero_optimizer.step(closure)
        else:
            # Standard optimizer step
            return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all parameters"""
        if self.zero_optimizer is not None:
            return self.zero_optimizer.zero_grad(set_to_none)
        else:
            return self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict"""
        if self.zero_optimizer is not None:
            return self.zero_optimizer.state_dict()
        else:
            return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict"""
        if self.zero_optimizer is not None:
            return self.zero_optimizer.load_state_dict(state_dict)
        else:
            return self.base_optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Get parameter groups from base optimizer"""
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Set parameter groups on base optimizer"""
        self.base_optimizer.param_groups = value

    def add_param_group(self, param_group: Dict[str, Any]):
        """Add parameter group to base optimizer"""
        return self.base_optimizer.add_param_group(param_group)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for optimizer states"""
        stats = {
            'enabled': True,
            'zero_enabled': self.enable_zero and self.zero_optimizer is not None,
            'distributed': self.is_distributed,
            'rank': self.rank,
            'world_size': self.world_size,
            'parameter_count': len(self.model_params)
        }

        if self.is_distributed and self.enable_zero:
            # Estimate memory savings from ZeRO-1
            # ZeRO-1 distributes optimizer states across processes
            total_optimizer_memory = sum(
                sum(state.numel() * state.element_size() for state in param_state.values() if isinstance(state, torch.Tensor))
                for param_state in self.base_optimizer.state.values()
            )

            local_optimizer_memory = total_optimizer_memory // self.world_size
            memory_savings = total_optimizer_memory - local_optimizer_memory

            stats.update({
                'total_optimizer_memory_mb': total_optimizer_memory / (1024**2),
                'local_optimizer_memory_mb': local_optimizer_memory / (1024**2),
                'memory_savings_mb': memory_savings / (1024**2),
                'memory_savings_percent': (memory_savings / total_optimizer_memory) * 100
            })

        return stats


def create_ramtorch_optimizer(optimizer_class, model_parameters: Iterator[torch.nn.Parameter],
                             enable_zero: bool = None, **optimizer_kwargs) -> Union[RamTorchOptimizerWrapper, Optimizer]:
    """
    Factory function to create RamTorch-enhanced optimizers.

    Args:
        optimizer_class: PyTorch optimizer class (e.g., torch.optim.AdamW)
        model_parameters: Iterator of model parameters
        enable_zero: Enable ZeRO-1 sharding (auto-detect if None)
        **optimizer_kwargs: Keyword arguments for optimizer initialization

    Returns:
        RamTorchOptimizerWrapper if RamTorch is available, otherwise standard optimizer
    """
    # Auto-detect ZeRO enablement
    if enable_zero is None:
        try:
            from backend.args import args
            enable_zero = getattr(args, 'ramtorch_zero_optimizer', False)
        except ImportError:
            enable_zero = False

    # Create base optimizer
    base_optimizer = optimizer_class(model_parameters, **optimizer_kwargs)

    # Wrap with RamTorch if available and requested
    if RAMTORCH_AVAILABLE and enable_zero:
        try:
            return RamTorchOptimizerWrapper(
                optimizer=base_optimizer,
                model_parameters=model_parameters,
                enable_zero=enable_zero
            )
        except Exception as e:
            print(f"Failed to create RamTorch optimizer: {e}")
            print("Falling back to standard optimizer")

    return base_optimizer


# Convenience functions for common optimizers
def create_ramtorch_adamw(model_parameters: Iterator[torch.nn.Parameter],
                         lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                         weight_decay: float = 1e-2, enable_zero: bool = None) -> Union[RamTorchOptimizerWrapper, torch.optim.AdamW]:
    """Create RamTorch-enhanced AdamW optimizer"""
    return create_ramtorch_optimizer(
        torch.optim.AdamW,
        model_parameters,
        enable_zero=enable_zero,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def create_ramtorch_sgd(model_parameters: Iterator[torch.nn.Parameter],
                       lr: float = 1e-3, momentum: float = 0, dampening: float = 0,
                       weight_decay: float = 0, nesterov: bool = False,
                       enable_zero: bool = None) -> Union[RamTorchOptimizerWrapper, torch.optim.SGD]:
    """Create RamTorch-enhanced SGD optimizer"""
    return create_ramtorch_optimizer(
        torch.optim.SGD,
        model_parameters,
        enable_zero=enable_zero,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov
    )


def create_ramtorch_adam(model_parameters: Iterator[torch.nn.Parameter],
                        lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                        weight_decay: float = 0, enable_zero: bool = None) -> Union[RamTorchOptimizerWrapper, torch.optim.Adam]:
    """Create RamTorch-enhanced Adam optimizer"""
    return create_ramtorch_optimizer(
        torch.optim.Adam,
        model_parameters,
        enable_zero=enable_zero,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def is_ramtorch_optimizer(optimizer) -> bool:
    """Check if an optimizer is RamTorch-enhanced"""
    return isinstance(optimizer, RamTorchOptimizerWrapper)


def get_optimizer_memory_stats(optimizer) -> Dict[str, Any]:
    """Get memory statistics from any optimizer (RamTorch or standard)"""
    if is_ramtorch_optimizer(optimizer):
        return optimizer.get_memory_stats()
    else:
        # Basic stats for standard optimizers
        param_count = sum(1 for group in optimizer.param_groups for param in group['params'])
        return {
            'enabled': False,
            'zero_enabled': False,
            'distributed': False,
            'parameter_count': param_count,
            'optimizer_type': type(optimizer).__name__
        }


# Example usage and testing
def test_ramtorch_optimizer():
    """Test RamTorch optimizer functionality"""
    if not RAMTORCH_AVAILABLE:
        print("RamTorch not available for testing")
        return False

    try:
        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )

        # Create RamTorch optimizer
        optimizer = create_ramtorch_adamw(model.parameters(), lr=1e-3, enable_zero=True)

        print(f"Created optimizer: {type(optimizer).__name__}")

        # Test basic functionality
        dummy_input = torch.randn(32, 128)
        output = model(dummy_input)
        loss = output.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✅ RamTorch optimizer test passed")

        # Print memory stats
        stats = get_optimizer_memory_stats(optimizer)
        print(f"Memory stats: {stats}")

        return True

    except Exception as e:
        print(f"❌ RamTorch optimizer test failed: {e}")
        return False


if __name__ == "__main__":
    print("RamTorch Optimizer Integration Test")
    print("=" * 40)
    test_ramtorch_optimizer()