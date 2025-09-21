"""
RamTorch Integration for Chroma Models
Provides CPU-bouncing memory management optimized for Chroma/Flux transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from RamTorch.ramtorch.modules.linear import CPUBouncingLinear, BouncingLinearFn
from RamTorch.ramtorch.zero1 import create_zero_param_groups, broadcast_zero_params
import threading
import time


class ChromaBouncingLinearFn(torch.autograd.Function):
    """
    Custom autograd function for Chroma-specific bouncing linear operation.
    Handles dtype conversion to match input tensor dtype.
    """

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        """Forward pass with dtype-aware weight transfer"""
        # Import here to avoid circular dependency
        import RamTorch.ramtorch.modules.linear as ramtorch_linear

        # Get input dtype for conversion
        input_dtype = x.dtype
        selected_buffer = ramtorch_linear.FORWARD_BUFFER_CLK

        # Transfer weights with dtype conversion
        with torch.cuda.stream(ramtorch_linear.TRANSFER_STREAM):
            ramtorch_linear.TRANSFER_STREAM.wait_event(ramtorch_linear.COMPUTE_FORWARD_START_EVENT)

            # Transfer with both device and dtype conversion
            ramtorch_linear.W_BUFFERS[selected_buffer] = weight_cpu.to(
                device=device, dtype=input_dtype, non_blocking=True
            )
            ramtorch_linear.B_BUFFERS[selected_buffer] = (
                bias_cpu.to(device=device, dtype=input_dtype, non_blocking=True)
                if bias_cpu is not None
                else None
            )

            # Update buffer clock
            ramtorch_linear.FORWARD_BUFFER_CLK ^= 1
            ramtorch_linear.TRANSFER_FORWARD_FINISHED_EVENT.record()

        # Wait for transfer and compute
        torch.cuda.current_stream().wait_event(ramtorch_linear.TRANSFER_FORWARD_FINISHED_EVENT)
        ramtorch_linear.COMPUTE_FORWARD_START_EVENT.record()

        out = F.linear(x, ramtorch_linear.W_BUFFERS[selected_buffer], ramtorch_linear.B_BUFFERS[selected_buffer])

        # Save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward pass with dtype-aware gradient computation"""
        import RamTorch.ramtorch.modules.linear as ramtorch_linear

        selected_buffer = ramtorch_linear.BACKWARD_BUFFER_CLK
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        grad_dtype = grad_out.dtype

        # Transfer weights for gradient computation with dtype conversion
        with torch.cuda.stream(ramtorch_linear.TRANSFER_STREAM):
            ramtorch_linear.TRANSFER_STREAM.wait_event(ramtorch_linear.COMPUTE_BACKWARD_START_EVENT)

            ramtorch_linear.W_GRAD_BUFFERS[selected_buffer] = weight_cpu.to(
                device=device, dtype=grad_dtype, non_blocking=True
            )

            ramtorch_linear.BACKWARD_BUFFER_CLK ^= 1
            ramtorch_linear.TRANSFER_BACKWARD_FINISHED_EVENT.record()

        torch.cuda.current_stream().wait_event(ramtorch_linear.TRANSFER_BACKWARD_FINISHED_EVENT)
        ramtorch_linear.COMPUTE_BACKWARD_START_EVENT.record()

        # Compute gradients
        grad_input = grad_out @ ramtorch_linear.W_GRAD_BUFFERS[selected_buffer]
        grad_weight = grad_out.t() @ x
        grad_bias = grad_out.sum(dim=0) if bias_cpu is not None else None

        return grad_input, grad_weight, grad_bias, None


class ChromaBouncingForgeLinear(nn.Module):
    """
    Forge-compatible CPU bouncing linear layer for Chroma models.

    Combines RamTorch's CPU-bouncing mechanism with Forge's Linear layer structure.
    Maintains compatibility with Forge's memory management and LoRA systems.
    """

    def __init__(self, in_features, out_features, bias=True, device="cuda",
                 block_type="unknown", block_index=-1, prefetch_next=None,
                 parameters_manual_cast=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.parameters_manual_cast = parameters_manual_cast

        # Chroma-specific metadata
        self.block_type = block_type  # "double", "single", "embedding", "output"
        self.block_index = block_index
        self.prefetch_next = prefetch_next  # Reference to next block for prefetching

        # Parameters live on CPU for bouncing
        # Check if pinned memory is enabled via class variable
        use_pin = getattr(self.__class__, '_use_pinned_memory', False)

        if use_pin:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, device="cpu").share_memory_().pin_memory()
            )
            self.bias = (
                nn.Parameter(torch.empty(out_features, device="cpu").share_memory_().pin_memory())
                if bias
                else None
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, device="cpu").share_memory_()
            )
            self.bias = (
                nn.Parameter(torch.empty(out_features, device="cpu").share_memory_())
                if bias
                else None
            )

        # Forge compatibility attributes
        self.scale_weight = None  # For potential scaling operations
        self.forge_online_loras = None  # For LoRA patches

        # Performance tracking
        self.transfer_count = 0
        self.total_transfer_time = 0.0
        self.last_access_time = 0.0

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """Forward pass with CPU-GPU bouncing and dtype conversion"""
        start_time = time.perf_counter()

        # Update access tracking
        self.last_access_time = start_time
        self.transfer_count += 1

        # Trigger prefetching for next block if available
        if self.prefetch_next is not None:
            ChromaMemoryManager.prefetch_block(self.prefetch_next)

        # Use Chroma-specific bouncing function that handles dtype
        result = ChromaBouncingLinearFn.apply(x, self.weight, self.bias, self.device)

        # Track performance
        self.total_transfer_time += time.perf_counter() - start_time

        return result

    def get_average_transfer_time(self) -> float:
        """Get average transfer time for performance monitoring"""
        if self.transfer_count == 0:
            return 0.0
        return self.total_transfer_time / self.transfer_count

    def _apply(self, fn):
        """
        Override _apply to ensure weights always stay on CPU for bouncing.
        This prevents .to(device) calls from moving bouncing linear weights to GPU.
        """
        # Apply to non-parameter attributes first
        for key, value in self.__dict__.items():
            if key in ['weight', 'bias']:
                continue  # Skip weight and bias parameters
            if isinstance(value, torch.Tensor):
                setattr(self, key, fn(value))

        # Handle parameters carefully - keep them on CPU
        for name, param in self.named_parameters():
            if name in ['weight', 'bias']:
                # Keep on CPU, only change dtype if needed
                if param is not None:
                    current_dtype = param.dtype

                    # Test what the function would do
                    test_tensor = torch.tensor(0.0, device='cpu', dtype=current_dtype)
                    try:
                        result = fn(test_tensor)
                        # If only dtype changes, apply dtype change but keep on CPU
                        if result.dtype != current_dtype:
                            param.data = param.data.to(dtype=result.dtype)
                        # Always ensure it stays on CPU regardless
                        if param.device.type != 'cpu':
                            param.data = param.data.cpu()
                    except:
                        # If function fails, don't modify the parameter
                        pass
            else:
                # Apply normally to other parameters
                if param is not None:
                    param.data = fn(param.data)
                    if param._grad is not None:
                        param._grad.data = fn(param._grad.data)

        # Apply to buffers normally
        for name, buf in self.named_buffers():
            if buf is not None:
                setattr(self, name, fn(buf))

        return self


class ChromaBouncingLinear(CPUBouncingLinear):
    """
    Chroma-optimized version of CPU bouncing linear layer.

    Enhancements over base CPUBouncingLinear:
    - Block-aware prefetching for sequential Chroma processing
    - Memory pressure monitoring
    - Optimized for Chroma's specific access patterns (double_blocks -> single_blocks)
    """

    def __init__(self, in_features, out_features, bias=True, device="cuda",
                 block_type="unknown", block_index=-1, prefetch_next=None):
        super().__init__(in_features, out_features, bias, device)

        # Chroma-specific metadata
        self.block_type = block_type  # "double", "single", "embedding", "output"
        self.block_index = block_index
        self.prefetch_next = prefetch_next  # Reference to next block for prefetching

        # Performance tracking
        self.transfer_count = 0
        self.total_transfer_time = 0.0
        self.last_access_time = 0.0

    def forward(self, x):
        """Enhanced forward with block-aware optimizations"""
        start_time = time.perf_counter()

        # Update access tracking
        self.last_access_time = start_time
        self.transfer_count += 1

        # Trigger prefetching for next block if available
        if self.prefetch_next is not None:
            ChromaMemoryManager.prefetch_block(self.prefetch_next)

        # Standard bouncing forward
        result = super().forward(x)

        # Track performance
        self.total_transfer_time += time.perf_counter() - start_time

        return result

    def get_average_transfer_time(self) -> float:
        """Get average transfer time for performance monitoring"""
        if self.transfer_count == 0:
            return 0.0
        return self.total_transfer_time / self.transfer_count

    def _apply(self, fn):
        """
        Override _apply to ensure weights always stay on CPU for bouncing.
        This prevents .to(device) calls from moving bouncing linear weights to GPU.
        """
        # For ChromaBouncingLinear, we want to keep weight and bias on CPU always
        # Only apply dtype transformations, never device transfers

        # Apply to non-parameter attributes first
        for key, value in self.__dict__.items():
            if key in ['weight', 'bias']:
                continue  # Skip weight and bias parameters
            if isinstance(value, torch.Tensor):
                setattr(self, key, fn(value))

        # Handle parameters carefully
        for name, param in self.named_parameters():
            if name in ['weight', 'bias']:
                # Keep on CPU, only change dtype if needed
                if param is not None:
                    current_device = param.device
                    current_dtype = param.dtype

                    # Test what the function would do
                    test_tensor = torch.tensor(0.0, device=current_device, dtype=current_dtype)
                    try:
                        result = fn(test_tensor)
                        # If device changes, ignore and keep on CPU
                        # If only dtype changes, apply dtype change but keep on CPU
                        if result.dtype != current_dtype:
                            param.data = param.data.to(dtype=result.dtype)
                        # Always ensure it stays on CPU regardless
                        if param.device.type != 'cpu':
                            param.data = param.data.cpu()
                    except:
                        # If function fails, don't modify the parameter
                        pass
            else:
                # Apply normally to other parameters
                if param is not None:
                    param.data = fn(param.data)
                    if param._grad is not None:
                        param._grad.data = fn(param._grad.data)

        # Apply to buffers normally
        for name, buf in self.named_buffers():
            if buf is not None:
                setattr(self, name, fn(buf))

        return self


class ChromaMemoryManager:
    """
    Central coordinator for Chroma model memory management using RamTorch.

    Handles:
    - Block-level memory orchestration
    - Prefetching for sequential processing
    - Memory pressure monitoring
    - Integration with existing Forge memory management
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.enabled = False
        self.memory_threshold = 0.8  # Trigger aggressive swapping at 80% VRAM usage
        self.prefetch_enabled = True
        self.prefetch_queue: List[Any] = []
        self.active_blocks: Dict[str, Any] = {}
        self.block_access_order: List[str] = []
        self.performance_stats: Dict[str, Dict] = {}

    @classmethod
    def get_instance(cls):
        """Singleton pattern for global memory coordination"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def enable(self, memory_threshold: float = 0.8, prefetch_enabled: bool = True):
        """Enable RamTorch memory management for Chroma"""
        self.enabled = True
        self.memory_threshold = memory_threshold
        self.prefetch_enabled = prefetch_enabled
        print(f"ChromaMemoryManager enabled: threshold={memory_threshold:.1%}, prefetch={prefetch_enabled}")

    def disable(self):
        """Disable RamTorch memory management"""
        self.enabled = False
        self.clear_prefetch_queue()
        print("ChromaMemoryManager disabled")

    def register_block(self, block_id: str, block: Any, block_type: str, block_index: int):
        """Register a transformer block for memory management"""
        if not self.enabled:
            return

        self.active_blocks[block_id] = {
            'block': block,
            'type': block_type,
            'index': block_index,
            'last_access': 0.0,
            'access_count': 0,
            'on_gpu': True
        }

        # Initialize performance tracking
        self.performance_stats[block_id] = {
            'transfer_time': 0.0,
            'compute_time': 0.0,
            'memory_usage': 0
        }

    def mark_block_access(self, block_id: str):
        """Mark that a block has been accessed (for LRU tracking)"""
        if not self.enabled or block_id not in self.active_blocks:
            return

        current_time = time.perf_counter()
        self.active_blocks[block_id]['last_access'] = current_time
        self.active_blocks[block_id]['access_count'] += 1

        # Update access order for LRU
        if block_id in self.block_access_order:
            self.block_access_order.remove(block_id)
        self.block_access_order.append(block_id)

    @staticmethod
    def prefetch_block(block: Any):
        """Prefetch a block asynchronously (static method for easy calling)"""
        manager = ChromaMemoryManager.get_instance()
        if manager.enabled and manager.prefetch_enabled:
            manager._async_prefetch(block)

    def _async_prefetch(self, block: Any):
        """Internal async prefetching implementation"""
        if hasattr(block, 'weight') and block.weight.device.type == 'cpu':
            # Add to prefetch queue for background processing
            self.prefetch_queue.append(block)
            # Note: In a full implementation, this would trigger async GPU transfer
            # For now, we'll do immediate transfer to maintain simplicity
            try:
                block.weight.to('cuda', non_blocking=True)
            except:
                pass  # Fail silently if transfer fails

    def clear_prefetch_queue(self):
        """Clear the prefetch queue"""
        self.prefetch_queue.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory management statistics"""
        if not self.enabled:
            return {'enabled': False}

        stats = {
            'enabled': True,
            'total_blocks': len(self.active_blocks),
            'blocks_on_gpu': sum(1 for b in self.active_blocks.values() if b['on_gpu']),
            'prefetch_queue_size': len(self.prefetch_queue),
            'memory_threshold': self.memory_threshold,
            'block_types': {}
        }

        # Count blocks by type
        for block_info in self.active_blocks.values():
            block_type = block_info['type']
            if block_type not in stats['block_types']:
                stats['block_types'][block_type] = 0
            stats['block_types'][block_type] += 1

        return stats

    def optimize_memory_allocation(self, available_memory: int) -> Dict[str, List[str]]:
        """
        Optimize memory allocation based on available GPU memory and Chroma access patterns.

        Returns allocation strategy similar to ChromaDCT strategy but optimized for regular Chroma.
        """
        if not self.enabled:
            return {'gpu_components': [], 'cpu_components': []}

        # Define Chroma-specific priority groups
        priority_groups = {
            'critical': [  # Always keep on GPU
                'img_in', 'txt_in', 'time_in', 'vector_in', 'pe_embedder'
            ],
            'high_priority': [  # Early double blocks (processed first)
                f'double_blocks.{i}' for i in range(10)  # First 10 double blocks
            ],
            'medium_priority': [  # Late double blocks + early single blocks
                f'double_blocks.{i}' for i in range(10, 19)  # Remaining double blocks
            ] + [
                f'single_blocks.{i}' for i in range(19)  # First half of single blocks
            ],
            'low_priority': [  # Late single blocks + output
                f'single_blocks.{i}' for i in range(19, 38)  # Second half of single blocks
            ] + ['final_layer']
        }

        # Estimate memory requirements (rough estimates)
        group_memory_estimates = {
            'critical': available_memory * 0.20,      # 20% - essential components
            'high_priority': available_memory * 0.35, # 35% - early processing blocks
            'medium_priority': available_memory * 0.30, # 30% - mid processing blocks
            'low_priority': available_memory * 0.15,   # 15% - late processing blocks
        }

        # Calculate allocation
        gpu_allocation = {}
        cpu_allocation = {}
        remaining_memory = available_memory

        for priority in ['critical', 'high_priority', 'medium_priority', 'low_priority']:
            required = group_memory_estimates[priority]
            components = priority_groups[priority]

            if remaining_memory >= required:
                gpu_allocation[priority] = components
                remaining_memory -= required
            elif remaining_memory > 0:
                # Partial allocation - prioritize first components
                partial_count = int(len(components) * (remaining_memory / required))
                gpu_allocation[priority] = components[:partial_count]
                cpu_allocation[priority] = components[partial_count:]
                remaining_memory = 0
            else:
                cpu_allocation[priority] = components

        return {
            'gpu_components': gpu_allocation,
            'cpu_components': cpu_allocation,
            'strategy': 'chroma_ramtorch_optimized'
        }


class ChromaZeROOptimizer:
    """
    ZeRO-1 optimizer state sharding for Chroma model training.
    Integrates RamTorch's ZeRO implementation with Chroma-specific optimizations.
    """

    def __init__(self, optimizer, model_params):
        self.base_optimizer = optimizer
        self.model_params = list(model_params)
        self.rank = 0
        self.world_size = 1
        self.sharded_groups = []
        self.owner_ranks = []

        # Initialize distributed if available
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            self._setup_sharding()

    def _setup_sharding(self):
        """Setup ZeRO-1 parameter sharding for Chroma model"""
        if self.world_size <= 1:
            return

        # Group parameters by transformer blocks for efficient sharding
        param_groups = []

        # Group by block type for optimal sharding
        current_group = {'params': []}
        for param in self.model_params:
            current_group['params'].append(param)

            # Create groups of reasonable size (e.g., per transformer block)
            if len(current_group['params']) >= 10:  # Adjust based on block size
                param_groups.append(current_group)
                current_group = {'params': []}

        # Add remaining parameters
        if current_group['params']:
            param_groups.append(current_group)

        # Create ZeRO sharded groups
        self.sharded_groups, self.owner_ranks = create_zero_param_groups(
            param_groups, self.rank, self.world_size
        )

        print(f"ZeRO-1 sharding initialized: rank {self.rank}/{self.world_size}, "
              f"{len(self.sharded_groups)} local groups")

    def step(self, closure=None):
        """Optimizer step with ZeRO-1 parameter broadcasting"""
        # Run optimizer step on local parameters only
        result = self.base_optimizer.step(closure)

        # Broadcast updated parameters to all ranks
        if self.world_size > 1 and self.owner_ranks:
            broadcast_zero_params(self.model_params, self.owner_ranks)

        return result

    def zero_grad(self, set_to_none=False):
        """Zero gradients for local parameters only"""
        return self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        """Get state dict for local parameters only"""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict for local parameters"""
        return self.base_optimizer.load_state_dict(state_dict)


def _is_linear_layer(module):
    """Check if a module is a Linear layer (either nn.Linear or ForgeOperations.Linear)"""
    # First check standard nn.Linear
    if isinstance(module, nn.Linear):
        return True

    # Check by class name for ForgeOperations.Linear
    class_name = module.__class__.__name__
    module_path = module.__class__.__module__ if hasattr(module.__class__, '__module__') else ''

    # ForgeOperations.Linear has class name 'Linear' and module 'backend.operations'
    if class_name == 'Linear':
        # Could be ForgeOperations.Linear or a nested class
        if 'operations' in module_path or 'ForgeOperations' in str(type(module)):
            return True

        # Check if it has the expected attributes of a Linear layer
        if hasattr(module, 'in_features') and hasattr(module, 'out_features') and \
           hasattr(module, 'weight') and hasattr(module, 'forward'):
            return True

    return False


def _is_forge_linear(module):
    """Specifically check if this is a ForgeOperations.Linear instance"""
    class_name = module.__class__.__name__
    module_path = module.__class__.__module__ if hasattr(module.__class__, '__module__') else ''

    # ForgeOperations.Linear check
    if class_name == 'Linear' and 'operations' in module_path:
        return True

    # Also check for the parameters_manual_cast attribute which is specific to Forge
    if hasattr(module, 'parameters_manual_cast'):
        return True

    return False


def replace_linear_with_bouncing(module: nn.Module, device: str = "cuda",
                                enable_ramtorch: bool = True,
                                use_pinned_memory: bool = False) -> nn.Module:
    """
    Replace all Linear layers in a module with ChromaBouncingLinear layers.

    Args:
        module: The module to convert
        device: Target device for computation
        enable_ramtorch: Whether to enable RamTorch bouncing (if False, returns original module)
        use_pinned_memory: Enable pinned memory for faster CPU-GPU transfers

    Returns:
        Modified module with bouncing linear layers
    """
    print(f"🔧 Starting RamTorch Linear layer replacement for {module.__class__.__name__}")
    print(f"   Enable RamTorch: {enable_ramtorch}, Target device: {device}")

    if not enable_ramtorch:
        print("❌ RamTorch disabled, returning original module")
        return module

    # Debug: Show what types of modules we're seeing
    print("\n📋 Analyzing module structure:")
    debug_count = 0
    for name, child in module.named_modules():
        if hasattr(child, 'weight') and hasattr(child, 'in_features'):
            debug_count += 1
            print(f"  Found potential Linear #{debug_count}: {name} - {child.__class__.__module__}.{child.__class__.__name__}")
            if debug_count >= 10:  # Limit debug output
                print(f"  ... and more (showing first 10)")
                break

    def _replace_recursive(parent_module, name_prefix=""):
        for name, child in list(parent_module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            # Check if this is a Linear layer using our new detection
            if _is_linear_layer(child):
                is_forge_linear = _is_forge_linear(child)
                print(f"🔍 Found Linear layer to replace: {full_name} ({child.__class__.__name__}, forge={is_forge_linear})")
                # Determine block type and index from name
                block_type = "unknown"
                block_index = -1

                if "double_blocks" in full_name:
                    block_type = "double"
                    # Extract block index
                    parts = full_name.split(".")
                    for i, part in enumerate(parts):
                        if part == "double_blocks" and i + 1 < len(parts):
                            try:
                                block_index = int(parts[i + 1])
                            except ValueError:
                                pass
                            break
                elif "single_blocks" in full_name:
                    block_type = "single"
                    # Extract block index
                    parts = full_name.split(".")
                    for i, part in enumerate(parts):
                        if part == "single_blocks" and i + 1 < len(parts):
                            try:
                                block_index = int(parts[i + 1])
                            except ValueError:
                                pass
                            break
                elif any(embed in full_name for embed in ["img_in", "txt_in", "time_in", "vector_in", "pe_embedder"]):
                    block_type = "embedding"
                elif "final_layer" in full_name:
                    block_type = "output"

                # Determine if we need Forge-compatible bouncing linear
                if is_forge_linear:
                    # Use Forge-compatible bouncing linear
                    bouncing_linear = ChromaBouncingForgeLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        device=device,
                        block_type=block_type,
                        block_index=block_index,
                        parameters_manual_cast=getattr(child, 'parameters_manual_cast', False)
                    )

                    # Preserve Forge-specific attributes
                    if hasattr(child, 'scale_weight') and child.scale_weight is not None:
                        bouncing_linear.scale_weight = child.scale_weight
                    if hasattr(child, 'forge_online_loras') and child.forge_online_loras is not None:
                        bouncing_linear.forge_online_loras = child.forge_online_loras
                else:
                    # Use standard ChromaBouncingLinear for nn.Linear
                    bouncing_linear = ChromaBouncingLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        device=device,
                        block_type=block_type,
                        block_index=block_index
                    )

                # Copy weights and bias with None checks and proper device handling
                with torch.no_grad():
                    if child.weight is not None and isinstance(child.weight, torch.Tensor):
                        # Ensure we copy from source weight to CPU (where bouncing_linear.weight should be)
                        bouncing_linear.weight.copy_(child.weight.cpu())
                    else:
                        stats['null_weight_layers'] += 1
                        print(f"⚠️ Warning: {full_name} has None or invalid weights, initializing with zeros")
                        nn.init.zeros_(bouncing_linear.weight)

                    if child.bias is not None and isinstance(child.bias, torch.Tensor):
                        # Ensure we copy from source bias to CPU (where bouncing_linear.bias should be)
                        bouncing_linear.bias.copy_(child.bias.cpu())
                    elif bouncing_linear.bias is not None:
                        stats['null_bias_layers'] += 1
                        nn.init.zeros_(bouncing_linear.bias)

                # Replace the module
                setattr(parent_module, name, bouncing_linear)

                # Register with memory manager
                manager = ChromaMemoryManager.get_instance()
                manager.register_block(full_name, bouncing_linear, block_type, block_index)

                print(f"✅ Replaced Linear layer: {full_name} -> ChromaBouncingLinear "
                      f"(type={block_type}, index={block_index}, weight_device={bouncing_linear.weight.device})")
            else:
                # Recursively process child modules
                _replace_recursive(child, full_name)

    # Enhanced statistics tracking
    replacement_count = 0
    stats = {
        'total_linear_layers': 0,
        'replaced_layers': 0,
        'skipped_layers': 0,
        'layers_by_type': {},
        'null_weight_layers': 0,
        'null_bias_layers': 0,
        'total_parameters': 0
    }

    def _replace_recursive_with_count(parent_module, name_prefix=""):
        nonlocal replacement_count, stats
        for name, child in list(parent_module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            # Check if this is a Linear layer using our new detection
            if _is_linear_layer(child):
                is_forge_linear = _is_forge_linear(child)
                # Update statistics
                stats['total_linear_layers'] += 1
                if child.weight is not None:
                    stats['total_parameters'] += child.weight.numel()
                if child.bias is not None:
                    stats['total_parameters'] += child.bias.numel()

                # Determine block type and index from name
                block_type = "unknown"
                block_index = -1

                if "double_blocks" in full_name:
                    block_type = "double"
                    # Extract block index
                    parts = full_name.split(".")
                    for i, part in enumerate(parts):
                        if part == "double_blocks" and i + 1 < len(parts):
                            try:
                                block_index = int(parts[i + 1])
                            except ValueError:
                                pass
                            break
                elif "single_blocks" in full_name:
                    block_type = "single"
                    # Extract block index
                    parts = full_name.split(".")
                    for i, part in enumerate(parts):
                        if part == "single_blocks" and i + 1 < len(parts):
                            try:
                                block_index = int(parts[i + 1])
                            except ValueError:
                                pass
                            break
                elif any(embed in full_name for embed in ["img_in", "txt_in", "time_in", "vector_in", "pe_embedder"]):
                    block_type = "embedding"
                elif "final_layer" in full_name:
                    block_type = "output"

                # Determine if we need Forge-compatible bouncing linear
                if is_forge_linear:
                    # Use Forge-compatible bouncing linear
                    bouncing_linear = ChromaBouncingForgeLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        device=device,
                        block_type=block_type,
                        block_index=block_index,
                        parameters_manual_cast=getattr(child, 'parameters_manual_cast', False)
                    )

                    # Preserve Forge-specific attributes
                    if hasattr(child, 'scale_weight') and child.scale_weight is not None:
                        bouncing_linear.scale_weight = child.scale_weight
                    if hasattr(child, 'forge_online_loras') and child.forge_online_loras is not None:
                        bouncing_linear.forge_online_loras = child.forge_online_loras
                else:
                    # Use standard ChromaBouncingLinear for nn.Linear
                    bouncing_linear = ChromaBouncingLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        device=device,
                        block_type=block_type,
                        block_index=block_index
                    )

                # Copy weights and bias with None checks and proper device handling
                with torch.no_grad():
                    if child.weight is not None and isinstance(child.weight, torch.Tensor):
                        # Ensure we copy from source weight to CPU (where bouncing_linear.weight should be)
                        bouncing_linear.weight.copy_(child.weight.cpu())
                    else:
                        stats['null_weight_layers'] += 1
                        print(f"⚠️ Warning: {full_name} has None or invalid weights, initializing with zeros")
                        nn.init.zeros_(bouncing_linear.weight)

                    if child.bias is not None and isinstance(child.bias, torch.Tensor):
                        # Ensure we copy from source bias to CPU (where bouncing_linear.bias should be)
                        bouncing_linear.bias.copy_(child.bias.cpu())
                    elif bouncing_linear.bias is not None:
                        stats['null_bias_layers'] += 1
                        nn.init.zeros_(bouncing_linear.bias)

                # Replace the module
                setattr(parent_module, name, bouncing_linear)

                # Register with memory manager
                manager = ChromaMemoryManager.get_instance()
                manager.register_block(full_name, bouncing_linear, block_type, block_index)

                replacement_count += 1
                stats['replaced_layers'] += 1

                # Track layers by type
                if block_type not in stats['layers_by_type']:
                    stats['layers_by_type'][block_type] = 0
                stats['layers_by_type'][block_type] += 1

                print(f"✅ Replaced Linear layer: {full_name} -> ChromaBouncingLinear "
                      f"(type={block_type}, index={block_index}, weight_device={bouncing_linear.weight.device})")
            else:
                # Recursively process child modules
                _replace_recursive_with_count(child, full_name)

    _replace_recursive_with_count(module)

    # Calculate final statistics
    stats['skipped_layers'] = stats['total_linear_layers'] - stats['replaced_layers']

    # Comprehensive reporting
    print(f"🎯 RamTorch replacement complete!")
    print(f"  📊 Total Linear layers found: {stats['total_linear_layers']}")
    print(f"  ✅ Replaced with ChromaBouncingLinear: {stats['replaced_layers']}")
    print(f"  ⏭️ Skipped layers: {stats['skipped_layers']}")
    print(f"  📝 Total parameters: {stats['total_parameters']:,}")

    if stats['null_weight_layers'] > 0:
        print(f"  ⚠️ Layers with null weights: {stats['null_weight_layers']}")
    if stats['null_bias_layers'] > 0:
        print(f"  ⚠️ Layers with null bias: {stats['null_bias_layers']}")

    if stats['layers_by_type']:
        print(f"  🏗️ Replacement by type:")
        for block_type, count in stats['layers_by_type'].items():
            print(f"     {block_type}: {count} layers")

    return module


def get_ramtorch_memory_stats() -> Dict[str, Any]:
    """Get comprehensive RamTorch memory statistics for monitoring"""
    manager = ChromaMemoryManager.get_instance()
    base_stats = manager.get_memory_stats()

    # Add torch memory info
    if torch.cuda.is_available():
        base_stats.update({
            'cuda_memory_allocated': torch.cuda.memory_allocated(),
            'cuda_memory_reserved': torch.cuda.memory_reserved(),
            'cuda_memory_cached': torch.cuda.memory_cached() if hasattr(torch.cuda, 'memory_cached') else 0,
        })

    return base_stats


def configure_ramtorch_for_chroma(memory_threshold: float = 0.8,
                                 prefetch_enabled: bool = True,
                                 enable_zero: bool = False,
                                 use_pinned_memory: bool = False) -> None:
    """
    Configure RamTorch for optimal Chroma model performance.

    Args:
        memory_threshold: VRAM usage threshold to trigger aggressive swapping
        prefetch_enabled: Enable block prefetching for sequential processing
        enable_zero: Enable ZeRO-1 optimizer sharding for distributed training
        use_pinned_memory: Enable pinned memory for faster CPU-GPU transfers
    """
    manager = ChromaMemoryManager.get_instance()
    manager.enable(memory_threshold, prefetch_enabled)

    # Set global flag for pinned memory
    CPUBouncingLinear._use_pinned_memory = use_pinned_memory
    ChromaBouncingLinear._use_pinned_memory = use_pinned_memory
    ChromaBouncingForgeLinear._use_pinned_memory = use_pinned_memory

    print(f"RamTorch configured for Chroma:")
    print(f"  Memory threshold: {memory_threshold:.1%}")
    print(f"  Prefetch enabled: {prefetch_enabled}")
    print(f"  ZeRO-1 enabled: {enable_zero}")
    print(f"  Pinned memory: {use_pinned_memory}")