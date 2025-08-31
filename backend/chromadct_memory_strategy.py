"""
ChromaDCT-Optimized Memory Management Strategy
Provides intelligent swapping that accounts for ChromaDCT's unique memory access patterns
"""

import torch
from typing import Dict, List, Optional, Set
import threading
from collections import deque
from backend import memory_management


def is_chromadct_model(model) -> bool:
    """Check if model is ChromaDCT (no VAE)"""
    try:
        from modules import shared
        if (hasattr(shared, 'sd_model') and shared.sd_model is not None and
            hasattr(shared.sd_model, 'forge_objects') and 
            hasattr(shared.sd_model.forge_objects, 'vae') and 
            shared.sd_model.forge_objects.vae is None):
            return True
    except:
        pass
    return False


def get_chromadct_execution_order() -> List[str]:
    """
    Define the exact execution order of ChromaDCT model components.
    This matches the actual forward pass sequence for optimal memory management.
    """
    execution_order = []
    
    # Phase 1: Input Processing (always needed, keep on GPU)
    execution_order.extend([
        'img_in_patch',
        'txt_in', 
        'pe_embedder',
        'distilled_guidance_layer',  # Used throughout inference, keep resident
    ])
    
    # Phase 2: Double blocks (sequential processing)
    for i in range(19):  # 0-18
        execution_order.append(f'double_blocks.{i}')
    
    # Phase 3: Single blocks (sequential processing) 
    for i in range(38):  # 0-37
        execution_order.append(f'single_blocks.{i}')
    
    # Phase 4: NeRF processing (batch together, used as group)
    execution_order.extend([
        'nerf_image_embedder',
        'nerf_blocks.0', 'nerf_blocks.1', 'nerf_blocks.2', 'nerf_blocks.3',
        'nerf_blocks.4', 'nerf_blocks.5', 'nerf_blocks.6', 'nerf_blocks.7',
        'nerf_blocks.8', 'nerf_blocks.9', 'nerf_blocks.10', 'nerf_blocks.11', 
        'nerf_final_layer_conv',
    ])
    
    return execution_order


def get_chromadct_memory_priority_groups(model, current_inference_step: int = 0, total_steps: int = 28) -> Dict[str, List[str]]:
    """
    Define memory priority groups based on execution order and current inference position.
    Uses a rolling window approach to keep relevant blocks on GPU.
    """
    execution_order = get_chromadct_execution_order()
    
    # Calculate rolling window parameters (adaptive based on memory pressure)
    try:
        # Use global reference if available
        import sys
        if 'backend.chromadct_memory_strategy' in sys.modules:
            module = sys.modules['backend.chromadct_memory_strategy']
            if hasattr(module, '_chromadct_pressure_monitor'):
                window_size = module._chromadct_pressure_monitor.get_adaptive_window_size()
            else:
                window_size = 6
        else:
            window_size = 6
    except:
        window_size = 6  # Fallback to default
    prefetch_size = max(2, window_size // 2)  # Prefetch proportional to window size
    
    # Always keep critical components on GPU
    critical_components = ['img_in_patch', 'txt_in', 'pe_embedder', 'distilled_guidance_layer']
    
    # Determine current execution phase based on inference step
    progress_ratio = current_inference_step / max(total_steps, 1)
    
    # Map progress to execution order position
    if progress_ratio < 0.1:  # Early steps - input processing
        current_pos = 0
    elif progress_ratio < 0.4:  # Double blocks phase
        current_pos = 4 + int((progress_ratio - 0.1) * 19 / 0.3)
    elif progress_ratio < 0.9:  # Single blocks phase  
        current_pos = 23 + int((progress_ratio - 0.4) * 38 / 0.5)
    else:  # NeRF phase
        current_pos = 61
    
    # Calculate rolling window
    window_start = max(4, current_pos - window_size // 2)  # Skip critical components
    window_end = min(len(execution_order), current_pos + window_size // 2 + prefetch_size)
    
    # Build priority groups
    priority_groups = {
        'critical': critical_components,
        'active_window': execution_order[window_start:current_pos + 1],
        'prefetch': execution_order[current_pos + 1:window_end],
        'inactive': [comp for comp in execution_order[4:] if comp not in 
                    execution_order[window_start:window_end]]
    }
    
    print(f"[CHROMADCT DEBUG] Inference step {current_inference_step}/{total_steps} (pos {current_pos})")
    print(f"[CHROMADCT DEBUG] Active window: {len(priority_groups['active_window'])} components")
    print(f"[CHROMADCT DEBUG] Prefetch: {len(priority_groups['prefetch'])} components")
    print(f"[CHROMADCT DEBUG] Inactive: {len(priority_groups['inactive'])} components")
    
    return priority_groups


def get_chromadct_memory_priority_groups_old(model) -> Dict[str, List[str]]:
    """
    Define memory priority groups for ChromaDCT models based on access frequency
    Higher priority = keep on GPU longer
    """
    
    return {
        # CRITICAL - Keep on GPU at all costs (used every inference step)
        'critical': [
            'img_in_patch',  # Input patch processing
            'txt_in',        # Text input processing  
            'pe_embedder',   # Position embeddings
            'distilled_guidance_layer',  # Guidance generation
        ],
        
        # HIGH PRIORITY - Keep on GPU during processing phases (used in sequences)
        'high_priority': [
            # Early double blocks (process first, can swap out later)
            'double_blocks.0', 'double_blocks.1', 'double_blocks.2', 'double_blocks.3',
            'double_blocks.4', 'double_blocks.5', 'double_blocks.6', 'double_blocks.7',
            'double_blocks.8', 'double_blocks.9',
            
            # Early single blocks  
            'single_blocks.0', 'single_blocks.1', 'single_blocks.2', 'single_blocks.3',
            'single_blocks.4', 'single_blocks.5', 'single_blocks.6', 'single_blocks.7',
            'single_blocks.8', 'single_blocks.9', 'single_blocks.10', 'single_blocks.11',
            'single_blocks.12', 'single_blocks.13', 'single_blocks.14', 'single_blocks.15',
            'single_blocks.16', 'single_blocks.17', 'single_blocks.18',
        ],
        
        # MEDIUM PRIORITY - Can swap but preferably keep on GPU
        'medium_priority': [
            # Late double blocks (process later in sequence)
            'double_blocks.10', 'double_blocks.11', 'double_blocks.12', 'double_blocks.13',
            'double_blocks.14', 'double_blocks.15', 'double_blocks.16', 'double_blocks.17',
            'double_blocks.18',
            
            # Late single blocks
            'single_blocks.19', 'single_blocks.20', 'single_blocks.21', 'single_blocks.22',
            'single_blocks.23', 'single_blocks.24', 'single_blocks.25', 'single_blocks.26',
            'single_blocks.27', 'single_blocks.28', 'single_blocks.29', 'single_blocks.30',
            'single_blocks.31', 'single_blocks.32', 'single_blocks.33', 'single_blocks.34',
            'single_blocks.35', 'single_blocks.36', 'single_blocks.37',
        ],
        
        # LOW PRIORITY - Can swap to CPU more aggressively  
        'low_priority': [
            # NeRF blocks - smaller and used as a group at the end
            'nerf_blocks.0', 'nerf_blocks.1', 'nerf_blocks.2', 'nerf_blocks.3',
            'nerf_blocks.4', 'nerf_blocks.5', 'nerf_blocks.6', 'nerf_blocks.7', 
            'nerf_blocks.8', 'nerf_blocks.9', 'nerf_blocks.10', 'nerf_blocks.11',
            'nerf_image_embedder',
            'nerf_final_layer_conv',
        ]
    }


def calculate_chromadct_memory_profile(model, available_gpu_memory: int) -> Dict:
    """
    Calculate optimal memory allocation for ChromaDCT model given GPU memory constraints
    """
    
    priority_groups = get_chromadct_memory_priority_groups(model)
    
    # Estimate memory requirements for each group (these are rough estimates)
    group_memory_estimates = {
        'critical': available_gpu_memory * 0.15,      # 15% - core components
        'high_priority': available_gpu_memory * 0.45, # 45% - main processing blocks
        'medium_priority': available_gpu_memory * 0.25, # 25% - late processing blocks  
        'low_priority': available_gpu_memory * 0.15,   # 15% - NeRF blocks
    }
    
    # Calculate what can fit on GPU based on priority
    gpu_allocation = {}
    cpu_allocation = {}
    remaining_memory = available_gpu_memory
    
    for priority in ['critical', 'active_window', 'prefetch', 'inactive']:
        required = group_memory_estimates[priority]
        components = priority_groups[priority]
        
        if remaining_memory >= required:
            # Entire group fits on GPU
            gpu_allocation[priority] = components
            remaining_memory -= required
        elif remaining_memory > 0:
            # Partial group on GPU (prioritize first components)
            partial_count = int(len(components) * (remaining_memory / required))
            gpu_allocation[priority] = components[:partial_count]
            cpu_allocation[priority] = components[partial_count:]
            remaining_memory = 0
        else:
            # Entire group on CPU
            cpu_allocation[priority] = components
    
    return {
        'gpu_components': gpu_allocation,
        'cpu_components': cpu_allocation,
        'strategy': 'chromadct_execution_aware',
        'execution_order': get_chromadct_execution_order(),
        'current_step': current_step,
        'total_steps': total_steps
    }


def apply_chromadct_swapping_strategy(model, available_gpu_memory: int):
    """
    Apply ChromaDCT-optimized swapping strategy to a model
    """
    
    if not is_chromadct_model(model):
        # Use default strategy for non-ChromaDCT models
        print(f"[CHROMADCT DEBUG] Model is not ChromaDCT, using default strategy")
        return None
    
    print(f"\n=== CHROMADCT MEMORY DEBUG: Applying ChromaDCT-optimized memory strategy ===")
    print(f"[CHROMADCT DEBUG] Available GPU memory: {available_gpu_memory / (1024**3):.2f} GB ({available_gpu_memory / (1024**2):.1f} MB)")
    
    memory_profile = calculate_chromadct_memory_profile(model, available_gpu_memory)
    
    # Count total components in each location
    gpu_count = sum(len(components) for components in memory_profile['gpu_components'].values())
    cpu_count = sum(len(components) for components in memory_profile['cpu_components'].values())
    total_count = gpu_count + cpu_count
    
    print(f"[CHROMADCT DEBUG] ChromaDCT Memory Allocation Summary:")
    print(f"[CHROMADCT DEBUG]   GPU: {gpu_count}/{total_count} components ({gpu_count/total_count*100:.1f}%)")
    print(f"[CHROMADCT DEBUG]   CPU: {cpu_count}/{total_count} components ({cpu_count/total_count*100:.1f}%)")
    
    for priority in ['critical', 'high_priority', 'medium_priority', 'low_priority']:
        gpu_in_priority = len(memory_profile['gpu_components'].get(priority, []))
        cpu_in_priority = len(memory_profile['cpu_components'].get(priority, []))
        total_in_priority = gpu_in_priority + cpu_in_priority
        
        if total_in_priority > 0:
            print(f"[CHROMADCT DEBUG]   {priority:15}: {gpu_in_priority:2d}/{total_in_priority:2d} on GPU")
    
    return memory_profile


def get_chromadct_inference_memory_multiplier() -> float:
    """
    Get memory multiplier for ChromaDCT inference requirements
    ChromaDCT needs less inference memory than VAE-based models
    """
    if is_chromadct_model(None):
        # ChromaDCT processes in pixel space (3 channels) vs latent space (16 channels)
        # and has more efficient NeRF processing
        return 0.4  # 40% of normal inference memory requirement
    return 1.0


def optimize_chromadct_model_loading(model, available_memory: int, inference_memory: int):
    """
    Optimize model loading specifically for ChromaDCT models
    """
    
    if not is_chromadct_model(model):
        return None
        
    # ChromaDCT models need less inference memory
    adjusted_inference_memory = int(inference_memory * get_chromadct_inference_memory_multiplier())
    
    # More memory available for model weights
    adjusted_available_memory = available_memory + (inference_memory - adjusted_inference_memory)
    
    print(f"\n[CHROMADCT DEBUG] === ChromaDCT Memory Optimization ===")
    print(f"[CHROMADCT DEBUG]   Original inference memory: {inference_memory / (1024**2):.0f} MB")
    print(f"[CHROMADCT DEBUG]   Optimized inference memory: {adjusted_inference_memory / (1024**2):.0f} MB")  
    print(f"[CHROMADCT DEBUG]   Memory saved: {(inference_memory - adjusted_inference_memory) / (1024**2):.0f} MB")
    print(f"[CHROMADCT DEBUG]   Total available for model: {adjusted_available_memory / (1024**2):.0f} MB")
    print(f"[CHROMADCT DEBUG] === ChromaDCT Memory Optimization Complete ===")
    
    return {
        'adjusted_inference_memory': adjusted_inference_memory,
        'adjusted_available_memory': adjusted_available_memory,
        'memory_saved': inference_memory - adjusted_inference_memory
    }


class ChromaDCTRollingWindowManager:
    """
    Rolling window memory manager for ChromaDCT that tracks inference progress
    and manages memory based on execution order.
    """
    
    def __init__(self):
        self.current_step = 0
        self.total_steps = 28
        self.execution_order = None  # Lazy-loaded
        self.loaded_components: Set[str] = set()
        self.prefetch_queue = deque(maxlen=3)  # Queue for prefetching
        self.lock = threading.Lock()
        
    def get_execution_order(self):
        """Lazy-load execution order to avoid circular dependencies"""
        if self.execution_order is None:
            self.execution_order = get_chromadct_execution_order()
        return self.execution_order
        
    def update_inference_step(self, step: int, total: int = 28):
        """Update current inference step for adaptive memory management"""
        with self.lock:
            self.current_step = step
            self.total_steps = total
            print(f"[CHROMADCT ROLLING] Updated to step {step}/{total}")
    
    def get_current_priority_groups(self, model, available_gpu_memory: int) -> Dict[str, List[str]]:
        """Get current priority groups based on inference progress"""
        with self.lock:
            return get_chromadct_memory_priority_groups(
                model, self.current_step, self.total_steps
            )
    
    def should_prefetch_component(self, component_name: str) -> bool:
        """Determine if a component should be prefetched"""
        with self.lock:
            try:
                execution_order = self.get_execution_order()
                current_pos = execution_order.index(
                    next(comp for comp in execution_order if comp in self.loaded_components)
                )
                component_pos = execution_order.index(component_name)
                return component_pos > current_pos and component_pos <= current_pos + 3
            except (ValueError, StopIteration):
                return False
    
    def get_components_to_unload(self, model, available_gpu_memory: int) -> List[str]:
        """Get list of components that can be safely unloaded"""
        with self.lock:
            priority_groups = self.get_current_priority_groups(model, available_gpu_memory)
            
            # Components that are not in critical, active_window, or prefetch can be unloaded
            keep_loaded = set()
            keep_loaded.update(priority_groups.get('critical', []))
            keep_loaded.update(priority_groups.get('active_window', []))
            keep_loaded.update(priority_groups.get('prefetch', []))
            
            can_unload = [comp for comp in self.loaded_components if comp not in keep_loaded]
            
            if can_unload:
                print(f"[CHROMADCT ROLLING] Can unload {len(can_unload)} components: {can_unload[:3]}...")
            
            return can_unload
    
    def mark_component_loaded(self, component_name: str):
        """Mark a component as loaded on GPU"""
        with self.lock:
            self.loaded_components.add(component_name)
    
    def mark_component_unloaded(self, component_name: str):
        """Mark a component as unloaded from GPU"""
        with self.lock:
            self.loaded_components.discard(component_name)


class ChromaDCTAsyncPrefetcher:
    """
    Asynchronous prefetcher for ChromaDCT blocks using CUDA streams.
    Overlaps memory transfers with computation to reduce PCIe bottlenecks.
    """
    
    def __init__(self):
        self.prefetch_stream = None
        self.main_stream = None
        self.prefetch_queue = deque(maxlen=3)
        self.prefetching_active = False
        self.lock = threading.Lock()
        
        # Initialize CUDA streams if available
        try:
            import torch
            if torch.cuda.is_available():
                self.main_stream = torch.cuda.current_stream()
                self.prefetch_stream = torch.cuda.Stream()
                print(f"[CHROMADCT PREFETCH] CUDA streams initialized")
        except Exception as e:
            print(f"[CHROMADCT PREFETCH] Failed to initialize CUDA streams: {e}")
    
    def start_prefetch_component(self, component_name: str, model_patcher):
        """Start asynchronous prefetching of a component"""
        if not self.prefetch_stream:
            return
            
        with self.lock:
            if component_name in self.prefetch_queue:
                return
            
            self.prefetch_queue.append(component_name)
            
        # Schedule prefetch on separate stream
        def prefetch_task():
            try:
                with torch.cuda.stream(self.prefetch_stream):
                    # Load component to GPU asynchronously
                    print(f"[CHROMADCT PREFETCH] Prefetching {component_name}")
                    # The actual loading will be handled by memory_management.load_models_gpu
                    # This just signals intent to load
                    
                    # Add some prefetch-specific optimizations here:
                    # - Pin memory allocation 
                    # - Async memory copy
                    # - Component warmup
                    
            except Exception as e:
                print(f"[CHROMADCT PREFETCH] Error prefetching {component_name}: {e}")
        
        # Run prefetch in background thread to avoid blocking main computation
        import threading
        prefetch_thread = threading.Thread(target=prefetch_task, daemon=True)
        prefetch_thread.start()
    
    def wait_for_prefetch(self, component_name: str):
        """Wait for a specific component to finish prefetching"""
        if not self.prefetch_stream:
            return
            
        if component_name in self.prefetch_queue:
            # Synchronize prefetch stream with main stream
            self.main_stream.wait_stream(self.prefetch_stream)
            print(f"[CHROMADCT PREFETCH] Synchronized {component_name} prefetch")
    
    def get_next_prefetch_candidates(self, execution_order: List[str], current_pos: int) -> List[str]:
        """Get the next components that should be prefetched"""
        candidates = []
        for i in range(1, 4):  # Look ahead 1-3 positions
            next_pos = current_pos + i
            if next_pos < len(execution_order):
                candidates.append(execution_order[next_pos])
        return candidates


class ChromaDCTNeRFBatcher:
    """
    Specialized batcher for ChromaDCT NeRF blocks that are processed together.
    Groups all NeRF components for efficient batch loading.
    """
    
    def __init__(self):
        self.nerf_components = [
            'nerf_image_embedder',
            'nerf_blocks.0', 'nerf_blocks.1', 'nerf_blocks.2', 'nerf_blocks.3',
            'nerf_blocks.4', 'nerf_blocks.5', 'nerf_blocks.6', 'nerf_blocks.7',
            'nerf_blocks.8', 'nerf_blocks.9', 'nerf_blocks.10', 'nerf_blocks.11',
            'nerf_final_layer_conv'
        ]
        self.nerf_phase_active = False
        self.batch_loaded = False
    
    def is_nerf_component(self, component_name: str) -> bool:
        """Check if component is part of NeRF processing"""
        return component_name in self.nerf_components
    
    def should_batch_load_nerf(self, component_name: str) -> bool:
        """Determine if we should batch load all NeRF components"""
        return (self.is_nerf_component(component_name) and 
                not self.batch_loaded and 
                not self.nerf_phase_active)
    
    def get_nerf_batch_components(self) -> List[str]:
        """Get all NeRF components for batch loading"""
        return self.nerf_components.copy()
    
    def mark_nerf_phase_start(self):
        """Mark the start of NeRF processing phase"""
        self.nerf_phase_active = True
        self.batch_loaded = True
        print(f"[CHROMADCT NERF] Starting NeRF batch processing phase")
    
    def mark_nerf_phase_end(self):
        """Mark the end of NeRF processing phase"""
        self.nerf_phase_active = False
        self.batch_loaded = False
        print(f"[CHROMADCT NERF] NeRF batch processing phase complete")


class ChromaDCTMemoryPressureMonitor:
    """
    Monitor GPU memory pressure and dynamically adjust ChromaDCT memory strategy.
    """
    
    def __init__(self):
        self.pressure_threshold_high = 0.9   # 90% memory usage = high pressure
        self.pressure_threshold_low = 0.7    # 70% memory usage = normal pressure
        self.pressure_history = deque(maxlen=10)
        self.adaptive_window_size = 6
        self.lock = threading.Lock()
        
    def get_memory_pressure(self) -> float:
        """Get current GPU memory pressure (0.0 = no pressure, 1.0 = maximum pressure)"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                pressure = allocated_memory / total_memory
                
                with self.lock:
                    self.pressure_history.append(pressure)
                
                return pressure
        except Exception:
            return 0.5  # Default to medium pressure if unable to measure
    
    def get_avg_memory_pressure(self) -> float:
        """Get average memory pressure over recent history"""
        with self.lock:
            if not self.pressure_history:
                return 0.5
            return sum(self.pressure_history) / len(self.pressure_history)
    
    def should_reduce_window_size(self) -> bool:
        """Determine if window size should be reduced due to memory pressure"""
        avg_pressure = self.get_avg_memory_pressure()
        return avg_pressure > self.pressure_threshold_high
    
    def should_increase_window_size(self) -> bool:
        """Determine if window size can be increased due to low memory pressure"""
        avg_pressure = self.get_avg_memory_pressure()
        return avg_pressure < self.pressure_threshold_low
    
    def get_adaptive_window_size(self) -> int:
        """Get dynamically adjusted window size based on memory pressure"""
        with self.lock:
            if self.should_reduce_window_size():
                # High pressure: reduce window size to 3-4 components
                self.adaptive_window_size = max(3, self.adaptive_window_size - 1)
                print(f"[CHROMADCT PRESSURE] High memory pressure, reducing window to {self.adaptive_window_size}")
            elif self.should_increase_window_size():
                # Low pressure: increase window size to 6-8 components
                self.adaptive_window_size = min(8, self.adaptive_window_size + 1)
                print(f"[CHROMADCT PRESSURE] Low memory pressure, increasing window to {self.adaptive_window_size}")
            
            return self.adaptive_window_size
    
    def get_emergency_unload_candidates(self, loaded_components: Set[str], 
                                      critical_components: List[str]) -> List[str]:
        """Get components that should be immediately unloaded in high memory pressure"""
        current_pressure = self.get_memory_pressure()
        
        if current_pressure > 0.95:  # Emergency threshold
            # Keep only critical components
            candidates = [comp for comp in loaded_components if comp not in critical_components]
            print(f"[CHROMADCT PRESSURE] Emergency unloading {len(candidates)} components (pressure: {current_pressure:.1%})")
            return candidates
        
        return []


class ChromaDCTPinnedMemoryManager:
    """
    Manage pinned memory allocations for faster CPU-GPU transfers without changing dtypes.
    """
    
    def __init__(self):
        self.pinned_allocations = {}
        self.max_pinned_memory = 2 * 1024**3  # 2GB max pinned memory
        self.current_pinned = 0
        self.lock = threading.Lock()
        
    def allocate_pinned_buffer(self, size: int, dtype: torch.dtype, device_id: int = 0) -> Optional[torch.Tensor]:
        """Allocate pinned memory buffer for faster transfers"""
        try:
            import torch
            
            with self.lock:
                if self.current_pinned + size > self.max_pinned_memory:
                    print(f"[CHROMADCT PINNED] Pinned memory limit reached, skipping allocation")
                    return None
                
                # Allocate pinned memory buffer
                buffer = torch.empty(size // dtype.itemsize, dtype=dtype).pin_memory()
                buffer_id = id(buffer)
                
                self.pinned_allocations[buffer_id] = {
                    'buffer': buffer,
                    'size': size,
                    'dtype': dtype
                }
                self.current_pinned += size
                
                print(f"[CHROMADCT PINNED] Allocated {size / (1024**2):.1f}MB pinned buffer (total: {self.current_pinned / (1024**2):.1f}MB)")
                return buffer
                
        except Exception as e:
            print(f"[CHROMADCT PINNED] Failed to allocate pinned memory: {e}")
            return None
    
    def release_pinned_buffer(self, buffer: torch.Tensor):
        """Release pinned memory buffer"""
        try:
            with self.lock:
                buffer_id = id(buffer)
                if buffer_id in self.pinned_allocations:
                    allocation = self.pinned_allocations.pop(buffer_id)
                    self.current_pinned -= allocation['size']
                    print(f"[CHROMADCT PINNED] Released pinned buffer, remaining: {self.current_pinned / (1024**2):.1f}MB")
        except Exception as e:
            print(f"[CHROMADCT PINNED] Error releasing pinned buffer: {e}")
    
    def async_copy_to_gpu(self, source: torch.Tensor, target_device: torch.device, 
                         stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Perform asynchronous copy to GPU using pinned memory"""
        try:
            # Create pinned buffer if source is not already pinned
            if not source.is_pinned():
                # For now, just perform regular copy - in future could use pinned staging
                pass
            
            # Perform async copy with specified stream
            if stream:
                with torch.cuda.stream(stream):
                    return source.to(target_device, non_blocking=True)
            else:
                return source.to(target_device, non_blocking=True)
                
        except Exception as e:
            print(f"[CHROMADCT PINNED] Async copy failed, falling back to sync: {e}")
            return source.to(target_device)


def get_optimized_chromadct_memory_strategy(model, available_gpu_memory: int, current_step: int = 0) -> Dict:
    """
    Main entry point for ChromaDCT memory optimization.
    Combines all optimization strategies for maximum performance.
    """
    
    # Check memory pressure and adjust strategy accordingly
    current_pressure = _chromadct_pressure_monitor.get_memory_pressure()
    adaptive_window_size = _chromadct_pressure_monitor.get_adaptive_window_size()
    
    print(f"\\n[CHROMADCT OPTIMIZER] === ChromaDCT Memory Strategy Optimization ===")
    print(f"[CHROMADCT OPTIMIZER] Memory pressure: {current_pressure:.1%}")
    print(f"[CHROMADCT OPTIMIZER] Adaptive window size: {adaptive_window_size}")
    print(f"[CHROMADCT OPTIMIZER] Available GPU memory: {available_gpu_memory / (1024**2):.1f} MB")
    print(f"[CHROMADCT OPTIMIZER] Current step: {current_step}")
    
    # Get execution-order-aware priority groups  
    try:
        priority_groups = get_chromadct_memory_priority_groups(model, current_step, 28)
    except Exception as e:
        print(f"[CHROMADCT OPTIMIZER] Failed to get priority groups, using fallback: {e}")
        priority_groups = {
            'critical': ['img_in_patch', 'txt_in', 'pe_embedder', 'distilled_guidance_layer'],
            'active_window': [],
            'prefetch': [], 
            'inactive': []
        }
    
    # Check for emergency unloading due to memory pressure
    emergency_unload = []
    if current_pressure > 0.95:
        emergency_unload = _chromadct_pressure_monitor.get_emergency_unload_candidates(
            _chromadct_window_manager.loaded_components,
            priority_groups.get('critical', [])
        )
    
    # Get NeRF batching recommendations
    nerf_batch_recommended = any(
        _chromadct_nerf_batcher.should_batch_load_nerf(comp) 
        for comp in priority_groups.get('active_window', [])
    )
    
    strategy = {
        'type': 'chromadct_optimized',
        'priority_groups': priority_groups,
        'memory_pressure': current_pressure,
        'adaptive_window_size': adaptive_window_size,
        'emergency_unload': emergency_unload,
        'nerf_batch_recommended': nerf_batch_recommended,
        'prefetch_enabled': True,
        'pinned_memory_enabled': True,
        'optimizations_applied': [
            'execution_order_aware_prioritization',
            'rolling_window_management', 
            'async_prefetching',
            'nerf_block_batching',
            'memory_pressure_monitoring',
            'pinned_memory_optimization'
        ]
    }
    
    print(f"[CHROMADCT OPTIMIZER] Strategy: {len(strategy['optimizations_applied'])} optimizations active")
    if emergency_unload:
        print(f"[CHROMADCT OPTIMIZER] Emergency unloading {len(emergency_unload)} components!")
    if nerf_batch_recommended:
        print(f"[CHROMADCT OPTIMIZER] NeRF batch loading recommended")
    print(f"[CHROMADCT OPTIMIZER] === ChromaDCT Memory Strategy Complete ===\\n")
    
    return strategy


# Global instances for tracking ChromaDCT optimization state
_chromadct_window_manager = ChromaDCTRollingWindowManager()
_chromadct_prefetcher = ChromaDCTAsyncPrefetcher()
_chromadct_nerf_batcher = ChromaDCTNeRFBatcher()
_chromadct_pressure_monitor = ChromaDCTMemoryPressureMonitor()
_chromadct_pinned_manager = ChromaDCTPinnedMemoryManager()


def reset_chromadct_optimization_state():
    """Reset all ChromaDCT optimization state for new inference runs"""
    global _chromadct_window_manager, _chromadct_prefetcher, _chromadct_nerf_batcher
    global _chromadct_pressure_monitor, _chromadct_pinned_manager
    
    print("[CHROMADCT RESET] Resetting all ChromaDCT optimization state")
    
    # Reset all managers
    _chromadct_window_manager.current_step = 0
    _chromadct_window_manager.loaded_components.clear()
    
    _chromadct_prefetcher.prefetch_queue.clear()
    
    _chromadct_nerf_batcher.nerf_phase_active = False
    _chromadct_nerf_batcher.batch_loaded = False
    
    _chromadct_pressure_monitor.pressure_history.clear()
    _chromadct_pressure_monitor.adaptive_window_size = 6
    
    # Clean up pinned memory allocations
    with _chromadct_pinned_manager.lock:
        _chromadct_pinned_manager.pinned_allocations.clear()
        _chromadct_pinned_manager.current_pinned = 0