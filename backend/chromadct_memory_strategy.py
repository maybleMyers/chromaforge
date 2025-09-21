"""
ChromaDCT-Optimized Memory Management Strategy
Provides intelligent swapping that accounts for ChromaDCT's unique memory access patterns
"""

import torch
from typing import Dict, List, Optional
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


def get_chromadct_memory_priority_groups(model) -> Dict[str, List[str]]:
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
    
    for priority in ['critical', 'high_priority', 'medium_priority', 'low_priority']:
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
        'strategy': 'chromadct_priority_based'
    }


def apply_chromadct_swapping_strategy(model, available_gpu_memory: int):
    """
    Apply ChromaDCT-optimized swapping strategy to a model
    """
    
    if not is_chromadct_model(model):
        # Use default strategy for non-ChromaDCT models
        return None
    
    print(f"Applying ChromaDCT-optimized memory strategy with {available_gpu_memory / (1024**3):.1f}GB available")
    
    memory_profile = calculate_chromadct_memory_profile(model, available_gpu_memory)
    
    # Count total components in each location
    gpu_count = sum(len(components) for components in memory_profile['gpu_components'].values())
    cpu_count = sum(len(components) for components in memory_profile['cpu_components'].values())
    total_count = gpu_count + cpu_count
    
    print(f"ChromaDCT Memory Allocation:")
    print(f"  GPU: {gpu_count}/{total_count} components ({gpu_count/total_count*100:.1f}%)")
    print(f"  CPU: {cpu_count}/{total_count} components ({cpu_count/total_count*100:.1f}%)")
    
    for priority in ['critical', 'high_priority', 'medium_priority', 'low_priority']:
        gpu_in_priority = len(memory_profile['gpu_components'].get(priority, []))
        cpu_in_priority = len(memory_profile['cpu_components'].get(priority, []))
        total_in_priority = gpu_in_priority + cpu_in_priority
        
        if total_in_priority > 0:
            print(f"  {priority:15}: {gpu_in_priority:2d}/{total_in_priority:2d} on GPU")
    
    return memory_profile


def get_chromadct_inference_memory_multiplier() -> float:
    """
    Get memory multiplier for ChromaDCT inference requirements
    ChromaDCT needs less inference memory than VAE-based models
    """
    if is_chromadct_model(None):
        # ChromaDCT processes in pixel space (3 channels) vs latent space (16 channels)
        # and has more efficient NeRF processing
        return 1  # reset
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
    
    print(f"ChromaDCT Memory Optimization:")
    print(f"  Original inference memory: {inference_memory / (1024**2):.0f} MB")
    print(f"  Optimized inference memory: {adjusted_inference_memory / (1024**2):.0f} MB")  
    print(f"  Additional model memory: {(inference_memory - adjusted_inference_memory) / (1024**2):.0f} MB")
    print(f"  Total available for model: {adjusted_available_memory / (1024**2):.0f} MB")
    
    return {
        'adjusted_inference_memory': adjusted_inference_memory,
        'adjusted_available_memory': adjusted_available_memory,
        'memory_saved': inference_memory - adjusted_inference_memory
    }