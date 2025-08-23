"""
Radiance Model Detection and Utilities
Provides helpers for detecting and working with Chroma Radiance models
"""

import torch
import modules.shared as shared
import modules.sd_models as sd_models
from modules import errors


def is_radiance_model(model=None):
    """
    Check if the given model (or current loaded model) is a radiance model.
    
    Args:
        model: Model to check (defaults to currently loaded model)
        
    Returns:
        bool: True if model is a radiance model, False otherwise
    """
    if model is None:
        model = shared.sd_model
    
    if model is None:
        return False
    
    # Method 1: Check if model has radiance marker method
    if hasattr(model, 'is_radiance_model') and callable(model.is_radiance_model):
        try:
            return model.is_radiance_model()
        except:
            pass
    
    # Method 2: Check model structure for NeRF components
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
            
            # Check for NeRF-specific components in the transformer
            nerf_indicators = [
                'nerf_image_embedder',
                'nerf_blocks',
                'nerf_final_layer'
            ]
            
            for indicator in nerf_indicators:
                if hasattr(diffusion_model, indicator):
                    return True
                    
                # Check in nested modules
                for name, module in diffusion_model.named_modules():
                    if indicator in name:
                        return True
        
        return False
        
    except Exception as e:
        print(f"Warning: Could not check radiance model status: {e}")
        return False


def get_model_type_string(model=None):
    """
    Get a string describing the model type.
    
    Args:
        model: Model to check (defaults to currently loaded model)
        
    Returns:
        str: "Radiance" or "Standard"
    """
    return "Radiance" if is_radiance_model(model) else "Standard"


def auto_detect_and_set_model_type():
    """
    Auto-detect the current model type and update the chroma_model_type setting.
    Returns the detected type.
    """
    detected_type = get_model_type_string()
    
    # Update the setting if it exists
    try:
        if hasattr(shared.opts, 'chroma_model_type'):
            current_setting = shared.opts.chroma_model_type
            if current_setting != detected_type:
                shared.opts.chroma_model_type = detected_type
                print(f"Auto-detected model type: {detected_type}")
        return detected_type
    except Exception as e:
        print(f"Note: Could not auto-set model type: {e}")
        return detected_type


def get_radiance_model_info(model=None):
    """
    Get detailed information about a radiance model.
    
    Args:
        model: Model to analyze (defaults to currently loaded model)
        
    Returns:
        dict: Information about the radiance model
    """
    if model is None:
        model = shared.sd_model
    
    info = {
        'is_radiance': is_radiance_model(model),
        'model_type': get_model_type_string(model),
        'has_nerf_embedder': False,
        'has_nerf_blocks': False,
        'has_nerf_final_layer': False,
        'nerf_components': []
    }
    
    if info['is_radiance'] and model is not None:
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                diffusion_model = model.model.diffusion_model
                
                # Check for specific NeRF components
                if hasattr(diffusion_model, 'nerf_image_embedder'):
                    info['has_nerf_embedder'] = True
                    info['nerf_components'].append('nerf_image_embedder')
                
                if hasattr(diffusion_model, 'nerf_blocks'):
                    info['has_nerf_blocks'] = True
                    info['nerf_components'].append('nerf_blocks')
                
                if hasattr(diffusion_model, 'nerf_final_layer'):
                    info['has_nerf_final_layer'] = True
                    info['nerf_components'].append('nerf_final_layer')
                
                # Find all NeRF-related modules
                for name, module in diffusion_model.named_modules():
                    if 'nerf' in name.lower() and name not in info['nerf_components']:
                        info['nerf_components'].append(name)
        
        except Exception as e:
            info['error'] = str(e)
    
    return info


def list_radiance_models():
    """
    List all available radiance models from the checkpoints.
    
    Returns:
        list: List of checkpoint info objects for radiance models
    """
    radiance_models = []
    
    try:
        # Refresh model list
        sd_models.list_models()
        
        for checkpoint_info in sd_models.checkpoints_list.values():
            # Check filename for radiance indicators
            filename_lower = checkpoint_info.filename.lower()
            if 'radiance' in filename_lower:
                radiance_models.append(checkpoint_info)
                continue
                
            # TODO: Could add more sophisticated detection by loading model metadata
            # For now, rely on filename-based detection
    
    except Exception as e:
        print(f"Warning: Could not list radiance models: {e}")
    
    return radiance_models


def recommend_radiance_settings():
    """
    Get recommended settings for radiance model generation.
    
    Returns:
        dict: Recommended settings
    """
    return {
        'sampler_name': 'Euler',
        'scheduler': 'Simple', 
        'steps': 20,
        'cfg_scale': 7.0,
        'radiance_guidance': 0.0,
        'radiance_attn_padding': 1,
        'width': 1024,
        'height': 1024,
        'batch_size': 1  # Radiance models may be memory intensive
    }


def validate_radiance_settings(settings):
    """
    Validate settings for radiance model generation.
    
    Args:
        settings: Dictionary of generation settings
        
    Returns:
        tuple: (is_valid, warnings_list)
    """
    warnings = []
    
    # Check guidance value
    guidance = settings.get('radiance_guidance', 0.0)
    if guidance < 0.0 or guidance > 10.0:
        warnings.append(f"Radiance guidance {guidance} is outside recommended range 0.0-10.0")
    
    # Check attention padding
    padding = settings.get('radiance_attn_padding', 1)
    if padding < 1 or padding > 16:
        warnings.append(f"Radiance attention padding {padding} is outside range 1-16")
    
    # Check resolution
    width = settings.get('width', 1024)
    height = settings.get('height', 1024)
    if width * height > 2048 * 2048:
        warnings.append(f"High resolution {width}x{height} may cause memory issues with radiance models")
    
    # Check batch size
    batch_size = settings.get('batch_size', 1)
    if batch_size > 2:
        warnings.append(f"Batch size {batch_size} may cause memory issues with radiance models")
    
    return len(warnings) == 0, warnings


# Auto-detect on module import if a model is loaded
try:
    if shared.sd_model is not None:
        auto_detect_and_set_model_type()
except:
    pass  # Ignore errors during module import