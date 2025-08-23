"""
Utility functions for Chroma Radiance models.
Handles latent space operations and radiance-specific functionality.
"""

import torch
import numpy as np
from PIL import Image
import modules.shared as shared


def image_to_radiance_latent(image, device=None):
    """
    Convert PIL image or torch tensor to radiance latent space.
    Radiance models work directly in RGB space with normalization to [-1, 1].
    """
    if device is None:
        device = shared.device
        
    if isinstance(image, Image.Image):
        # Convert PIL to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
    elif isinstance(image, torch.Tensor):
        image_tensor = image
        # Ensure proper format
        if image_tensor.max() > 1.1:  # Likely in [0, 255] range
            image_tensor = image_tensor / 255.0
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
    else:
        raise ValueError("Input must be PIL Image or torch Tensor")
    
    # Move to device and normalize to [-1, 1]
    image_tensor = image_tensor.to(device)
    latent = image_tensor * 2.0 - 1.0
    
    return latent


def radiance_latent_to_image(latent):
    """
    Convert radiance latent space to displayable images.
    """
    # Convert from [-1, 1] to [0, 1]
    image_tensor = (latent + 1.0) * 0.5
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    
    # Convert to PIL images
    images = []
    for i in range(image_tensor.shape[0]):
        img = image_tensor[i]
        # Convert from CHW to HWC
        img = img.permute(1, 2, 0).cpu().numpy()
        # Convert to 0-255 range
        img = (img * 255).astype(np.uint8)
        # Create PIL image
        pil_img = Image.fromarray(img)
        images.append(pil_img)
    
    return images


def validate_radiance_model():
    """
    Check if the currently loaded model is a radiance model.
    """
    if not hasattr(shared.sd_model, 'is_radiance_model'):
        return False, "Model does not support radiance functionality"
    
    if not shared.sd_model.is_radiance_model():
        return False, "Current model is not a radiance model"
    
    return True, "Radiance model ready"


def get_radiance_model_info():
    """
    Get information about the current radiance model.
    """
    if not validate_radiance_model()[0]:
        return None
        
    info = {
        "model_type": "ChromaRadiance",
        "has_vae": False,
        "operates_in_latent_space": True,
        "patch_size": getattr(shared.sd_model.forge_objects.unet.model, 'params', {}).get('patch_size', 16),
        "input_channels": 3,
        "output_channels": 3,
    }
    
    return info


def prepare_radiance_noise(batch_size, height, width, device=None, dtype=None):
    """
    Create properly formatted noise for radiance models.
    """
    if device is None:
        device = shared.device
    if dtype is None:
        dtype = shared.sd_model.dtype if hasattr(shared.sd_model, 'dtype') else torch.float32
    
    # Radiance models work with 3-channel RGB data
    shape = (batch_size, 3, height, width)
    noise = torch.randn(shape, device=device, dtype=dtype)
    
    return noise


def radiance_dimensions_valid(width, height, patch_size=16):
    """
    Check if image dimensions are valid for radiance models.
    Radiance models require dimensions to be multiples of patch size.
    """
    if width % patch_size != 0 or height % patch_size != 0:
        return False, f"Dimensions must be multiples of {patch_size}. Got {width}x{height}"
    
    if width < patch_size or height < patch_size:
        return False, f"Dimensions must be at least {patch_size}x{patch_size}"
    
    return True, "Dimensions valid"


def estimate_radiance_memory_usage(width, height, batch_size=1, steps=20):
    """
    Estimate GPU memory usage for radiance generation.
    """
    # Base model memory (approximate)
    model_memory = 12 * 1024**3  # ~12GB for base model
    
    # Latent memory (3 channels for radiance)
    latent_size = batch_size * 3 * height * width * 4  # 4 bytes per float32
    
    # Additional memory for gradients and temporary tensors during generation
    generation_overhead = latent_size * steps * 2  # Rough estimate
    
    total_memory = model_memory + latent_size + generation_overhead
    
    return {
        "model_memory_gb": model_memory / 1024**3,
        "latent_memory_mb": latent_size / 1024**2,
        "generation_overhead_mb": generation_overhead / 1024**2,
        "total_estimated_gb": total_memory / 1024**3
    }


def get_default_radiance_settings():
    """
    Get default settings optimized for radiance models.
    """
    return {
        "steps": 20,
        "cfg_scale": 7.0,
        "width": 1024,
        "height": 1024,
        "batch_size": 1,
        "batch_count": 1,
        "sampler": "Euler",
        "radiance_guidance": 0.0,
        "radiance_attn_padding": 1,
        "seed": -1,
    }


def radiance_post_process_image(image, enhance_contrast=True, adjust_gamma=None):
    """
    Post-process radiance-generated images.
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL
        if image.ndim == 4:
            image = image[0]  # Take first image from batch
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    if enhance_contrast:
        # Enhance contrast slightly
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
    
    if adjust_gamma is not None:
        # Apply gamma correction
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.power(image_array, 1.0 / adjust_gamma)
        image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array)
    
    return image