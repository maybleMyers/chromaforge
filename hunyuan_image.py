"""
HunyuanImage-3.0 Backend for Image Generation
Supports text-to-image and image-to-image generation with the 80B MoE model.
Supports multi-GPU distribution, quantization, and CPU offloading.

Requirements:
- pip install transformers>=4.51.0 accelerate torch torchvision bitsandbytes
- For multi-GPU: pip install accelerate

Usage:
    python hunyuan_image.py --models-dir models/LLM --port 7864
"""

import os
import gc
import time
import json
import argparse
import tempfile
import threading
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from io import BytesIO
import base64

import torch
import gradio as gr
from gradio import themes
from PIL import Image
from dataclasses import dataclass
import copy

# Image size options for generation
# With dynamic resolution patch, multiple base sizes are supported (256, 512, 768, 1024, 1536, 2048)
# Each base_size generates a ResolutionGroup with different aspect ratios around that size.
IMAGE_SIZE_OPTIONS = [
    # Small (base_size=512)
    "512x512",
    "512x384",
    "384x512",
    "512x288",
    "288x512",
    # Medium (base_size=768)
    "768x768",
    "768x576",
    "576x768",
    "768x432",
    "432x768",
    # Standard (base_size=1024)
    "1024x1024",
    "1024x768",
    "768x1024",
    "1280x720",
    "720x1280",
    "1152x896",
    "896x1152",
    # Large (base_size=1536)
    "1536x1536",
    "1536x1152",
    "1152x1536",
    "1920x1080",
    "1080x1920",
    # Extra Large (base_size=2048)
    "2048x2048",
    "2048x1536",
    "1536x2048",
]

# Settings file path
SETTINGS_FILE = "hunyuan_image_settings.json"

# Global stop flag for generation
stop_generation: bool = False

# Modules to skip during quantization (keep in full precision)
# These are small but critical for image quality and numerical stability
HUNYUAN_SKIP_MODULES = [
    # VAE - pixel reconstruction quality (encoder + decoder)
    'vae',
    # Vision Encoder - semantic understanding (SigLIP2)
    'vision_model', 'vision_aligner',
    # Image Generation Head - latent output
    'final_layer', 'patch_embed',
    # Timestep/Guidance Embeddings - diffusion timing
    'time_embed', 'time_embed_2', 'timestep_emb', 'guidance_emb', 'timestep_r_emb',
    # MoE Router Gates - MUST stay FP32 for numerical stability
    'wg',
    # Token Embeddings
    'wte', 'lm_head',
    # All LayerNorms - must stay FP32 for stability
    'ln_f', 'layernorm', 'input_layernorm', 'post_attention_layernorm',
    'key_layernorm', 'query_layernorm',  # QK attention layernorms
]

# Bot task options for generation
BOT_TASK_OPTIONS = ["think_recaption", "recaption", "image", "auto"]

# System prompt options
SYSTEM_PROMPT_OPTIONS = ["en_unified", "en_recaption", "en_think_recaption", "en_vanilla", "dynamic", "custom", "None"]

# MoE implementation options
MOE_IMPL_OPTIONS = ["eager", "flashinfer"]

# Default settings values
DEFAULT_SETTINGS = {
    # Model Settings
    "model_name": None,  # Will use first available if None
    "dtype": "bfloat16",
    "num_gpus": 1,
    "max_memory_per_gpu": 0,  # 0 = auto
    "cpu_offload": False,
    "cpu_offload_ram": 0,  # 0 = auto
    "flash_attention": False,
    "moe_impl": "eager",
    "moe_drop_tokens": True,
    # Generation Settings
    "default_steps": 50,
    "default_guidance": 2.5,  # Model default is 2.5
    "default_flow_shift": 3.0,  # Model default
    "default_image_size": "1024x1024",
    "infer_align_image_size": True,
    "use_taylor_cache": False,
    # Prompt Settings
    "system_prompt_type": "en_unified",
    "custom_system_prompt": "",
    "bot_task": "think_recaption",
    "verbose": 2,
}


def save_settings(
    # Model Settings
    model_name: str,
    dtype: str,
    num_gpus: int,
    max_memory_per_gpu: int,
    cpu_offload: bool,
    cpu_offload_ram: int,
    flash_attention: bool,
    moe_impl: str,
    moe_drop_tokens: bool,
    # Generation Settings
    default_steps: int,
    default_guidance: float,
    default_flow_shift: float,
    default_image_size: str,
    infer_align_image_size: bool,
    use_taylor_cache: bool,
    # Prompt Settings
    system_prompt_type: str,
    custom_system_prompt: str,
    bot_task: str,
    verbose: int,
) -> str:
    """Save all settings to a JSON file."""
    settings = {
        # Model Settings
        "model_name": model_name,
        "dtype": dtype,
        "num_gpus": num_gpus,
        "max_memory_per_gpu": max_memory_per_gpu,
        "cpu_offload": cpu_offload,
        "cpu_offload_ram": cpu_offload_ram,
        "flash_attention": flash_attention,
        "moe_impl": moe_impl,
        "moe_drop_tokens": moe_drop_tokens,
        # Generation Settings
        "default_steps": default_steps,
        "default_guidance": default_guidance,
        "default_flow_shift": default_flow_shift,
        "default_image_size": default_image_size,
        "infer_align_image_size": infer_align_image_size,
        "use_taylor_cache": use_taylor_cache,
        # Prompt Settings
        "system_prompt_type": system_prompt_type,
        "custom_system_prompt": custom_system_prompt,
        "bot_task": bot_task,
        "verbose": verbose,
    }

    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"[hunyuan_image] Settings saved to {SETTINGS_FILE}")
        return f"Settings saved to {SETTINGS_FILE}"
    except Exception as e:
        print(f"[hunyuan_image] Error saving settings: {e}")
        return f"Error saving settings: {e}"


def load_settings() -> Dict[str, Any]:
    """Load settings from JSON file. Returns defaults if file doesn't exist."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            # Merge with defaults to handle missing keys
            merged = {**DEFAULT_SETTINGS, **settings}
            print(f"[hunyuan_image] Settings loaded from {SETTINGS_FILE}")
            return merged
        else:
            print(f"[hunyuan_image] No settings file found, using defaults")
            return DEFAULT_SETTINGS.copy()
    except Exception as e:
        print(f"[hunyuan_image] Error loading settings: {e}, using defaults")
        return DEFAULT_SETTINGS.copy()


# Try to import psutil for memory detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. CPU offload memory detection will use defaults.")


def convert_model_to_fp8_scaled(model, skip_patterns=None):
    """
    Convert model linear layers to FP8 with per-tensor scaling.
    Saves ~50% memory. Works on any GPU (older GPUs compute in fp16/fp32).

    Intelligently skips layers that would lose too much precision in FP8.

    Args:
        model: The model to convert
        skip_patterns: List of layer name patterns to skip (e.g., ['lm_head', 'embed'])

    Returns:
        model with FP8 weights where safe
    """
    import torch.nn as nn

    if skip_patterns is None:
        # Default: skip embedding, output head, norm, and vision encoder layers
        # Vision layers are sensitive to quantization (based on analysis)
        skip_patterns = ['embed', 'lm_head', 'wte', 'wpe', 'norm', 'visual']

    converted_count = 0
    skipped_count = 0
    total_params_before = 0
    total_params_after = 0

    for name, child in model.named_modules():
        if isinstance(child, nn.Linear) and not hasattr(child, 'fp8_converted'):
            weight = child.weight.data
            original_dtype = weight.dtype

            # Skip if already FP8
            if original_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                continue

            # Skip meta tensors (shouldn't happen after CPU load, but be safe)
            if weight.device.type == 'meta':
                skipped_count += 1
                continue

            # Skip patterns that shouldn't be converted
            if any(pattern in name.lower() for pattern in skip_patterns):
                skipped_count += 1
                continue

            # Check if layer can be safely converted to FP8
            # FP8 E4M3 has range ~[-448, 448] with limited precision
            weight_float = weight.float()
            abs_max = weight_float.abs().max()
            non_zero = weight_float[weight_float != 0]
            abs_min = non_zero.abs().min() if non_zero.numel() > 0 else torch.tensor(1.0)

            # Skip if dynamic range is too large (would lose small values)
            if abs_max > 0 and abs_min > 0:
                dynamic_range = abs_max / abs_min
                if dynamic_range > 1e6:  # Very large dynamic range
                    skipped_count += 1
                    continue

            # Calculate memory before
            param_bytes_before = weight.numel() * weight.element_size()
            total_params_before += param_bytes_before

            # Compute scale factor (FP8 E4M3 max value is ~448)
            if abs_max == 0:
                abs_max = torch.tensor(1.0, device=weight.device, dtype=torch.float32)
            scale = (abs_max / 448.0).float()

            # Convert to FP8
            fp8_weight = (weight.float() / scale).to(torch.float8_e4m3fn)

            # Store FP8 weight and scale
            child.weight = nn.Parameter(fp8_weight, requires_grad=False)
            child.register_buffer('scale_weight', scale.view(1))
            child.computation_dtype = original_dtype  # Use original dtype for computation

            # Calculate memory after
            total_params_after += fp8_weight.numel() * fp8_weight.element_size()
            total_params_after += 4  # scale is float32

            # Replace forward method
            original_forward = child.forward
            child.original_forward = original_forward
            child.forward = lambda x, m=child: _fp8_linear_forward(m, x)
            child.fp8_converted = True

            converted_count += 1

    if converted_count > 0 or skipped_count > 0:
        mem_before_gb = total_params_before / (1024**3)
        mem_after_gb = total_params_after / (1024**3)
        reduction = (1 - mem_after_gb / mem_before_gb) * 100 if mem_before_gb > 0 else 0
        print(f"[FP8] Converted {converted_count} linear layers to FP8 scaled, skipped {skipped_count}")
        print(f"[FP8] Linear layer memory: {mem_before_gb:.2f}GB -> {mem_after_gb:.2f}GB ({reduction:.1f}% reduction)")

    # Force garbage collection to free original weights
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def _fp8_linear_forward(layer, input):
    """FP8 linear forward with dequantization."""
    weight = layer.weight
    scale = layer.scale_weight
    computation_dtype = getattr(layer, 'computation_dtype', torch.float16)

    # Dequantize weight to computation dtype
    weight_dequant = weight.to(computation_dtype) * scale.to(computation_dtype)

    # Compute in original dtype
    input_cast = input.to(computation_dtype) if input.dtype != computation_dtype else input

    # Standard linear
    if layer.bias is not None:
        output = torch.nn.functional.linear(input_cast, weight_dequant, layer.bias.to(computation_dtype))
    else:
        output = torch.nn.functional.linear(input_cast, weight_dequant, None)

    return output

# Try to import video processing utilities
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Video support will be limited.")

# Check for transformers and accelerate
try:
    import transformers
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Error: transformers not installed or incompatible version: {e}")

# HunyuanImage-3.0 uses trust_remote_code to load model classes from the model directory
# No need to import from a local hunyuan3/ folder
from transformers import AutoModelForCausalLM
HUNYUAN_AVAILABLE = TRANSFORMERS_AVAILABLE  # Available if transformers is installed

try:
    import accelerate
    from accelerate import init_empty_weights
    ACCELERATE_AVAILABLE = True
    print(f"Accelerate version: {accelerate.__version__}")
except ImportError:
    ACCELERATE_AVAILABLE = False
    init_empty_weights = None
    print("Warning: accelerate not installed. Multi-GPU support will be limited.")

try:
    from optimum.quanto import requantize
    QUANTO_AVAILABLE = True
    print("Quanto available")
except ImportError:
    QUANTO_AVAILABLE = False
    print("Warning: optimum-quanto not installed. Quanto quantization unavailable.")

try:
    from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Quantized model saving unavailable.")


def apply_quantization_dtype_fix(model):
    """
    Monkey-patch the model to fix dtype mismatches when using quantization.

    When BitsAndBytes quantization is applied, the transformer backbone produces
    hidden_states in a different dtype than the skipped modules (patch_embed,
    timestep_emb, etc.). The scatter_ operations require matching dtypes.

    This patch wraps the problematic methods to cast src tensors before scatter.
    """
    import types

    # Store original methods
    original_instantiate_vae_image_tokens = model.instantiate_vae_image_tokens
    original_instantiate_vit_image_tokens = model.instantiate_vit_image_tokens
    original_instantiate_continuous_tokens = model.instantiate_continuous_tokens
    original_instantiate_guidance_tokens = model.instantiate_guidance_tokens
    original_instantiate_timestep_r_tokens = model.instantiate_timestep_r_tokens

    def patched_instantiate_vae_image_tokens(self, hidden_states, timesteps, images, image_mask, guidance=None, timesteps_r=None):
        """Patched version with dtype casting for quantization compatibility."""
        # Handle the hidden_states is None case - no scatter ops, just call original
        if hidden_states is None:
            return original_instantiate_vae_image_tokens(hidden_states, timesteps, images, image_mask, guidance, timesteps_r)

        if images is None:
            return hidden_states

        bsz, seqlen, n_embd = hidden_states.shape
        target_dtype = hidden_states.dtype

        if isinstance(images, torch.Tensor):
            assert images.ndim == 4, f"images should be a 4-D tensor, got {images.ndim}-D tensor"
            assert isinstance(timesteps, torch.Tensor), f"timesteps should be 1-D tensor, got {type(timesteps)}"

            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
            t_emb = self.time_embed(timesteps)
            image_seq, token_h, token_w = self.patch_embed(images, t_emb)
            image_scatter_index = index.masked_select(image_mask.bool()).reshape(bsz, -1)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq.to(target_dtype),  # Cast for quantization compatibility
            )
        else:  # list
            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
            for i, (image_i, t_i) in enumerate(zip(images, timesteps)):
                t_i_emb = self.time_embed(t_i)

                if isinstance(image_i, torch.Tensor):
                    image_i_seq, _, _ = self.patch_embed(image_i, t_i_emb)
                elif isinstance(image_i, list):
                    image_i_seq_list = []
                    for j in range(len(image_i)):
                        image_ij = image_i[j].unsqueeze(0)
                        assert image_ij.ndim == 4, f"image_ij should have size of (1, C, H, W), got {list(image_ij.size())}"
                        image_i_seq_j = self.patch_embed(image_ij, t_i_emb[j:j + 1])[0]
                        image_i_seq_list.append(image_i_seq_j)
                    image_i_seq = torch.cat(image_i_seq_list, dim=1)
                else:
                    raise TypeError(f"image_i should be a torch.Tensor or a list, got {type(image_i)}")

                image_i_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                hidden_states[i:i + 1].scatter_(
                    dim=1,
                    index=image_i_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_i_seq.reshape(1, -1, n_embd).to(target_dtype),  # Cast for quantization compatibility
                )

        return hidden_states

    def patched_instantiate_vit_image_tokens(self, hidden_states, images, image_masks, **image_kwargs):
        """Patched version with dtype casting for quantization compatibility."""
        if images is None:
            return hidden_states

        bsz, seqlen, n_embd = hidden_states.shape
        target_dtype = hidden_states.dtype
        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)

        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                n = 1
            else:
                n = 1
            image_embeds = self._forward_vision_encoder(images, **image_kwargs)
            image_seqlen = image_embeds.size(1)

            image_scatter_index = index.masked_select(image_masks.bool()).reshape(bsz, -1)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_embeds.reshape(bsz, n * image_seqlen, n_embd).to(target_dtype),  # Cast
            )
        elif isinstance(images, list):
            for i, (image, image_mask) in enumerate(zip(images, image_masks)):
                cur_kwargs = {k: v[i] for k, v in image_kwargs.items()} if image_kwargs is not None else {}
                image_embed = self._forward_vision_encoder(image, **cur_kwargs)
                n, image_seqlen, n_embd_local = image_embed.shape
                image_embed = image_embed.reshape(n * image_seqlen, n_embd_local)

                image_scatter_index = index[i:i+1].masked_select(image_mask.bool()).reshape(1, -1)
                hidden_states[i:i+1].scatter_(
                    dim=1,
                    index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_embed.reshape(1, -1, n_embd).to(target_dtype),  # Cast
                )
        else:
            raise ValueError(f"und_images should be Tensor or List, but got {type(images)}")

        return hidden_states

    def patched_instantiate_continuous_tokens(self, hidden_states, timesteps=None, timesteps_index=None):
        """Patched version with dtype casting for quantization compatibility."""
        bsz, seqlen, n_embd = hidden_states.shape
        target_dtype = hidden_states.dtype

        if isinstance(timesteps, list):
            for i, timestep in enumerate(timesteps):
                timestep_src = self.timestep_emb(timestep)
                hidden_states[i:i+1].scatter_(
                    dim=1,
                    index=timesteps_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                    src=timestep_src.reshape(1, -1, n_embd).to(target_dtype),  # Cast
                )
        else:
            timesteps_src = self.timestep_emb(timesteps.reshape(-1))
            hidden_states.scatter_(
                dim=1,
                index=timesteps_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=timesteps_src.reshape(bsz, -1, n_embd).to(target_dtype),  # Cast
            )

        return hidden_states

    def patched_instantiate_guidance_tokens(self, hidden_states, guidance=None, guidance_index=None):
        """Patched version with dtype casting for quantization compatibility."""
        bsz, seqlen, n_embd = hidden_states.shape
        target_dtype = hidden_states.dtype

        guidance_src = self.guidance_emb(guidance.reshape(-1))
        hidden_states.scatter_(
            dim=1,
            index=guidance_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=guidance_src.reshape(bsz, -1, n_embd).to(target_dtype),  # Cast
        )

        return hidden_states

    def patched_instantiate_timestep_r_tokens(self, hidden_states, timesteps_r=None, timesteps_r_index=None):
        """Patched version with dtype casting for quantization compatibility."""
        bsz, seqlen, n_embd = hidden_states.shape
        target_dtype = hidden_states.dtype

        if isinstance(timesteps_r, list):
            for i, timestep_r in enumerate(timesteps_r):
                timestep_r_src = self.timestep_r_emb(timestep_r)
                hidden_states[i:i+1].scatter_(
                    dim=1,
                    index=timesteps_r_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                    src=timestep_r_src.reshape(1, -1, n_embd).to(target_dtype),  # Cast
                )
        else:
            timesteps_r_src = self.timestep_r_emb(timesteps_r.reshape(-1))
            hidden_states.scatter_(
                dim=1,
                index=timesteps_r_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=timesteps_r_src.reshape(bsz, -1, n_embd).to(target_dtype),  # Cast
            )

        return hidden_states

    # Apply patches
    model.instantiate_vae_image_tokens = types.MethodType(patched_instantiate_vae_image_tokens, model)
    model.instantiate_vit_image_tokens = types.MethodType(patched_instantiate_vit_image_tokens, model)
    model.instantiate_continuous_tokens = types.MethodType(patched_instantiate_continuous_tokens, model)
    model.instantiate_guidance_tokens = types.MethodType(patched_instantiate_guidance_tokens, model)
    model.instantiate_timestep_r_tokens = types.MethodType(patched_instantiate_timestep_r_tokens, model)

    # === Patch HunyuanStaticCache.update for index_copy_ dtype compatibility ===
    # The KV cache buffers may be in a different dtype than key/value states with CPU offload
    import importlib
    model_module = type(model).__module__
    hunyuan_module = importlib.import_module(model_module)

    if hasattr(hunyuan_module, 'HunyuanStaticCache'):
        HunyuanStaticCache = hunyuan_module.HunyuanStaticCache

        def patched_cache_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            """Patched update with dtype casting for quantization compatibility."""
            cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None

            # Lazy initialization if needed
            if self.layers[layer_idx].keys is None:
                self.layers[layer_idx].lazy_initialization(key_states)

            k_out = self.layers[layer_idx].keys
            v_out = self.layers[layer_idx].values

            # Get target dtype from the cache buffers and cast if needed
            target_dtype = k_out.dtype
            if key_states.dtype != target_dtype:
                key_states = key_states.to(target_dtype)
            if value_states.dtype != target_dtype:
                value_states = value_states.to(target_dtype)

            if cache_position is None:
                k_out.copy_(key_states)
                v_out.copy_(value_states)
            else:
                if cache_position.dim() == 1:
                    k_out.index_copy_(2, cache_position, key_states)
                    v_out.index_copy_(2, cache_position, value_states)

                    if self.dynamic:
                        end = cache_position[-1].item() + 1
                        k_out = k_out[:, :, :end]
                        v_out = v_out[:, :, :end]
                else:
                    assert cache_position.dim() == 2
                    batch_size, idx_size = cache_position.shape
                    for i in range(batch_size):
                        unbatched_dim = 1
                        k_out[i].index_copy_(unbatched_dim, cache_position[i], key_states[i])
                        v_out[i].index_copy_(unbatched_dim, cache_position[i], value_states[i])

                    if self.dynamic:
                        assert len(cache_position) == 1
                        end = cache_position[0, -1].item() + 1
                        k_out = k_out[:, :, :end]
                        v_out = v_out[:, :, :end]

            return k_out, v_out

        # Replace the class method (affects all instances)
        HunyuanStaticCache.update = patched_cache_update
        print("[hunyuan_image] Applied KV cache dtype compatibility patch")

    print("[hunyuan_image] Applied quantization dtype compatibility patches")
    return model


def apply_global_scatter_dtype_fix():
    """
    Patch torch.Tensor.scatter_ to automatically cast src tensor to destination dtype.
    This fixes dtype mismatches when using BitsAndBytes quantization with image generation.
    The existing apply_quantization_dtype_fix only patches 5 methods, but there are 20+
    scatter operations in the image generation pipeline that also need dtype casting.
    """
    if hasattr(torch.Tensor, '_original_scatter_'):
        return  # Already patched

    original_scatter = torch.Tensor.scatter_

    def patched_scatter_(self, dim, index, src, **kwargs):
        if isinstance(src, torch.Tensor) and src.dtype != self.dtype:
            src = src.to(self.dtype)
        return original_scatter(self, dim, index, src, **kwargs)

    torch.Tensor._original_scatter_ = original_scatter
    torch.Tensor.scatter_ = patched_scatter_
    print("[hunyuan_image] Applied global scatter_ dtype fix")


# Supported base sizes from the tokenizer vocabulary
SUPPORTED_BASE_SIZES = [256, 512, 768, 1024, 1536, 2048]


def get_best_base_size(width: int, height: int) -> int:
    """
    Determine the best base_size for a given resolution.
    The base_size should be close to max(width, height) but from supported values.
    """
    max_dim = max(width, height)

    # Find the closest supported base_size
    best_base = SUPPORTED_BASE_SIZES[0]
    min_diff = abs(max_dim - best_base)

    for base in SUPPORTED_BASE_SIZES:
        diff = abs(max_dim - base)
        if diff < min_diff:
            min_diff = diff
            best_base = base

    return best_base


def apply_dynamic_resolution_patch(model):
    """
    Patch the model's image_processor to support dynamic base_size selection.
    This allows generating images at different resolutions (512x512, 768x768, etc.)
    instead of always snapping to the default 1024-based resolutions.
    """
    import numpy as np
    import importlib

    # Get the tokenization module to access Resolution and ResolutionGroup classes
    model_module = type(model).__module__
    hunyuan_module = importlib.import_module(model_module)
    tokenization_module_name = model_module.rsplit('.', 1)[0] + '.tokenization_hunyuan_image_3'

    try:
        tokenization_module = importlib.import_module(tokenization_module_name)
        Resolution = tokenization_module.Resolution
        ResolutionGroup = tokenization_module.ResolutionGroup
        ImageInfo = tokenization_module.ImageInfo
    except (ImportError, AttributeError) as e:
        print(f"[hunyuan_image] Warning: Could not import tokenization module: {e}")
        return model

    # Cache for ResolutionGroups by base_size
    resolution_group_cache = {}

    def get_resolution_group(base_size: int) -> ResolutionGroup:
        """Get or create a ResolutionGroup for the given base_size."""
        if base_size not in resolution_group_cache:
            extra_resolutions = []
            # Add common aspect ratio resolutions for this base_size
            if base_size >= 720:
                # 16:9 and 9:16 for video-like aspect ratios
                w16_9 = int(base_size * 16 / 9 / 16) * 16  # Align to 16
                h16_9 = int(base_size * 9 / 16 / 16) * 16
                extra_resolutions.extend([
                    Resolution(base_size, int(base_size * 3 / 4 / 16) * 16),  # 4:3
                    Resolution(int(base_size * 3 / 4 / 16) * 16, base_size),  # 3:4
                ])
            resolution_group_cache[base_size] = ResolutionGroup(
                base_size=base_size,
                align=16,
                extra_resolutions=extra_resolutions if extra_resolutions else None
            )
        return resolution_group_cache[base_size]

    # Store reference to original build_gen_image_info
    original_build_gen_image_info = model.image_processor.build_gen_image_info

    def patched_build_gen_image_info(image_size, add_guidance_token=False, add_timestep_r_token=False):
        """
        Patched version that selects appropriate base_size based on requested resolution.
        """
        # Parse image size to get dimensions
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                # Use original for ratio-based sizes
                return original_build_gen_image_info(image_size, add_guidance_token, add_timestep_r_token)
            elif 'x' in image_size:
                h, w = [int(s) for s in image_size.split('x')]
            elif ':' in image_size:
                parts = [int(s) for s in image_size.split(':')]
                w, h = parts[0], parts[1]
            else:
                return original_build_gen_image_info(image_size, add_guidance_token, add_timestep_r_token)
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            h, w = image_size[0], image_size[1]
        else:
            return original_build_gen_image_info(image_size, add_guidance_token, add_timestep_r_token)

        # Determine the best base_size for this resolution
        best_base = get_best_base_size(w, h)

        # Get or create the appropriate ResolutionGroup
        reso_group = get_resolution_group(best_base)

        # Get target size and ratio index from this resolution group
        target_w, target_h = reso_group.get_target_size(w, h)
        base_size, ratio_idx = reso_group.get_base_size_and_ratio_index(w, h)

        # Calculate token dimensions
        vae_h_factor = model.image_processor.vae_info.h_factor
        vae_w_factor = model.image_processor.vae_info.w_factor
        token_height = target_h // vae_h_factor
        token_width = target_w // vae_w_factor

        # Create ImageInfo with the correct base_size
        image_info = ImageInfo(
            image_type="gen_image",
            image_width=target_w,
            image_height=target_h,
            token_width=token_width,
            token_height=token_height,
            base_size=base_size,
            ratio_index=ratio_idx,
            add_guidance_token=add_guidance_token,
            add_timestep_r_token=add_timestep_r_token,
        )

        return image_info

    # Apply the patch
    model.image_processor.build_gen_image_info = patched_build_gen_image_info
    model.image_processor._resolution_group_cache = resolution_group_cache
    model.image_processor._get_resolution_group = get_resolution_group

    print(f"[hunyuan_image] Applied dynamic resolution patch (supported base sizes: {SUPPORTED_BASE_SIZES})")
    return model


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get information about available GPUs."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": total_mem / (1024**3),
                "free_memory_gb": free_mem / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    return gpus


def print_gpu_status():
    """Print current GPU memory status."""
    gpus = get_gpu_info()
    if gpus:
        print("\n" + "=" * 60)
        print("GPU Status:")
        for gpu in gpus:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {gpu['free_memory_gb']:.2f} GB free / {gpu['total_memory_gb']:.2f} GB total")
        print("=" * 60 + "\n")
    else:
        print("No CUDA GPUs available.")


def log_gpu_memory(label: str = "", device: int = 0):
    """Log current GPU memory usage for debugging memory spikes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        used_gb = total_gb - free_gb
        print(f"[GPU Memory] {label}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, used={used_gb:.2f}GB/{total_gb:.2f}GB")
        return {"allocated": allocated, "reserved": reserved, "used": used_gb, "total": total_gb}
    return None


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 data URL."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{b64_data}"


def extract_video_frames(
    video_path: str,
    max_frames: int = 8,
    target_size: Optional[Tuple[int, int]] = None
) -> List[Image.Image]:
    """Extract frames from a video file for VLM processing."""
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv-python is required for video processing.")

    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return frames

        # Calculate frame indices to extract (evenly spaced)
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i * (total_frames - 1) / (max_frames - 1)) for i in range(max_frames)]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                if target_size:
                    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
                frames.append(pil_image)
    finally:
        cap.release()

    return frames


def find_hunyuan_models(models_dir: str) -> List[Dict[str, str]]:
    """
    Find HunyuanImage-3.0 models in a directory.
    Looks for directories containing config.json with model_type="hunyuan_image_3_moe".
    """
    models = []
    models_path = Path(models_dir)

    if not models_path.exists():
        return models

    # Look for directories with config.json (HuggingFace model format)
    for config_file in models_path.rglob("config.json"):
        model_dir = config_file.parent

        # Skip if this is a subdirectory of another model
        if any(p.name in ["tokenizer", "processor", "preprocessor"] for p in model_dir.parents):
            continue

        # Try to read config to detect model type
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                model_type_str = config.get("model_type", "").lower()

                # Detect HunyuanImage-3.0 models
                if "hunyuan_image_3" in model_type_str or "hunyuan-image" in model_type_str:
                    models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "type": "hunyuan-image",
                    })
        except Exception:
            pass

    return sorted(models, key=lambda x: x["name"])


class HunyuanImage3Backend:
    """
    Backend for HunyuanImage-3.0 image generation model.
    Supports text-to-image and image-to-image generation.
    Supports multi-GPU inference with automatic device mapping.
    """

    def __init__(self, models_dir: str = "models/LLM"):
        self.models_dir = models_dir
        self.model: Optional[HunyuanImage3ForCausalMM] = None
        self.current_model_path: Optional[str] = None
        self.device_map = None
        self.use_flash_attention = False

        # Check GPU availability
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[hunyuan_image] Detected {self.num_gpus} GPU(s)")
        print_gpu_status()

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available HunyuanImage-3.0 models."""
        return find_hunyuan_models(self.models_dir)

    def get_model_names(self) -> List[str]:
        """Get list of model names for dropdown."""
        models = self.get_available_models()
        if not models:
            return ["No models found"]
        return [m["name"] for m in models]

    def _create_device_map(self, num_gpus: int = 2) -> Union[str, Dict[str, int]]:
        """
        Create a device map for multi-GPU inference.
        Uses 'auto' for automatic distribution when accelerate is available.
        """
        if not ACCELERATE_AVAILABLE:
            print("Warning: accelerate not installed, using single GPU")
            return {"": 0}

        if num_gpus <= 1 or self.num_gpus <= 1:
            return "auto"

        # Use auto device map - accelerate will handle distribution
        return "auto"

    def load_model(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        num_gpus: int = 1,
        max_memory_per_gpu: Optional[int] = None,
        cpu_offload: bool = False,
        cpu_offload_ram: Optional[int] = None,
        flash_attention: bool = False,
        moe_impl: str = "eager",
        moe_drop_tokens: bool = True,
        progress=gr.Progress(),
    ) -> str:
        """
        Load a HunyuanImage-3.0 model with multi-GPU support.

        Args:
            model_name: Name of the model to load
            dtype: Data type (bfloat16, float16, q4_nf4, q8_fp16)
            num_gpus: Number of GPUs to use
            max_memory_per_gpu: Max memory per GPU in GB (None = auto)
            cpu_offload: Enable CPU offloading for large models
            cpu_offload_ram: Max CPU RAM to use for offloading in GB
            flash_attention: Use Flash Attention 2 for faster inference
            progress: Gradio progress callback

        Returns:
            Status message
        """
        self.use_flash_attention = flash_attention

        if not TRANSFORMERS_AVAILABLE:
            return "Error: transformers not installed"

        if not HUNYUAN_AVAILABLE:
            return "Error: transformers not available for HunyuanImage-3.0"

        # Find the model
        models = self.get_available_models()
        model_info = next((m for m in models if m["name"] == model_name), None)

        if model_info is None:
            return f"Error: Model '{model_name}' not found"

        model_path = model_info["path"]

        # Unload current model if any
        if self.model is not None:
            self.unload_model()

        try:
            progress(0.1, desc="Configuring device map...")

            # Determine torch dtype and quantization mode
            use_q8 = dtype in ["q8_partial", "q8_fp16"]
            use_q4_nf4 = dtype == "q4_nf4"
            use_fp8 = dtype == "fp8"
            use_quanto = dtype == "quanto_int4"

            # Check Quanto availability early
            if use_quanto and not QUANTO_AVAILABLE:
                return "Error: optimum-quanto not installed. Run: pip install optimum-quanto"

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "fp8": torch.bfloat16,  # Load in bf16 first, then convert to FP8
                "q8_partial": torch.bfloat16,
                "q8_fp16": torch.float16,
                "q4_nf4": torch.bfloat16,
                "quanto_int4": torch.bfloat16,  # Load in bf16 first, then apply quanto
            }
            torch_dtype = dtype_map.get(dtype, torch.bfloat16)

            # Create device map for multi-GPU or CPU offloading
            actual_gpus = min(num_gpus, self.num_gpus) if self.num_gpus > 0 else 0
            max_memory = None

            use_auto_device_map = (actual_gpus > 1) or cpu_offload or (max_memory_per_gpu is not None and max_memory_per_gpu > 0)

            if use_auto_device_map and self.num_gpus > 0:
                device_map = "auto"
                max_memory = {}

                if max_memory_per_gpu is not None and max_memory_per_gpu > 0:
                    for i in range(actual_gpus):
                        max_memory[i] = f"{max_memory_per_gpu}GiB"
                elif cpu_offload:
                    for i in range(actual_gpus):
                        if torch.cuda.is_available():
                            free_mem, total_mem = torch.cuda.mem_get_info(i)
                            gpu_mem_gb = int((free_mem / (1024**3)) * 0.9)
                            max_memory[i] = f"{gpu_mem_gb}GiB"
                            print(f"[hunyuan_image] GPU {i}: auto-detected {gpu_mem_gb}GB available")

                if cpu_offload and cpu_offload_ram and cpu_offload_ram > 0:
                    max_memory["cpu"] = f"{cpu_offload_ram}GiB"
                    print(f"[hunyuan_image] CPU offloading: up to {cpu_offload_ram}GB RAM")
                elif max_memory:
                    max_memory["cpu"] = "32GiB"

                if not max_memory:
                    max_memory = None

                print(f"[hunyuan_image] Using device_map='auto' for {actual_gpus} GPU(s)")
            else:
                # Use cuda:0 for single GPU without offloading ("all" is not a valid device_map value)
                device_map = "cuda:0" if self.num_gpus > 0 else "cpu"
                print(f"[hunyuan_image] Using device_map='{device_map}'")

            progress(0.2, desc="Loading tokenizer...")

            # Load tokenizer
            tokenizer_path = model_path
            print(f"[hunyuan_image] Loading tokenizer from {tokenizer_path}")

            progress(0.3, desc="Configuring quantization...")

            # Configure quantization
            quantization_config = None
            model_kwargs = {
                "trust_remote_code": True,
            }
            # Only use low_cpu_mem_usage with device_map="auto" (can interfere with "all")
            if device_map == "auto":
                model_kwargs["low_cpu_mem_usage"] = True

            if use_q4_nf4:
                print("[hunyuan_image] Using 4-bit NF4 quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=HUNYUAN_SKIP_MODULES,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload,  # Enable CPU offload support
                )
                model_kwargs["quantization_config"] = quantization_config
            elif use_q8:
                print("[hunyuan_image] Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=HUNYUAN_SKIP_MODULES,
                    llm_int8_threshold=6.0,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload,  # Enable CPU offload support
                )
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch_dtype

            if device_map is not None:
                model_kwargs["device_map"] = device_map
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory

            # Flash attention
            if flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("[hunyuan_image] Using Flash Attention 2")
            else:
                model_kwargs["attn_implementation"] = "sdpa"

            # MoE implementation
            model_kwargs["moe_impl"] = moe_impl
            model_kwargs["moe_drop_tokens"] = moe_drop_tokens
            print(f"[hunyuan_image] MoE implementation: {moe_impl}, drop_tokens: {moe_drop_tokens}")

            progress(0.4, desc="Loading HunyuanImage-3.0 model...")
            print(f"[hunyuan_image] Loading model from {model_path}")
            log_gpu_memory("Before model load")

            # Handle Quanto loading specially - load structure only, then apply quantized weights
            if use_quanto:
                # Check for quantized weights (either in model_path directly or in quanto_int4 subfolder)
                quanto_weights_path = os.path.join(model_path, "model_quanto_int4.safetensors")
                quanto_map_path = os.path.join(model_path, "quantization_map.json")

                # Also check subfolder for backwards compatibility
                if not os.path.exists(quanto_weights_path):
                    quanto_dir = os.path.join(model_path, "quanto_int4")
                    quanto_weights_path = os.path.join(quanto_dir, "model_quanto_int4.safetensors")
                    quanto_map_path = os.path.join(quanto_dir, "quantization_map.json")

                if not os.path.exists(quanto_weights_path) or not os.path.exists(quanto_map_path):
                    return (f"Error: No pre-quantized weights found.\n"
                            f"Expected: {quanto_weights_path}\n"
                            f"Run: python quantize_hunyuan.py --model-path <original_model_path>")

                print(f"[hunyuan_image] Loading pre-quantized model (skipping bf16 weights)...")
                progress(0.3, desc="Creating model structure...")

                # Load config and create model structure WITHOUT weights
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

                # Set model config options
                config.attn_implementation = model_kwargs.get("attn_implementation", "sdpa")
                config.moe_impl = moe_impl
                config.moe_drop_tokens = moe_drop_tokens

                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(
                        config,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                    )

                progress(0.5, desc="Loading quantized weights...")
                log_gpu_memory("After model structure, before quanto weights")

                # Load quantization map
                with open(quanto_map_path, 'r') as f:
                    qmap = json.load(f)

                # Load quantized state dict
                print(f"[hunyuan_image] Loading quantized weights from {quanto_weights_path}...")
                state_dict = safetensors_load(quanto_weights_path, device="cpu")

                progress(0.7, desc="Applying quantized weights...")

                # Apply quantized weights - this materializes the model with quanto tensors
                requantize(self.model, state_dict, qmap, device="cuda")

                log_gpu_memory("After Quanto loading")
                print(f"[hunyuan_image] Loaded pre-quantized model successfully")

            else:
                # Normal loading for non-quanto models
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs,
                )

            progress(0.8, desc="Loading tokenizer into model...")
            log_gpu_memory("After model load, before tokenizer")

            # Load tokenizer into model
            self.model.load_tokenizer(tokenizer_path)
            log_gpu_memory("After tokenizer load")

            # Apply dynamic resolution patch to support multiple base sizes (512, 768, 1024, etc.)
            self.model = apply_dynamic_resolution_patch(self.model)

            # Apply FP8 conversion if requested (only for non-quanto)
            if use_fp8 and not use_quanto:
                log_gpu_memory("Before FP8 conversion")
                print("[hunyuan_image] Converting model to FP8 scaled...")
                self.model = convert_model_to_fp8_scaled(self.model, skip_patterns=HUNYUAN_SKIP_MODULES)
                log_gpu_memory("After FP8 conversion")

            # Apply quantization compatibility patches if using quantization
            if use_q4_nf4 or use_q8 or use_quanto:
                self.model = apply_quantization_dtype_fix(self.model)
                apply_global_scatter_dtype_fix()

            self.current_model_path = model_path
            self.device_map = device_map

            progress(0.9, desc="Finalizing...")

            # Set eval mode
            self.model.eval()

            print_gpu_status()

            dtype_str = dtype
            if use_fp8:
                dtype_str = "fp8"
            elif use_q4_nf4:
                dtype_str = "q4_nf4"
            elif use_q8:
                dtype_str = "q8"

            progress(1.0, desc="Done!")
            return f"Loaded: {model_name} ({dtype_str}) on {actual_gpus} GPU(s)"

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.model = None
            return f"Error loading model: {str(e)}"

    def unload_model(self) -> str:
        """Unload the current model and free GPU memory."""
        if self.model is None:
            return "No model loaded"

        model_name = Path(self.current_model_path).stem if self.current_model_path else "model"

        try:
            # Delete model
            if self.model is not None:
                del self.model

            self.model = None
            self.current_model_path = None
            self.device_map = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

            print_gpu_status()
            return f"Unloaded: {model_name}"

        except Exception as e:
            return f"Error unloading: {str(e)}"

    def get_status(self) -> str:
        """Get current status."""
        if self.model is None:
            return "No model loaded"

        model_name = Path(self.current_model_path).stem if self.current_model_path else "Unknown"

        # Get device info
        if self.model is not None and hasattr(self.model, "hf_device_map"):
            unique_devices = set(self.model.hf_device_map.values())
            device_str = f"devices: {unique_devices}"
        else:
            device_str = "single device"

        return f"Loaded: {model_name} ({device_str})"


    def generate_image(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        seed: int = -1,
        image_size: str = "1024x1024",
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        flow_shift: float = 3.0,
        system_prompt_type: Optional[str] = "en_unified",
        custom_system_prompt: Optional[str] = None,
        bot_task: str = "think_recaption",
        verbose: int = 2,
        infer_align_image_size: bool = True,
        use_taylor_cache: bool = False,
        progress=gr.Progress(),
    ) -> Tuple[Optional[Image.Image], Dict[str, Any], str]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the image to generate
            reference_image: Optional reference image for image-to-image generation
            seed: Random seed (-1 for random)
            image_size: Output image size (e.g., "1024x1024")
            num_inference_steps: Number of diffusion steps
            guidance_scale: Diffusion guidance scale (diff_guidance_scale)
            flow_shift: Flow rectified flow shift parameter
            system_prompt_type: System prompt type (en_unified, en_recaption, etc.)
            custom_system_prompt: Custom system prompt (when system_prompt_type is "custom")
            bot_task: Bot task (think_recaption, recaption, image, auto)
            verbose: Verbosity level (0-2)
            infer_align_image_size: Align output size to input image size (for I2I)
            use_taylor_cache: Use Taylor Cache for faster inference
            progress: Gradio progress callback

        Returns:
            Tuple of (generated_image, stats_dict, cot_reasoning_text)
        """
        if self.model is None:
            return None, {"error": "No model loaded"}

        try:
            start_time = time.perf_counter()

            # Parse image size
            if "x" in image_size:
                width, height = map(int, image_size.split("x"))
            else:
                width = height = int(image_size)

            # Handle seed
            actual_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**32)
            print(f"[hunyuan_image] Generating with seed={actual_seed}, size={width}x{height}, steps={num_inference_steps}")

            progress(0.1, desc="Preparing generation...")

            # Build generation kwargs
            gen_kwargs = {
                "prompt": prompt,
                "image": reference_image,
                "image_size": (height, width),  # HunyuanImage uses (H, W)
                "diff_infer_steps": num_inference_steps,
                "diff_guidance_scale": guidance_scale,
                "flow_shift": flow_shift,
                "seed": actual_seed,
                "verbose": verbose,
                "use_taylor_cache": use_taylor_cache,
            }

            # Also set on generation_config directly (some model versions ignore kwargs)
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.diff_infer_steps = num_inference_steps
                self.model.generation_config.diff_guidance_scale = guidance_scale
                self.model.generation_config.flow_shift = flow_shift
                print(f"[hunyuan_image] Set generation_config: steps={num_inference_steps}, guidance={guidance_scale}, flow_shift={flow_shift}")

            # Add Taylor cache parameters when enabled
            if use_taylor_cache:
                gen_kwargs["taylor_cache_interval"] = 3  # Full computation every 3 steps
                gen_kwargs["taylor_cache_order"] = 2     # 2nd order Taylor approximation
                gen_kwargs["taylor_cache_enable_first_enhance"] = True
                gen_kwargs["taylor_cache_first_enhance_steps"] = 3
                gen_kwargs["taylor_cache_enable_tailing_enhance"] = True
                gen_kwargs["taylor_cache_tailing_enhance_steps"] = 1
                gen_kwargs["taylor_cache_low_freqs_order"] = 0
                gen_kwargs["taylor_cache_high_freqs_order"] = 2

            # Add infer_align_image_size for I2I mode
            if reference_image is not None:
                gen_kwargs["infer_align_image_size"] = infer_align_image_size

            # Add system prompt if specified
            if system_prompt_type and system_prompt_type != "None":
                if system_prompt_type == "custom" and custom_system_prompt:
                    gen_kwargs["use_system_prompt"] = "custom"
                    gen_kwargs["system_prompt"] = custom_system_prompt
                else:
                    gen_kwargs["use_system_prompt"] = system_prompt_type

            # Add bot task
            if bot_task:
                gen_kwargs["bot_task"] = bot_task

            print(f"[hunyuan_image] Generation kwargs: guidance={guidance_scale}, flow_shift={flow_shift}, taylor_cache={use_taylor_cache}")

            # Log memory before generation
            log_gpu_memory("Before generation")

            # Generate image using the model's generate_image API
            cot_text, outputs = self.model.generate_image(**gen_kwargs)

            # Log memory after generation
            log_gpu_memory("After generation")

            progress(0.9, desc="Processing output...")

            # Get the generated image
            if outputs and hasattr(outputs, 'images') and len(outputs.images) > 0:
                generated_image = outputs.images[0]
            elif isinstance(outputs, list) and len(outputs) > 0:
                generated_image = outputs[0]
            else:
                generated_image = outputs

            end_time = time.perf_counter()
            generation_time = end_time - start_time

            stats = {
                "time": generation_time,
                "seed": actual_seed,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "flow_shift": flow_shift,
                "size": f"{width}x{height}",
            }

            # Extract CoT reasoning text
            cot_reasoning = ""
            if cot_text and isinstance(cot_text, list) and len(cot_text) > 0:
                cot_reasoning = cot_text[0] if isinstance(cot_text[0], str) else str(cot_text[0])
            elif isinstance(cot_text, str):
                cot_reasoning = cot_text

            progress(1.0, desc="Done!")
            print(f"[hunyuan_image] Generation completed in {generation_time:.2f}s")

            return generated_image, stats, cot_reasoning

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, {"error": str(e)}, ""

    def generate_image_streaming(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        seed: int = -1,
        image_size: str = "1024x1024",
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        flow_shift: float = 3.0,
        system_prompt_type: Optional[str] = "en_unified",
        custom_system_prompt: Optional[str] = None,
        bot_task: str = "think_recaption",
        verbose: int = 2,
        infer_align_image_size: bool = True,
        use_taylor_cache: bool = False,
    ):
        """
        Generator that yields (cot_text, stats_text, image) tuples during generation.
        Streams the thinking/reasoning text in real-time with tok/s display.
        """
        if self.model is None:
            yield "", "No model loaded", None
            return

        # Parse image size
        if "x" in image_size:
            width, height = map(int, image_size.split("x"))
        else:
            width = height = int(image_size)

        # Handle seed
        actual_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**32)
        print(f"[hunyuan_image] Streaming generation: seed={actual_seed}, size={width}x{height}, steps={num_inference_steps}")

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "image": reference_image,
            "image_size": (height, width),
            "diff_infer_steps": num_inference_steps,
            "diff_guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": actual_seed,
            "verbose": 1,  # Keep minimal console output
            "use_taylor_cache": use_taylor_cache,
        }

        # Also set on generation_config directly (some model versions ignore kwargs)
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.diff_infer_steps = num_inference_steps
            self.model.generation_config.diff_guidance_scale = guidance_scale
            self.model.generation_config.flow_shift = flow_shift
            # Set image_size as list of strings (model's expected format)
            self.model.generation_config.image_size = [f"{width}x{height}"]
            print(f"[hunyuan_image] Set generation_config: steps={num_inference_steps}, guidance={guidance_scale}, flow_shift={flow_shift}, image_size={width}x{height}")

        # Add Taylor cache parameters when enabled
        if use_taylor_cache:
            gen_kwargs["taylor_cache_interval"] = 3  # Full computation every 3 steps
            gen_kwargs["taylor_cache_order"] = 2     # 2nd order Taylor approximation
            gen_kwargs["taylor_cache_enable_first_enhance"] = True
            gen_kwargs["taylor_cache_first_enhance_steps"] = 3
            gen_kwargs["taylor_cache_enable_tailing_enhance"] = True
            gen_kwargs["taylor_cache_tailing_enhance_steps"] = 1
            gen_kwargs["taylor_cache_low_freqs_order"] = 0
            gen_kwargs["taylor_cache_high_freqs_order"] = 2

        # Add infer_align_image_size for I2I mode
        if reference_image is not None:
            gen_kwargs["infer_align_image_size"] = infer_align_image_size

        # Add system prompt if specified
        if system_prompt_type and system_prompt_type != "None":
            if system_prompt_type == "custom" and custom_system_prompt:
                gen_kwargs["use_system_prompt"] = "custom"
                gen_kwargs["system_prompt"] = custom_system_prompt
            else:
                gen_kwargs["use_system_prompt"] = system_prompt_type

        # Add bot task
        if bot_task:
            gen_kwargs["bot_task"] = bot_task

        print(f"[hunyuan_image] Streaming with steps={num_inference_steps}, guidance={guidance_scale}, flow_shift={flow_shift}")

        # Log memory before generation
        log_gpu_memory("Before streaming generation")

        # Store generation result from thread
        generation_result = {"image": None, "error": None, "cot_text": None}

        def run_generation():
            try:
                cot_text, outputs = self.model.generate_image(**gen_kwargs)
                # Extract the generated image
                if outputs and hasattr(outputs, 'images') and len(outputs.images) > 0:
                    generation_result["image"] = outputs.images[0]
                elif isinstance(outputs, list) and len(outputs) > 0:
                    generation_result["image"] = outputs[0]
                else:
                    generation_result["image"] = outputs
                generation_result["cot_text"] = cot_text
            except Exception as e:
                import traceback
                traceback.print_exc()
                generation_result["error"] = str(e)

        # Start generation in background thread
        thread = threading.Thread(target=run_generation)
        start_time = time.perf_counter()
        thread.start()

        # Poll for completion while showing progress
        while thread.is_alive():
            # Check global stop flag
            global stop_generation
            if stop_generation:
                print("[hunyuan_image] Generation stopped by user")
                break

            elapsed = time.perf_counter() - start_time
            stats = f"Generating... | {elapsed:.1f}s"
            yield "", stats, None
            time.sleep(0.5)

        thread.join()

        # Log memory after generation complete
        log_gpu_memory("After diffusion complete")

        # Final result
        total_time = time.perf_counter() - start_time

        if generation_result["error"]:
            error_stats = f"Error: {generation_result['error']}"
            yield "", error_stats, None
        else:
            image = generation_result["image"]
            cot_text = generation_result.get("cot_text", "")
            # Format cot_text if it's a list
            if isinstance(cot_text, list) and len(cot_text) > 0:
                cot_text = cot_text[0] if isinstance(cot_text[0], str) else str(cot_text[0])
            elif not isinstance(cot_text, str):
                cot_text = str(cot_text) if cot_text else ""
            final_stats = f"{total_time:.1f}s | Seed: {actual_seed} | {width}x{height}"
            print(f"[hunyuan_image] Generation complete: {total_time:.1f}s")
            yield cot_text, final_stats, image

    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL image to temporary file and return path."""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vlm_img_{int(time.time())}_{id(image)}.png")
        image.save(temp_path)
        return temp_path

# Global backend instance
hunyuan_backend = None

def initialize_backend(models_dir: str = "models/LLM"):
    """Initialize the global HunyuanImage backend."""
    global hunyuan_backend
    hunyuan_backend = HunyuanImage3Backend(models_dir)


def refresh_models_handler():
    """Refresh the list of available models."""
    if hunyuan_backend is None:
        return gr.update(choices=["Backend not initialized"])
    models = hunyuan_backend.get_model_names()
    return gr.update(choices=models, value=models[0] if models else None)


def load_model_handler(
    model_name: str,
    dtype: str,
    num_gpus: int,
    max_memory_per_gpu: Optional[int],
    cpu_offload: bool = False,
    cpu_offload_ram: Optional[int] = None,
    flash_attention: bool = False,
    moe_impl: str = "eager",
    moe_drop_tokens: bool = True,
    progress=gr.Progress()
):
    """Handle model loading."""
    if hunyuan_backend is None:
        return "Backend not initialized"

    # Convert 0 to None for auto
    max_mem = None if max_memory_per_gpu == 0 else max_memory_per_gpu
    cpu_ram = None if cpu_offload_ram == 0 else cpu_offload_ram

    return hunyuan_backend.load_model(
        model_name=model_name,
        dtype=dtype,
        num_gpus=num_gpus,
        max_memory_per_gpu=max_mem,
        cpu_offload=cpu_offload,
        cpu_offload_ram=cpu_ram,
        flash_attention=flash_attention,
        moe_impl=moe_impl,
        moe_drop_tokens=moe_drop_tokens,
        progress=progress,
    )


def unload_model_handler():
    """Handle model unloading."""
    if hunyuan_backend is None:
        return "Backend not initialized"
    return hunyuan_backend.unload_model()


def status_handler():
    """Handle status request."""
    if hunyuan_backend is None:
        return "Backend not initialized"
    return hunyuan_backend.get_status()


def generate_handler(
    prompt: str,
    reference_image: Optional[Image.Image],
    seed: int,
    image_size: str,
    steps: int,
    guidance: float,
    flow_shift: float,
    system_prompt_type: str,
    custom_system_prompt: str,
    bot_task: str,
    verbose: int,
    infer_align_image_size: bool,
    use_taylor_cache: bool,
    progress=gr.Progress()
):
    """Generator handler for streaming image generation with real-time CoT display."""
    global stop_generation

    # Check if stop was requested before starting
    if stop_generation:
        stop_generation = False
        yield None, "Generation cancelled", ""
        return

    if hunyuan_backend is None:
        yield None, "Backend not initialized", ""
        return

    # Reset stop flag
    stop_generation = False

    # Use the streaming generator
    for cot_text, stats, image in hunyuan_backend.generate_image_streaming(
        prompt=prompt,
        reference_image=reference_image,
        seed=int(seed),
        image_size=image_size,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        flow_shift=float(flow_shift),
        system_prompt_type=system_prompt_type if system_prompt_type != "None" else None,
        custom_system_prompt=custom_system_prompt if system_prompt_type == "custom" else None,
        bot_task=bot_task,
        verbose=int(verbose),
        infer_align_image_size=infer_align_image_size,
        use_taylor_cache=use_taylor_cache,
    ):
        # Check stop flag during streaming
        if stop_generation:
            stop_generation = False
            yield image, stats + " [Stopped]", cot_text
            return

        yield image, stats, cot_text


def stop_generation_handler():
    """Set the stop flag to interrupt generation."""
    global stop_generation
    stop_generation = True
    return "Generation stop requested", ""


def clear_handler():
    """Clear the output and reset inputs."""
    return None, "", "", -1, ""




def create_ui():
    """Create the Gradio interface for HunyuanImage-3.0."""
    # Load saved settings
    saved_settings = load_settings()

    # Use Default theme like the main webui
    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )
    hunyuan_theme = themes.Default(**default_theme_args)

    hunyuan_css = """
    .stats-display {
        font-family: monospace;
        font-size: 14px;
        padding: 8px 12px;
        border-radius: 5px;
    }
    .green-btn {
        background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
        color: white !important;
        border: none !important;
    }
    .green-btn:hover {
        background: linear-gradient(to bottom right, #27ae60, #1e8449) !important;
    }
    .red-btn {
        background: linear-gradient(to bottom right, #e74c3c, #c0392b) !important;
        color: white !important;
        border: none !important;
    }
    .red-btn:hover {
        background: linear-gradient(to bottom right, #c0392b, #a93226) !important;
    }
    """

    # Get initial model list
    initial_models = hunyuan_backend.get_model_names() if hunyuan_backend else ["Initialize backend first"]
    num_gpus = hunyuan_backend.num_gpus if hunyuan_backend else 0

    with gr.Blocks(title="Chromaforge HunyuanImage-3.0", theme=hunyuan_theme, css=hunyuan_css) as demo:
        gr.Markdown("# Chromaforge HunyuanImage-3.0 Image Generator")

        with gr.Tabs():
            # Generate Tab
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=4,
                        )
                        reference_image = gr.Image(
                            label="Reference Image (optional, for I2I)",
                            type="pil",
                            height=250,
                        )

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Generated Image",
                            type="pil",
                            height=400,
                        )

                with gr.Row():
                    image_size = gr.Dropdown(
                        choices=IMAGE_SIZE_OPTIONS,
                        value=saved_settings.get("default_image_size", "1024x1024"),
                        label="Image Size",
                    )
                    steps_slider = gr.Slider(
                        minimum=8,
                        maximum=100,
                        value=saved_settings.get("default_steps", 50),
                        step=1,
                        label="Steps",
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=saved_settings.get("default_guidance", 2.5),
                        step=0.1,
                        label="Guidance Scale",
                    )
                    flow_shift_slider = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=saved_settings.get("default_flow_shift", 3.0),
                        step=0.1,
                        label="Flow Shift",
                    )
                    seed_input = gr.Number(
                        value=-1,
                        label="Seed (-1 = random)",
                    )

                with gr.Accordion("Advanced Prompt Settings", open=False):
                    with gr.Row():
                        system_prompt_type = gr.Dropdown(
                            choices=SYSTEM_PROMPT_OPTIONS,
                            value=saved_settings.get("system_prompt_type", "en_unified"),
                            label="System Prompt Type",
                        )
                        bot_task = gr.Dropdown(
                            choices=BOT_TASK_OPTIONS,
                            value=saved_settings.get("bot_task", "think_recaption"),
                            label="Bot Task",
                        )
                        verbose_slider = gr.Slider(
                            minimum=0,
                            maximum=2,
                            value=saved_settings.get("verbose", 2),
                            step=1,
                            label="Verbose Level",
                        )
                    custom_system_prompt = gr.Textbox(
                        label="Custom System Prompt",
                        placeholder="Enter custom system prompt (only used when System Prompt Type is 'custom')...",
                        lines=2,
                        value=saved_settings.get("custom_system_prompt", ""),
                    )
                    with gr.Row():
                        infer_align_image_size_check = gr.Checkbox(
                            value=saved_settings.get("infer_align_image_size", True),
                            label="Align I2I Output Size",
                            info="Match output size to input image (for I2I)",
                        )
                        use_taylor_cache_check = gr.Checkbox(
                            value=saved_settings.get("use_taylor_cache", False),
                            label="Taylor Cache",
                            info="Faster inference (experimental)",
                        )

                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary", elem_classes=["green-btn"], scale=2)
                    stop_btn = gr.Button("Stop", variant="stop", elem_classes=["red-btn"], scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                with gr.Accordion("CoT Reasoning (streams in real-time)", open=True):
                    cot_output = gr.Textbox(
                        label="Chain-of-Thought Reasoning",
                        interactive=False,
                        lines=10,
                        max_lines=10,
                        placeholder="Reasoning output will stream here when using think_recaption bot task...",
                        autoscroll=True,
                    )

                stats_display = gr.Textbox(
                    label="Stats",
                    interactive=False,
                    elem_classes=["stats-display"],
                )

            # Settings Tab
            with gr.TabItem("Settings"):
                gr.Markdown("### Model Settings")

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=initial_models,
                        value=saved_settings.get("model_name") or (initial_models[0] if initial_models else None),
                        label="Model",
                        interactive=True,
                    )
                    refresh_btn = gr.Button("Refresh", scale=0)

                with gr.Row():
                    dtype_dropdown = gr.Dropdown(
                        choices=["bfloat16", "float16", "fp8", "q4_nf4", "q8_fp16", "quanto_int4"],
                        value=saved_settings.get("dtype", "bfloat16"),
                        label="Precision",
                    )
                    num_gpus_slider = gr.Slider(
                        minimum=1,
                        maximum=max(8, num_gpus),
                        value=saved_settings.get("num_gpus", 1),
                        step=1,
                        label="Number of GPUs",
                    )
                    max_memory_slider = gr.Slider(
                        minimum=0,
                        maximum=80,
                        value=saved_settings.get("max_memory_per_gpu", 0),
                        step=1,
                        label="Max Memory/GPU (GB, 0=auto)",
                    )

                with gr.Row():
                    cpu_offload_check = gr.Checkbox(
                        value=saved_settings.get("cpu_offload", False),
                        label="CPU Offload",
                    )
                    cpu_ram_slider = gr.Slider(
                        minimum=0,
                        maximum=256,
                        value=saved_settings.get("cpu_offload_ram", 64),
                        step=8,
                        label="CPU RAM (GB)",
                    )
                    flash_attention_check = gr.Checkbox(
                        value=saved_settings.get("flash_attention", False),
                        label="Flash Attention 2",
                    )

                with gr.Row():
                    moe_impl_dropdown = gr.Dropdown(
                        choices=MOE_IMPL_OPTIONS,
                        value=saved_settings.get("moe_impl", "eager"),
                        label="MoE Implementation",
                        info="FlashInfer is ~3x faster (10min first-run compile)",
                    )
                    moe_drop_tokens_check = gr.Checkbox(
                        value=saved_settings.get("moe_drop_tokens", True),
                        label="MoE Drop Tokens",
                    )

                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="primary", elem_classes=["green-btn"])
                    unload_btn = gr.Button("Unload Model", variant="secondary", elem_classes=["red-btn"])

                status_display = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="No model loaded",
                )

                gr.Markdown("### Default Generation Settings")

                with gr.Row():
                    default_steps = gr.Slider(
                        minimum=8,
                        maximum=100,
                        value=saved_settings.get("default_steps", 50),
                        step=1,
                        label="Default Steps",
                    )
                    default_guidance = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=saved_settings.get("default_guidance", 2.5),
                        step=0.1,
                        label="Default Guidance",
                    )
                    default_flow_shift = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=saved_settings.get("default_flow_shift", 3.0),
                        step=0.1,
                        label="Default Flow Shift",
                    )
                    default_size = gr.Dropdown(
                        choices=IMAGE_SIZE_OPTIONS,
                        value=saved_settings.get("default_image_size", "1024x1024"),
                        label="Default Size",
                    )

                with gr.Row():
                    default_infer_align_image_size = gr.Checkbox(
                        value=saved_settings.get("infer_align_image_size", True),
                        label="Default Align I2I Output Size",
                    )
                    default_use_taylor_cache = gr.Checkbox(
                        value=saved_settings.get("use_taylor_cache", False),
                        label="Default Taylor Cache",
                    )

                gr.Markdown("### Default Prompt Settings")

                with gr.Row():
                    default_system_prompt_type = gr.Dropdown(
                        choices=SYSTEM_PROMPT_OPTIONS,
                        value=saved_settings.get("system_prompt_type", "en_unified"),
                        label="Default System Prompt Type",
                    )
                    default_bot_task = gr.Dropdown(
                        choices=BOT_TASK_OPTIONS,
                        value=saved_settings.get("bot_task", "think_recaption"),
                        label="Default Bot Task",
                    )
                    default_verbose = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=saved_settings.get("verbose", 2),
                        step=1,
                        label="Default Verbose Level",
                    )

                default_custom_system_prompt = gr.Textbox(
                    label="Default Custom System Prompt",
                    placeholder="Enter default custom system prompt...",
                    lines=2,
                    value=saved_settings.get("custom_system_prompt", ""),
                )

                save_btn = gr.Button("Save Settings")
                save_status = gr.Textbox(label="", interactive=False, visible=False)

        # Event handlers
        refresh_btn.click(
            refresh_models_handler,
            outputs=[model_dropdown],
        )

        load_btn.click(
            load_model_handler,
            inputs=[
                model_dropdown, dtype_dropdown, num_gpus_slider,
                max_memory_slider, cpu_offload_check, cpu_ram_slider,
                flash_attention_check, moe_impl_dropdown, moe_drop_tokens_check,
            ],
            outputs=[status_display],
        )

        unload_btn.click(
            unload_model_handler,
            outputs=[status_display],
        )

        generate_btn.click(
            generate_handler,
            inputs=[
                prompt_input, reference_image, seed_input, image_size,
                steps_slider, guidance_slider, flow_shift_slider,
                system_prompt_type, custom_system_prompt, bot_task, verbose_slider,
                infer_align_image_size_check, use_taylor_cache_check,
            ],
            outputs=[output_image, stats_display, cot_output],
        )

        stop_btn.click(
            stop_generation_handler,
            outputs=[stats_display, cot_output],
        )

        clear_btn.click(
            clear_handler,
            outputs=[output_image, prompt_input, stats_display, seed_input, cot_output],
        )

        save_btn.click(
            save_settings,
            inputs=[
                model_dropdown, dtype_dropdown, num_gpus_slider,
                max_memory_slider, cpu_offload_check, cpu_ram_slider,
                flash_attention_check, moe_impl_dropdown, moe_drop_tokens_check,
                default_steps, default_guidance, default_flow_shift, default_size,
                default_infer_align_image_size, default_use_taylor_cache,
                default_system_prompt_type, default_custom_system_prompt,
                default_bot_task, default_verbose,
            ],
            outputs=[save_status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Chromaforge HunyuanImage-3.0 Backend")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/LLM",
        help="Directory containing HunyuanImage models (default: models/LLM)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7864,
        help="Port to run the server on (default: 7864)",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Listen on 0.0.0.0 to allow external connections",
    )

    args = parser.parse_args()

    # Initialize the backend
    print(f"[hunyuan_image] Initializing backend with models_dir={args.models_dir}")
    initialize_backend(args.models_dir)

    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
