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
from PIL import Image, PngImagePlugin
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
    "1536x512",
    "1664x512",
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

# Modules to keep on GPU during CPU offloading (for FP8 performance)
# Attention layers are small but latency-sensitive - keeping them on GPU improves speed
HUNYUAN_GPU_MODULES = [
    'self_attn',  # Attention layers - small but latency-critical
    'input_layernorm', 'post_attention_layernorm',  # LayerNorms before/after attention
    'wte', 'ln_f',  # Embeddings and final norm
    'vae',  # VAE should stay on GPU
    'vision_model', 'vision_aligner',  # Vision encoder
    'final_layer', 'patch_embed',  # Image generation head
]

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

# SDNQ dtype mapping: UI option -> SDNQ weights_dtype
SDNQ_DTYPE_MAP = {
    "sdnq_int8": "int8",
    "sdnq_fp8": "float8_e4m3fn",
    "sdnq_int4": "int4",
    "sdnq_uint4": "uint4",
}

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
    "keep_attention_on_gpu": False,
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
    keep_attention_on_gpu: bool,
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
        "keep_attention_on_gpu": keep_attention_on_gpu,
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
    Convert model linear layers to FP8 with per-tensor scaling at runtime.
    Saves ~50% memory. Works on any GPU (older GPUs compute in fp16/fp32).

    Note: This runtime conversion may not work well with device_map='auto' because
    weights may be on meta device. For best results, use convert_to_fp8.py to
    pre-convert the model offline.

    Args:
        model: The model to convert
        skip_patterns: List of layer name patterns to skip (e.g., ['lm_head', 'embed'])

    Returns:
        model with FP8 weights where safe
    """
    import torch.nn as nn

    if skip_patterns is None:
        skip_patterns = ['embed', 'lm_head', 'wte', 'wpe', 'norm', 'visual']

    # Normalize patterns
    skip_patterns = [p.lower() for p in skip_patterns]

    converted_count = 0
    skipped_meta = 0
    skipped_pattern = 0
    skipped_range = 0
    skipped_fp8 = 0
    total_params_before = 0
    total_params_after = 0

    # First pass: count layers and check for meta tensors
    total_linear = 0
    meta_count = 0
    for name, child in model.named_modules():
        if isinstance(child, nn.Linear):
            total_linear += 1
            if child.weight.data.device.type == 'meta':
                meta_count += 1

    print(f"[FP8] Found {total_linear} linear layers, {meta_count} on meta device")

    if meta_count > total_linear * 0.5:
        print(f"[FP8] WARNING: Most weights are on meta device (device_map offloading)")
        print(f"[FP8] Runtime FP8 conversion will skip these layers.")
        print(f"[FP8] For proper FP8 support, pre-convert the model using:")
        print(f"[FP8]   python convert_to_fp8.py --input <model_path> --output <output_path>")

    for name, child in model.named_modules():
        if isinstance(child, nn.Linear) and not hasattr(child, 'fp8_converted'):
            weight = child.weight.data
            original_dtype = weight.dtype

            # Skip if already FP8
            if original_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                skipped_fp8 += 1
                continue

            # Skip meta tensors
            if weight.device.type == 'meta':
                skipped_meta += 1
                continue

            # Skip patterns that shouldn't be converted
            if any(pattern in name.lower() for pattern in skip_patterns):
                skipped_pattern += 1
                continue

            # Check dynamic range
            weight_float = weight.float()
            abs_max = weight_float.abs().max()
            non_zero = weight_float[weight_float != 0]
            abs_min = non_zero.abs().min() if non_zero.numel() > 0 else torch.tensor(1.0)

            if abs_max > 0 and abs_min > 0:
                dynamic_range = abs_max / abs_min
                if dynamic_range > 1e6:
                    skipped_range += 1
                    continue

            # Calculate memory before
            param_bytes_before = weight.numel() * weight.element_size()
            total_params_before += param_bytes_before

            # Compute scale factor
            if abs_max == 0:
                abs_max = torch.tensor(1.0, device=weight.device, dtype=torch.float32)
            scale = (abs_max / 448.0).float()

            # Convert to FP8
            fp8_weight = (weight.float() / scale).to(torch.float8_e4m3fn)

            # Store FP8 weight and scale
            child.weight = nn.Parameter(fp8_weight, requires_grad=False)
            child.register_buffer('scale_weight', scale.view(1))
            child.computation_dtype = original_dtype

            # Calculate memory after
            total_params_after += fp8_weight.numel() * fp8_weight.element_size()
            total_params_after += 4

            # Replace forward method
            child.original_forward = child.forward
            child.forward = lambda x, m=child: _fp8_linear_forward(m, x)
            child.fp8_converted = True

            converted_count += 1

    # Print summary
    total_skipped = skipped_meta + skipped_pattern + skipped_range + skipped_fp8
    print(f"[FP8] Converted {converted_count} layers to FP8")
    print(f"[FP8] Skipped {total_skipped}: {skipped_meta} meta, {skipped_pattern} pattern, {skipped_range} range, {skipped_fp8} already FP8")

    if total_params_before > 0:
        mem_before_gb = total_params_before / (1024**3)
        mem_after_gb = total_params_after / (1024**3)
        reduction = (1 - mem_after_gb / mem_before_gb) * 100
        print(f"[FP8] Memory: {mem_before_gb:.2f}GB -> {mem_after_gb:.2f}GB ({reduction:.1f}% reduction)")

    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def estimate_attention_gpu_memory(num_layers: int = 32, hidden_size: int = 4096,
                                   num_heads: int = 32, num_kv_heads: int = 8,
                                   head_dim: int = 128, bytes_per_param: float = 2.0):
    """
    Estimate GPU memory needed for attention layers + non-MoE components.

    Returns memory in GB.
    """
    # Per-layer attention: Q, K, V, O projections
    q_params = hidden_size * (num_heads * head_dim)  # 4096 * 4096
    k_params = hidden_size * (num_kv_heads * head_dim)  # 4096 * 1024
    v_params = hidden_size * (num_kv_heads * head_dim)  # 4096 * 1024
    o_params = (num_heads * head_dim) * hidden_size  # 4096 * 4096
    attn_params_per_layer = q_params + k_params + v_params + o_params

    # LayerNorms per layer (2 per layer)
    ln_params_per_layer = 2 * hidden_size * 2  # input_ln + post_attn_ln, weight + bias

    # MoE gate + shared_mlp per layer (small)
    gate_params_per_layer = hidden_size * 64  # gate to 64 experts
    shared_mlp_params = hidden_size * 3072 * 3  # up, gate, down projections

    # Per-layer total (non-expert)
    per_layer = attn_params_per_layer + ln_params_per_layer + gate_params_per_layer + shared_mlp_params
    transformer_params = per_layer * num_layers

    # Embeddings
    vocab_size = 133120
    embed_params = vocab_size * hidden_size

    # Vision encoder (SigLIP2) ~400M params
    vision_params = 400_000_000

    # VAE ~80M params
    vae_params = 80_000_000

    # Final LN + lm_head
    final_params = hidden_size + vocab_size * hidden_size

    total_params = transformer_params + embed_params + vision_params + vae_params + final_params
    total_bytes = total_params * bytes_per_param
    total_gb = total_bytes / (1024**3)

    return total_gb


def create_attention_priority_device_map(num_layers: int = 32, num_experts: int = 64):
    """
    Create a device_map that keeps attention layers on GPU while offloading MoE experts to CPU.

    Based on HunyuanImage-3.0 model structure:
    - model.layers.X.self_attn: attention -> GPU
    - model.layers.X.mlp: MoE block (experts + gate + shared) -> CPU
    - Global embeddings/heads -> GPU
    """
    device_map = {}

    # 1. Global modules to GPU (Lightweight components)
    gpu_modules = [
        'model.wte', 'model.ln_f',
        'vae', 'vision_model', 'vision_aligner',
        'final_layer', 'patch_embed',
        'time_embed', 'time_embed_2',
        'timestep_emb', 'guidance_emb', 'timestep_r_emb',
        'lm_head',
    ]
    for module in gpu_modules:
        device_map[module] = 0

    # 2. Per-layer mapping
    for i in range(num_layers):
        # Attention components -> GPU (Latency sensitive, relatively small)
        device_map[f'model.layers.{i}.self_attn'] = 0
        device_map[f'model.layers.{i}.input_layernorm'] = 0
        device_map[f'model.layers.{i}.post_attention_layernorm'] = 0
        
        # Whole MLP block (Experts + Gate + Shared) -> CPU
        # Mapping the parent 'mlp' to CPU is safer than mapping just 'mlp.experts'
        # because it prevents the parent container from defaulting to GPU.
        device_map[f'model.layers.{i}.mlp'] = 'cpu'

    # Estimate GPU memory for non-MoE components
    gpu_memory_gb = estimate_attention_gpu_memory(num_layers=num_layers)

    experts_on_cpu = num_layers * num_experts
    print(f"[device_map] Attention-priority: ~{gpu_memory_gb:.1f}GB on GPU, {experts_on_cpu} MoE experts on CPU (entire MLP block offloaded)")

    return device_map, gpu_memory_gb


def patch_accelerate_for_fp8(fp8_weight_names: set = None):
    """
    Patch accelerate's disk offload to handle FP8 tensors.
    FP8 stored as uint8 bytes (lossless), restored as FP8 when loading.

    Args:
        fp8_weight_names: Set of weight names that are FP8 (e.g., "model.layers.0.mlp.experts.0.down_proj.weight")
                          If provided, these will be restored to FP8 when loaded from disk.
    """
    from accelerate.utils import offload as accel_offload
    from accelerate.utils import modeling as accel_modeling
    import os

    # Store FP8 weight names for lookup during load
    if fp8_weight_names:
        if not hasattr(accel_offload, '_fp8_weight_names'):
            accel_offload._fp8_weight_names = set()
        accel_offload._fp8_weight_names.update(fp8_weight_names)
        print(f"[FP8] Registered {len(fp8_weight_names)} FP8 weight names for disk offload restoration")

    if hasattr(accel_offload, '_fp8_patched'):
        return

    original_load_offloaded_weight = accel_offload.load_offloaded_weight

    _load_count = [0]
    _fp8_restore_count = [0]
    _fp8_already_count = [0]
    _debug_logged = [0]
    _dtype_counts = [{}]

    def patched_load_offloaded_weight(weight_file, weight_info):
        _load_count[0] += 1
        weight = original_load_offloaded_weight(weight_file, weight_info)

        # Track dtype distribution
        dtype_str = str(weight.dtype)
        _dtype_counts[0][dtype_str] = _dtype_counts[0].get(dtype_str, 0) + 1

        # Extract weight name from file path (e.g., "model.layers.0.mlp.weight.dat" -> "model.layers.0.mlp.weight")
        weight_name = os.path.basename(weight_file)
        if weight_name.endswith('.dat'):
            weight_name = weight_name[:-4]

        # Check if this weight should be FP8
        fp8_names = getattr(accel_offload, '_fp8_weight_names', set())
        is_fp8_weight = weight_name in fp8_names

        if is_fp8_weight:
            if weight.dtype == torch.uint8:
                # FP8 was saved as uint8 bytes, restore it
                weight = weight.view(torch.float8_e4m3fn)
                _fp8_restore_count[0] += 1
                if _fp8_restore_count[0] <= 3:
                    print(f"[FP8 Load] Restored from uint8: {weight_name}")
            elif weight.dtype == torch.float8_e4m3fn:
                # Already FP8, no conversion needed
                _fp8_already_count[0] += 1
                if _fp8_already_count[0] <= 3:
                    print(f"[FP8 Load] Already FP8: {weight_name}")
            else:
                # Unexpected dtype for FP8 weight - log for debug
                if _debug_logged[0] < 5:
                    print(f"[FP8 Load DEBUG] FP8 weight loaded as {weight.dtype}: {weight_name}")
                    _debug_logged[0] += 1

        return weight

    def print_offload_stats():
        fp8_names = getattr(accel_offload, '_fp8_weight_names', set())
        print(f"[FP8 Offload Stats] Loaded: {_load_count[0]}, FP8 Restored: {_fp8_restore_count[0]}, Already FP8: {_fp8_already_count[0]}")
        print(f"[FP8 Offload Stats] Registered FP8 weights: {len(fp8_names)}")
        print(f"[FP8 Offload Stats] Loaded dtype distribution: {_dtype_counts[0]}")

    # Patch in BOTH modules to ensure it's used regardless of import order
    accel_offload.load_offloaded_weight = patched_load_offloaded_weight
    if hasattr(accel_modeling, 'load_offloaded_weight'):
        accel_modeling.load_offloaded_weight = patched_load_offloaded_weight

    accel_offload.print_fp8_offload_stats = print_offload_stats
    accel_offload._fp8_patched = True
    print("[FP8] Patched accelerate disk offload for FP8 support")


def apply_fp8_linear_forward_patch():
    """
    Patch nn.Linear.forward globally to handle FP8 dequantization.

    This approach doesn't modify module.weight, so it works with accelerate's
    CPU offloading which manages weight tensors directly.

    The patched forward:
    1. Checks if module has scale_weight (FP8 layer)
    2. If weight is FP8, dequantizes on-the-fly for computation
    3. Never modifies module.weight - accelerate can manage it normally
    """
    import torch.nn as nn
    import torch.nn.functional as F

    if hasattr(nn.Linear, '_original_forward_fp8'):
        print("[FP8] Linear forward already patched")
        return

    original_forward = nn.Linear.forward

    # Debug counters
    _fp8_dequant_count = [0]
    _non_fp8_count = [0]
    _has_scale_no_fp8 = [0]
    _debug_printed = [False]
    _dequant_values_logged = [0]

    def fp8_linear_forward(self, input):
        weight = self.weight

        # Check if this is an FP8 layer with scale
        has_scale = hasattr(self, 'scale_weight')
        is_fp8 = weight.dtype == torch.float8_e4m3fn

        if has_scale and is_fp8:
            _fp8_dequant_count[0] += 1
            # Get scale, move to same device as input if needed
            scale = self.scale_weight
            if scale.device != input.device:
                scale = scale.to(input.device)

            # Move weight to input device if needed (accelerate may have it on different device)
            if weight.device != input.device:
                weight = weight.to(input.device)

            # Dequantize: FP8 * scale -> computation dtype
            # Use input dtype for computation (typically bf16)
            weight_dequant = weight.to(input.dtype) * scale.to(input.dtype)

            # Log first few dequantizations to verify values
            if _dequant_values_logged[0] < 3:
                scale_val = scale.item() if scale.numel() == 1 else scale.mean().item()
                w_min = weight_dequant.min().item()
                w_max = weight_dequant.max().item()
                w_mean = weight_dequant.mean().item()
                print(f"[FP8 Dequant #{_dequant_values_logged[0]}] scale={scale_val:.6f}, dequant range=[{w_min:.4f}, {w_max:.4f}], mean={w_mean:.6f}")
                _dequant_values_logged[0] += 1

            # Compute linear with dequantized weight
            bias = self.bias
            if bias is not None and bias.device != input.device:
                bias = bias.to(input.device)
            return F.linear(input, weight_dequant, bias)
        elif has_scale and not is_fp8:
            _has_scale_no_fp8[0] += 1
            # Log first few cases of scale but no FP8
            if _has_scale_no_fp8[0] <= 3:
                print(f"[FP8 DEBUG] Has scale but weight is {weight.dtype}: {type(self).__name__}")
        else:
            _non_fp8_count[0] += 1

        # Print stats periodically
        total = _fp8_dequant_count[0] + _non_fp8_count[0] + _has_scale_no_fp8[0]
        if total > 0 and total % 1000 == 0 and not _debug_printed[0]:
            print(f"[FP8 Forward Stats] Dequant: {_fp8_dequant_count[0]}, Non-FP8: {_non_fp8_count[0]}, Scale+NoFP8: {_has_scale_no_fp8[0]}")
            _debug_printed[0] = True

        # Non-FP8 layer: use original forward
        return original_forward(self, input)

    def print_fp8_forward_stats():
        print(f"[FP8 Forward Stats] Dequant: {_fp8_dequant_count[0]}, Non-FP8: {_non_fp8_count[0]}, Scale+NoFP8: {_has_scale_no_fp8[0]}")

    nn.Linear._print_fp8_stats = staticmethod(print_fp8_forward_stats)

    nn.Linear._original_forward_fp8 = original_forward
    nn.Linear.forward = fp8_linear_forward
    print("[FP8] Patched nn.Linear.forward for FP8 dequantization")


def apply_fp8_hooks_to_model(model, computation_dtype=torch.bfloat16):
    """
    Enable FP8 support for a pre-converted model.

    This patches nn.Linear.forward globally to dequantize FP8 weights on-the-fly.
    Compatible with accelerate's CPU offloading since we don't modify module.weight.

    Args:
        model: Model with pre-converted FP8 weights and scale_weight buffers
        computation_dtype: Dtype to use for computation (default: bfloat16)

    Returns:
        model (unchanged, but nn.Linear.forward is now patched globally)
    """
    import torch.nn as nn

    # Apply the global Linear forward patch
    apply_fp8_linear_forward_patch()

    # Count FP8 layers for logging
    fp8_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'scale_weight'):
            fp8_count += 1

    if fp8_count > 0:
        print(f"[FP8] Model has {fp8_count} FP8 layers with scales (dequantization enabled)")

    return model


def is_fp8_converted_model(model_path: str) -> bool:
    """Check if a model was pre-converted to FP8 by checking config.json."""
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('fp8_converted', False)
        except (json.JSONDecodeError, IOError):
            pass
    return False


def load_fp8_model_with_offload(
    model_path: str,
    device_map: dict | str = "auto",
    max_memory: dict | None = None,
    offload_folder: str | None = None,
    **model_kwargs
):
    """
    Load a pre-converted FP8 model while preserving FP8 dtypes.

    This bypasses transformers' from_pretrained which converts FP8 to f32 during
    device placement. Instead we:
    1. Create model structure with empty weights
    2. Compute device map
    3. Load weights directly from safetensors preserving dtype
    4. Dispatch model to devices

    Args:
        model_path: Path to the FP8 model directory
        device_map: Device map for model placement ("auto" or dict)
        max_memory: Max memory per device
        offload_folder: Folder for disk offloading
        **model_kwargs: Additional kwargs for model config

    Returns:
        Model with FP8 weights preserved
    """
    import json
    import os
    from pathlib import Path
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
    from accelerate.utils import set_module_tensor_to_device, get_balanced_memory

    model_path = Path(model_path)
    print(f"[FP8 Loader] Loading FP8 model from {model_path}")

    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Apply model kwargs to config
    for key, value in model_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create model with empty weights
    print("[FP8 Loader] Creating model structure with empty weights...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Get model's tied weights info
    model.tie_weights()

    # Find all safetensors files
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]
        weight_map = None

    print(f"[FP8 Loader] Found {len(shard_files)} weight shard(s)")

    # Compute device map if "auto"
    if device_map == "auto":
        print("[FP8 Loader] Computing device map...")

        # Get balanced memory allocation
        if max_memory is None:
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=getattr(model, "_no_split_modules", []),
                dtype=torch.bfloat16,  # Use bf16 for memory estimation (FP8 is similar)
                low_zero=False,
            )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=getattr(model, "_no_split_modules", []),
            dtype=torch.bfloat16,
        )

    print(f"[FP8 Loader] Device map computed, loading weights to CPU...")

    # Load ALL weights to CPU first, preserving dtypes
    # dispatch_model will handle moving them to correct devices
    fp8_loaded = 0
    bf16_loaded = 0
    f32_loaded = 0
    fp8_weight_names = set()  # Track FP8 weight names for disk offload restoration

    for shard_name in shard_files:
        shard_path = model_path / shard_name
        if not shard_path.exists():
            print(f"[FP8 Loader] Warning: shard {shard_name} not found")
            continue

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Skip scale_weight buffers - we'll load them after dispatch
                if key.endswith('.scale_weight'):
                    continue

                tensor = f.get_tensor(key)

                # Load to CPU, preserving original dtype
                try:
                    set_module_tensor_to_device(
                        model,
                        key,
                        "cpu",
                        value=tensor,
                        dtype=tensor.dtype,  # Preserve original dtype!
                    )

                    # Count dtypes and track FP8 weight names
                    if tensor.dtype == torch.float8_e4m3fn:
                        fp8_loaded += 1
                        fp8_weight_names.add(key)
                    elif tensor.dtype == torch.bfloat16:
                        bf16_loaded += 1
                    elif tensor.dtype == torch.float32:
                        f32_loaded += 1
                except Exception as e:
                    # Some keys might not map to model parameters
                    pass

    print(f"[FP8 Loader] Loaded weights: {fp8_loaded} FP8, {bf16_loaded} bf16, {f32_loaded} f32")

    # Patch accelerate BEFORE dispatch to handle FP8 disk offload
    # Pass the FP8 weight names so they can be restored to FP8 when loaded from disk
    if offload_folder and fp8_weight_names:
        patch_accelerate_for_fp8(fp8_weight_names)

    # CRITICAL: Apply FP8 Linear forward patch BEFORE dispatch_model
    # dispatch_model creates hooks that store references to module.forward
    # If we patch after dispatch, the hooks won't use our patched forward
    apply_fp8_linear_forward_patch()

    # Dispatch model to devices with offloading
    # This moves weights from CPU to their target devices based on device_map
    if offload_folder:
        print(f"[FP8 Loader] Dispatching model with DISK offload to: {offload_folder}")
    else:
        print("[FP8 Loader] Dispatching model with CPU offload (no disk)")
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
    )

    # Now load scale_weight buffers AFTER dispatch
    # This ensures scales are on the same device as their weights
    scales_loaded = 0
    scales_failed = 0
    scales_debug_logged = 0
    for shard_name in shard_files:
        shard_path = model_path / shard_name
        if not shard_path.exists():
            continue

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith('.scale_weight'):
                    module_name = key.rsplit('.scale_weight', 1)[0]
                    try:
                        module = model.get_submodule(module_name)
                        scale = f.get_tensor(key)
                        # Move scale to same device as the weight
                        if hasattr(module, 'weight') and module.weight is not None:
                            weight_device = module.weight.device
                            weight_dtype = module.weight.dtype
                            if weight_device.type != 'meta':
                                scale = scale.to(weight_device)
                            # Log first few for debugging
                            if scales_debug_logged < 3:
                                print(f"[FP8 Scale Debug] {module_name}: weight={weight_dtype}@{weight_device}, scale={scale.shape}@{scale.device}")
                                scales_debug_logged += 1
                        module.register_buffer('scale_weight', scale)
                        scales_loaded += 1
                    except (AttributeError, Exception) as e:
                        scales_failed += 1
                        if scales_failed <= 3:
                            print(f"[FP8 Scale Debug] Failed to load scale for {module_name}: {e}")

    print(f"[FP8 Loader] Loaded {scales_loaded} scale buffers")

    # Diagnostic: check actual weight dtypes after dispatch
    dtype_counts = {}
    scale_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            dtype_key = str(module.weight.dtype)
            device_key = str(module.weight.device)
            key = f"{dtype_key}@{device_key}"
            dtype_counts[key] = dtype_counts.get(key, 0) + 1
        if hasattr(module, 'scale_weight'):
            scale_count += 1
    print(f"[FP8 Loader] After dispatch - weight dtype distribution:")
    for key, count in sorted(dtype_counts.items()):
        print(f"  {key}: {count}")
    print(f"[FP8 Loader] Modules with scale_weight: {scale_count}")

    # Load generation_config
    from transformers import GenerationConfig
    gen_config_path = model_path / "generation_config.json"
    if gen_config_path.exists():
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        print(f"[FP8 Loader] Loaded generation_config")

    print(f"[FP8 Loader] Model loaded successfully with FP8 weights preserved")
    print(f"[FP8 Loader] DEBUG: Look for '[FP8 Dequant #]' messages during generation to verify dequantization is working")

    return model


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
    from sdnq import SDNQConfig
    from sdnq.common import use_torch_compile as sdnq_triton_available
    SDNQ_AVAILABLE = True
    print("SDNQ available")
except ImportError:
    SDNQ_AVAILABLE = False
    sdnq_triton_available = False
    print("Warning: sdnq not installed. SDNQ quantization unavailable.")

try:
    from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Quantized model saving unavailable.")


def create_sdnq_config(
    weights_dtype: str = "int8",
    use_quantized_matmul: bool = True,
    quantization_device: str = None,
    return_device: str = None,
) -> "SDNQConfig":
    """
    Create SDNQConfig for HunyuanImage-3.0 with appropriate skip modules.

    Args:
        weights_dtype: Quantization dtype (int8, int4, uint4, float8_e4m3fn)
        use_quantized_matmul: Use Triton-optimized quantized matmul kernels
        quantization_device: Device for quantization calculation (e.g., "cpu" to save VRAM)
        return_device: Device for quantized weights (e.g., "cpu" for offloading)

    Returns:
        SDNQConfig instance
    """
    return SDNQConfig(
        weights_dtype=weights_dtype,
        group_size=0,  # auto
        use_quantized_matmul=use_quantized_matmul and sdnq_triton_available,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=[
            # VAE - pixel reconstruction
            'vae',
            # Vision encoder
            'vision_model', 'vision_aligner',
            # Image generation head
            'final_layer', 'patch_embed',
            # Timestep/guidance embeddings
            'time_embed', 'time_embed_2', 'timestep_emb', 'guidance_emb', 'timestep_r_emb',
            # MoE gates - must stay fp32
            'wg',
            # Token embeddings
            'wte', 'lm_head',
            # LayerNorms
            'ln_f', 'layernorm', 'input_layernorm', 'post_attention_layernorm',
            'key_layernorm', 'query_layernorm',
        ],
    )


def clear_gpu_memory(sync: bool = True):
    """
    Aggressively clear GPU memory to reduce memory spikes.

    Args:
        sync: Whether to synchronize CUDA before clearing (ensures all ops complete)
    """
    gc.collect()
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def apply_quantization_dtype_fix(model):
    """
    Monkey-patch the model to fix dtype mismatches when using quantization.

    When BitsAndBytes quantization is applied, the transformer backbone produces
    hidden_states in a different dtype than the skipped modules (patch_embed,
    timestep_emb, etc.). The scatter_ operations require matching dtypes.

    This patch wraps the problematic methods to cast src tensors before scatter.

    Also adds memory clearing after each major embedding operation to reduce
    memory spikes during the initial phase of image generation.
    """
    import types

    # Store original methods
    original_instantiate_vae_image_tokens = model.instantiate_vae_image_tokens
    original_instantiate_vit_image_tokens = model.instantiate_vit_image_tokens
    original_instantiate_continuous_tokens = model.instantiate_continuous_tokens
    original_instantiate_guidance_tokens = model.instantiate_guidance_tokens
    original_instantiate_timestep_r_tokens = model.instantiate_timestep_r_tokens

    def patched_instantiate_vae_image_tokens(self, hidden_states, timesteps, images, image_mask, guidance=None, timesteps_r=None):
        """Patched version with dtype casting for quantization compatibility and memory clearing."""
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
            # Clear intermediate tensors to reduce memory spike
            del image_seq, t_emb, index, image_scatter_index
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
                    del image_i_seq_list
                else:
                    raise TypeError(f"image_i should be a torch.Tensor or a list, got {type(image_i)}")

                image_i_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                hidden_states[i:i + 1].scatter_(
                    dim=1,
                    index=image_i_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_i_seq.reshape(1, -1, n_embd).to(target_dtype),  # Cast for quantization compatibility
                )
                del image_i_seq, t_i_emb, image_i_index
            del index

        # Clear GPU memory after VAE token instantiation (major memory consumer)
        clear_gpu_memory(sync=False)
        return hidden_states

    def patched_instantiate_vit_image_tokens(self, hidden_states, images, image_masks, **image_kwargs):
        """Patched version with dtype casting for quantization compatibility and memory clearing."""
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
            # Clear intermediate tensors
            del image_embeds, image_scatter_index
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
                del image_embed, image_scatter_index
        else:
            raise ValueError(f"und_images should be Tensor or List, but got {type(images)}")

        del index
        # Clear GPU memory after VIT token instantiation (vision encoder is memory-heavy)
        clear_gpu_memory(sync=False)
        return hidden_states

    def patched_instantiate_continuous_tokens(self, hidden_states, timesteps=None, timesteps_index=None):
        """Patched version with dtype casting for quantization compatibility and memory clearing."""
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
                del timestep_src
        else:
            timesteps_src = self.timestep_emb(timesteps.reshape(-1))
            hidden_states.scatter_(
                dim=1,
                index=timesteps_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=timesteps_src.reshape(bsz, -1, n_embd).to(target_dtype),  # Cast
            )
            del timesteps_src

        return hidden_states

    def patched_instantiate_guidance_tokens(self, hidden_states, guidance=None, guidance_index=None):
        """Patched version with dtype casting for quantization compatibility and memory clearing."""
        bsz, seqlen, n_embd = hidden_states.shape
        target_dtype = hidden_states.dtype

        guidance_src = self.guidance_emb(guidance.reshape(-1))
        hidden_states.scatter_(
            dim=1,
            index=guidance_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=guidance_src.reshape(bsz, -1, n_embd).to(target_dtype),  # Cast
        )
        del guidance_src

        return hidden_states

    def patched_instantiate_timestep_r_tokens(self, hidden_states, timesteps_r=None, timesteps_r_index=None):
        """Patched version with dtype casting for quantization compatibility and memory clearing."""
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
                del timestep_r_src
        else:
            timesteps_r_src = self.timestep_r_emb(timesteps_r.reshape(-1))
            hidden_states.scatter_(
                dim=1,
                index=timesteps_r_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=timesteps_r_src.reshape(bsz, -1, n_embd).to(target_dtype),  # Cast
            )
            del timesteps_r_src

        return hidden_states

    # Apply patches
    model.instantiate_vae_image_tokens = types.MethodType(patched_instantiate_vae_image_tokens, model)
    model.instantiate_vit_image_tokens = types.MethodType(patched_instantiate_vit_image_tokens, model)
    model.instantiate_continuous_tokens = types.MethodType(patched_instantiate_continuous_tokens, model)
    model.instantiate_guidance_tokens = types.MethodType(patched_instantiate_guidance_tokens, model)
    model.instantiate_timestep_r_tokens = types.MethodType(patched_instantiate_timestep_r_tokens, model)

    # === Patch HunyuanStaticCache for chunked initialization and dtype compatibility ===
    # The KV cache buffers may be in a different dtype than key/value states with CPU offload
    # Chunked initialization reduces memory spikes by allocating cache incrementally
    import importlib
    model_module = type(model).__module__
    hunyuan_module = importlib.import_module(model_module)

    if hasattr(hunyuan_module, 'HunyuanStaticCache'):
        HunyuanStaticCache = hunyuan_module.HunyuanStaticCache

        # Configuration for chunked cache
        INITIAL_CACHE_SIZE = 512  # Start with smaller allocation
        CACHE_GROWTH_FACTOR = 2.0  # Double when growing
        MIN_GROWTH_SIZE = 256  # Minimum growth increment

        def chunked_lazy_initialization(layer_cache, key_states, initial_size=INITIAL_CACHE_SIZE):
            """
            Initialize cache with a smaller initial size instead of full max_seq_len.
            This reduces the initial memory spike significantly.
            """
            batch_size, num_heads, seq_len, head_dim = key_states.shape
            device = key_states.device
            dtype = key_states.dtype

            # Use smaller initial size, but at least as large as the current sequence
            cache_size = max(initial_size, seq_len)

            # Store the actual allocated size for growth tracking
            layer_cache._allocated_size = cache_size

            # Allocate smaller initial buffers
            layer_cache.keys = torch.zeros(
                (batch_size, num_heads, cache_size, head_dim),
                device=device,
                dtype=dtype
            )
            layer_cache.values = torch.zeros(
                (batch_size, num_heads, cache_size, head_dim),
                device=device,
                dtype=dtype
            )

        def grow_cache(layer_cache, required_size):
            """
            Grow the cache to accommodate more tokens.
            Uses growth factor to avoid frequent reallocations.
            """
            current_size = layer_cache._allocated_size
            if required_size <= current_size:
                return  # No growth needed

            # Calculate new size with growth factor
            new_size = max(
                int(current_size * CACHE_GROWTH_FACTOR),
                current_size + MIN_GROWTH_SIZE,
                required_size
            )

            old_keys = layer_cache.keys
            old_values = layer_cache.values
            batch_size, num_heads, _, head_dim = old_keys.shape

            # Allocate new larger buffers
            layer_cache.keys = torch.zeros(
                (batch_size, num_heads, new_size, head_dim),
                device=old_keys.device,
                dtype=old_keys.dtype
            )
            layer_cache.values = torch.zeros(
                (batch_size, num_heads, new_size, head_dim),
                device=old_values.device,
                dtype=old_values.dtype
            )

            # Copy existing data
            layer_cache.keys[:, :, :current_size, :] = old_keys
            layer_cache.values[:, :, :current_size, :] = old_values

            # Update allocated size
            layer_cache._allocated_size = new_size

            # Free old buffers
            del old_keys, old_values

        def patched_cache_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            """Patched update with chunked initialization and dtype casting."""
            cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None

            # Chunked lazy initialization - start with smaller allocation
            if self.layers[layer_idx].keys is None:
                chunked_lazy_initialization(self.layers[layer_idx], key_states)

            # Check if we need to grow the cache
            if cache_position is not None:
                if cache_position.dim() == 1:
                    max_pos = cache_position[-1].item() + 1
                else:
                    max_pos = cache_position.max().item() + 1

                current_allocated = getattr(self.layers[layer_idx], '_allocated_size',
                                           self.layers[layer_idx].keys.size(2))

                if max_pos > current_allocated:
                    grow_cache(self.layers[layer_idx], max_pos)

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
        print(f"[hunyuan_image] Applied chunked KV cache patch (initial={INITIAL_CACHE_SIZE}, growth={CACHE_GROWTH_FACTOR}x)")

    print("[hunyuan_image] Applied quantization dtype compatibility patches")
    return model


def apply_global_scatter_dtype_fix():
    """
    Patch torch.Tensor.scatter_ and scatter to automatically cast src tensor to destination dtype.
    This fixes dtype mismatches when using BitsAndBytes quantization or FP8 with image generation.
    The existing apply_quantization_dtype_fix only patches 5 methods, but there are 20+
    scatter operations in the image generation pipeline that also need dtype casting.
    """
    patched_count = 0

    # Patch in-place scatter_
    if not hasattr(torch.Tensor, '_original_scatter_'):
        original_scatter_ = torch.Tensor.scatter_

        def patched_scatter_(self, dim, index, src, **kwargs):
            if isinstance(src, torch.Tensor) and src.dtype != self.dtype:
                src = src.to(self.dtype)
            return original_scatter_(self, dim, index, src, **kwargs)

        torch.Tensor._original_scatter_ = original_scatter_
        torch.Tensor.scatter_ = patched_scatter_
        patched_count += 1

    # Also patch non-in-place scatter (for completeness)
    if hasattr(torch.Tensor, 'scatter') and not hasattr(torch.Tensor, '_original_scatter'):
        original_scatter = torch.Tensor.scatter

        def patched_scatter(self, dim, index, src, **kwargs):
            if isinstance(src, torch.Tensor) and src.dtype != self.dtype:
                src = src.to(self.dtype)
            return original_scatter(self, dim, index, src, **kwargs)

        torch.Tensor._original_scatter = original_scatter
        torch.Tensor.scatter = patched_scatter
        patched_count += 1

    if patched_count > 0:
        print(f"[hunyuan_image] Applied global scatter dtype fix ({patched_count} methods patched)")


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


def quote(text: Any) -> Optional[str]:
    """Quote text if it contains special characters (for infotext format)."""
    if text is None:
        return None
    text = str(text)
    if ',' not in text and '\n' not in text and ':' not in text:
        return text
    return json.dumps(text, ensure_ascii=False)


def create_hunyuan_infotext(
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    image_size: str,
    system_prompt_type: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
    bot_task: Optional[str] = None,
    use_taylor_cache: bool = False,
    model_name: Optional[str] = None,
    cot_summary: Optional[str] = None,
    generation_time: Optional[float] = None,
) -> str:
    """
    Create infotext string for HunyuanImage-3.0 generated images.
    Format is compatible with webui.py PNG Info tab.

    Format:
        {prompt}
        Steps: 50, CFG scale: 2.5, Seed: 123456, Size: 1024x1024, Flow shift: 3.0, ...
    """
    params = {
        "Steps": steps,
        "CFG scale": guidance_scale,
        "Seed": seed,
        "Size": image_size,
        "Flow shift": flow_shift,
        "Sampler": "HunyuanImage Flow",
    }

    if model_name:
        params["Model"] = model_name

    if system_prompt_type and system_prompt_type != "None":
        params["System prompt"] = system_prompt_type

    if custom_system_prompt and system_prompt_type == "custom":
        # Truncate long custom prompts
        truncated = custom_system_prompt[:200]
        if len(custom_system_prompt) > 200:
            truncated += "..."
        params["Custom system prompt"] = quote(truncated)

    if bot_task:
        params["Bot task"] = bot_task

    if use_taylor_cache:
        params["Taylor cache"] = "True"

    if generation_time is not None:
        params["Time"] = f"{generation_time:.1f}s"

    if cot_summary:
        # Truncate CoT to first 500 chars for readability
        truncated = str(cot_summary)[:500].replace('\n', ' ').strip()
        if len(str(cot_summary)) > 500:
            truncated += "..."
        params["CoT summary"] = quote(truncated)

    # Build the parameters string
    params_text = ", ".join(
        f"{k}: {v}" for k, v in params.items() if v is not None
    )

    return f"{prompt}\n{params_text}"


def embed_png_metadata(
    image: Image.Image,
    infotext: str,
    cot_full: Optional[str] = None,
) -> Image.Image:
    """
    Embed metadata into a PIL Image for PNG saving.

    Args:
        image: PIL Image to embed metadata into
        infotext: The main parameters string (stored in 'parameters' key)
        cot_full: Optional full CoT text (stored in 'hunyuan_cot' key)

    Returns:
        PIL Image with metadata attached (via image.info dict)
    """
    # Create a copy to avoid modifying the original
    img_with_meta = image.copy()

    # Store in image.info for PNG saving
    img_with_meta.info['parameters'] = infotext

    if cot_full:
        img_with_meta.info['hunyuan_cot'] = str(cot_full)

    return img_with_meta


def save_hunyuan_image(
    image: Image.Image,
    infotext: str,
    seed: int,
    output_dir: str = "outputs/hunyuan",
) -> str:
    """
    Save image as PNG with embedded metadata to disk.

    Args:
        image: PIL Image to save
        infotext: Generation parameters string (stored in 'parameters' PNG chunk)
        seed: Seed used for generation (used in filename)
        output_dir: Directory to save images to

    Returns:
        Absolute path to the saved PNG file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"hunyuan_{timestamp}_{seed}.png"
    filepath = os.path.join(output_dir, filename)

    # Create PNG metadata
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", infotext)

    # Save to disk as PNG with metadata
    image.save(filepath, format="PNG", pnginfo=pnginfo)
    print(f"[hunyuan_image] Saved image with metadata: {filepath}")

    return os.path.abspath(filepath)


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
        keep_attention_on_gpu: bool = False,
        progress=gr.Progress(),
    ) -> str:
        """
        Load a HunyuanImage-3.0 model with multi-GPU support.

        Args:
            model_name: Name of the model to load
            dtype: Data type (bfloat16, float16, q4_nf4, q8_fp16, fp8, sdnq_int8, sdnq_fp8, sdnq_int4, sdnq_uint4)
            num_gpus: Number of GPUs to use
            max_memory_per_gpu: Max memory per GPU in GB (None = auto)
            cpu_offload: Enable CPU offloading for large models
            cpu_offload_ram: Max CPU RAM to use for offloading in GB
            flash_attention: Use Flash Attention 2 for faster inference
            keep_attention_on_gpu: Keep attention layers on GPU when offloading (faster for FP8)
            progress: Gradio progress callback

        Returns:
            Status message
        """
        self.use_flash_attention = flash_attention
        self.keep_attention_on_gpu = keep_attention_on_gpu

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
            use_sdnq = dtype in SDNQ_DTYPE_MAP

            # Check Quanto availability early
            if use_quanto and not QUANTO_AVAILABLE:
                return "Error: optimum-quanto not installed. Run: pip install optimum-quanto"

            # Check SDNQ availability early
            if use_sdnq and not SDNQ_AVAILABLE:
                return "Error: sdnq not installed. Run: pip install sdnq"

            # Check if this is a pre-converted FP8 model
            is_preconverted_fp8 = use_fp8 and is_fp8_converted_model(model_path)

            if is_preconverted_fp8:
                print("[hunyuan_image] Detected pre-converted FP8 model")
                # Don't set torch_dtype - it would cast FP8 weights to bf16
                # We'll manually handle dtype consistency after loading
                torch_dtype = None
            elif use_sdnq:
                # SDNQ handles dtype internally, use bfloat16 for compute
                torch_dtype = torch.bfloat16
            else:
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "fp8": torch.bfloat16,  # Runtime conversion: load bf16 first
                    "q8_partial": torch.bfloat16,
                    "q8_fp16": torch.float16,
                    "q4_nf4": torch.bfloat16,
                    "quanto_int4": torch.bfloat16,
                }
                torch_dtype = dtype_map.get(dtype, torch.bfloat16)

            # Create device map for multi-GPU or CPU offloading
            actual_gpus = min(num_gpus, self.num_gpus) if self.num_gpus > 0 else 0
            max_memory = None

            use_auto_device_map = (actual_gpus > 1) or cpu_offload or (max_memory_per_gpu is not None and max_memory_per_gpu > 0)

            # Use attention-priority device map if requested
            if keep_attention_on_gpu and cpu_offload and self.num_gpus > 0:
                device_map, estimated_gpu_gb = create_attention_priority_device_map(num_layers=32, num_experts=64)

                # Set max_memory based on estimated needs + headroom
                max_memory = {}
                required_gpu_gb = int(estimated_gpu_gb * 1.2) + 2  # 20% headroom + 2GB buffer

                if max_memory_per_gpu is not None and max_memory_per_gpu > 0:
                    # User specified, use it but warn if too low
                    gpu_limit = max_memory_per_gpu
                    if gpu_limit < required_gpu_gb:
                        print(f"[hunyuan_image] Warning: {gpu_limit}GB may be too low, need ~{required_gpu_gb}GB for attention layers")
                else:
                    # Auto-detect available memory
                    if torch.cuda.is_available():
                        free_mem, total_mem = torch.cuda.mem_get_info(0)
                        available_gb = free_mem / (1024**3)
                        gpu_limit = max(required_gpu_gb, int(available_gb * 0.9))
                    else:
                        gpu_limit = required_gpu_gb

                for i in range(actual_gpus):
                    max_memory[i] = f"{gpu_limit}GiB"

                if cpu_offload_ram and cpu_offload_ram > 0:
                    max_memory["cpu"] = f"{cpu_offload_ram}GiB"
                else:
                    max_memory["cpu"] = "128GiB"

                print(f"[hunyuan_image] Attention-priority: GPU={gpu_limit}GB, CPU={max_memory['cpu']}")
            elif use_auto_device_map and self.num_gpus > 0:
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

            if use_sdnq:
                # Use SDNQ quantization
                sdnq_weights_dtype = SDNQ_DTYPE_MAP[dtype]
                print(f"[hunyuan_image] Using SDNQ quantization: {sdnq_weights_dtype}")
                # Don't override return_device - let device_map control placement
                # SDNQ will quantize on whatever device the weight is loaded to
                quantization_config = create_sdnq_config(
                    weights_dtype=sdnq_weights_dtype,
                    use_quantized_matmul=True,
                    quantization_device=None,
                    return_device=None,  # Let device_map handle placement
                )
                model_kwargs["quantization_config"] = quantization_config
                # SDNQ handles dtype internally, don't set torch_dtype
            elif use_q4_nf4:
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
                # Only set torch_dtype if specified (None for pre-converted FP8)
                if torch_dtype is not None:
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

                # Load generation_config (not loaded by from_config)
                from transformers import GenerationConfig
                gen_config_path = os.path.join(model_path, "generation_config.json")
                if os.path.exists(gen_config_path):
                    self.model.generation_config = GenerationConfig.from_pretrained(model_path)
                    print(f"[hunyuan_image] Loaded generation_config")

                progress(0.5, desc="Loading quantized weights...")
                log_gpu_memory("After model structure, before quanto weights")

                # Load quantization map
                with open(quanto_map_path, 'r') as f:
                    qmap = json.load(f)

                # Load quantized state dict
                print(f"[hunyuan_image] Loading quantized weights from {quanto_weights_path}...")
                state_dict = safetensors_load(quanto_weights_path, device="cpu")

                progress(0.7, desc="Applying quantized weights...")

                # Apply quantized weights to CPU first (don't pass device to avoid OOM)
                requantize(self.model, state_dict, qmap)

                # Free state_dict memory
                del state_dict
                gc.collect()

                progress(0.8, desc="Dispatching model to devices...")

                # Use accelerate to dispatch model with CPU offloading
                from accelerate import infer_auto_device_map, dispatch_model

                # Build max_memory dict for offloading
                dispatch_max_memory = {}
                if self.num_gpus > 0:
                    # Leave some room for activations
                    dispatch_max_memory[0] = f"{int(torch.cuda.get_device_properties(0).total_memory * 0.9 / 1024**3)}GiB"
                if cpu_offload and cpu_offload_ram:
                    dispatch_max_memory["cpu"] = f"{cpu_offload_ram}GiB"
                else:
                    dispatch_max_memory["cpu"] = "64GiB"

                print(f"[hunyuan_image] Dispatching with max_memory: {dispatch_max_memory}")

                device_map = infer_auto_device_map(self.model, max_memory=dispatch_max_memory)
                self.model = dispatch_model(self.model, device_map=device_map)

                log_gpu_memory("After Quanto loading and dispatch")
                print(f"[hunyuan_image] Loaded pre-quantized model successfully")

            elif is_preconverted_fp8:
                # Use custom FP8 loader that preserves FP8 dtypes during loading
                print("[hunyuan_image] Using custom FP8 loader to preserve dtypes...")

                # Prepare offload folder if needed
                offload_folder = None
                if cpu_offload:
                    offload_folder = os.path.join(os.path.dirname(model_path), ".offload_cache")
                    os.makedirs(offload_folder, exist_ok=True)

                self.model = load_fp8_model_with_offload(
                    model_path=model_path,
                    device_map=device_map if device_map else "auto",
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    attn_implementation=model_kwargs.get("attn_implementation", "sdpa"),
                    moe_impl=model_kwargs.get("moe_impl", "auto"),
                    moe_drop_tokens=model_kwargs.get("moe_drop_tokens", False),
                )

            elif use_sdnq and cpu_offload:
                # SDNQ with CPU offload: quantize on GPU (fast), store on CPU (avoids OOM)
                print("[hunyuan_image] SDNQ + CPU offload: GPU quantization with CPU storage...")

                # Use GPU for fast quantization, but store weights on CPU
                sdnq_weights_dtype = SDNQ_DTYPE_MAP[dtype]
                cpu_quantization_config = create_sdnq_config(
                    weights_dtype=sdnq_weights_dtype,
                    use_quantized_matmul=True,
                    quantization_device="cuda:0" if self.num_gpus > 0 else None,  # Quantize on GPU (fast)
                    return_device="cpu",  # Store quantized weights on CPU (avoids OOM)
                )

                sdnq_kwargs = model_kwargs.copy()
                sdnq_kwargs["quantization_config"] = cpu_quantization_config
                sdnq_kwargs["device_map"] = "cpu"  # Load structure to CPU
                sdnq_kwargs.pop("max_memory", None)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **sdnq_kwargs,
                )

                log_gpu_memory("After SDNQ load to CPU")

                # Free GPU memory used during quantization before dispatching
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_gpu_memory("After cleanup, before dispatch")

                progress(0.7, desc="Dispatching to GPU/CPU...")

                # Now dispatch: attention to GPU, MoE experts stay on CPU
                from accelerate import dispatch_model
                explicit_device_map, gpu_gb = create_attention_priority_device_map(
                    num_layers=32, num_experts=64
                )
                print(f"[hunyuan_image] Dispatching: ~{gpu_gb:.1f}GB to GPU, MoE experts on CPU")

                self.model = dispatch_model(self.model, device_map=explicit_device_map)
                device_map = explicit_device_map

                log_gpu_memory("After dispatch to GPU/CPU")

            else:
                # Normal loading for non-quanto, non-FP8 models
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

            # Handle FP8: runtime conversion for non-preconverted models
            # (Pre-converted FP8 models are handled by load_fp8_model_with_offload above)
            if use_fp8 and not use_quanto and not is_preconverted_fp8:
                log_gpu_memory("Before FP8 conversion")
                print("[hunyuan_image] Converting model to FP8 at runtime...")
                print("[hunyuan_image] Note: For better results, use convert_to_fp8.py to pre-convert the model")
                self.model = convert_model_to_fp8_scaled(self.model, skip_patterns=HUNYUAN_SKIP_MODULES)
                log_gpu_memory("After FP8 conversion")

            # Apply quantization/FP8 compatibility patches for dtype mismatches
            # FP8 also has mixed dtypes: FP8 layers output bf16, but VAE/embeddings may be float32
            if use_q4_nf4 or use_q8 or use_quanto or use_fp8 or use_sdnq:
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

            # Clear GPU memory before generation to minimize spike
            clear_gpu_memory(sync=True)
            log_gpu_memory("Before generation (after cleanup)")

            # Generate image using the model's generate_image API
            cot_text, outputs = self.model.generate_image(**gen_kwargs)

            # Log memory after generation
            log_gpu_memory("After generation")

            # Clear GPU memory after generation to release intermediate tensors
            clear_gpu_memory(sync=True)

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

            # Embed PNG metadata for webui compatibility
            model_name = Path(self.current_model_path).stem if self.current_model_path else None
            infotext = create_hunyuan_infotext(
                prompt=prompt,
                seed=actual_seed,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
                flow_shift=flow_shift,
                image_size=f"{width}x{height}",
                system_prompt_type=system_prompt_type,
                custom_system_prompt=custom_system_prompt,
                bot_task=bot_task,
                use_taylor_cache=use_taylor_cache,
                model_name=model_name,
                cot_summary=cot_reasoning if cot_reasoning else None,
                generation_time=generation_time,
            )
            # Save to disk with PNG metadata
            filepath = save_hunyuan_image(generated_image, infotext, actual_seed)

            progress(1.0, desc="Done!")
            print(f"[hunyuan_image] Generation completed in {generation_time:.2f}s")

            return filepath, stats, cot_reasoning

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

        # Clear GPU memory before generation to minimize spike
        clear_gpu_memory(sync=True)
        log_gpu_memory("Before streaming generation (after cleanup)")

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
                # Clear intermediate tensors after generation
                clear_gpu_memory(sync=False)
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
        # Final cleanup
        clear_gpu_memory(sync=True)

        # Print FP8 stats after generation
        import torch.nn as nn
        if hasattr(nn.Linear, '_print_fp8_stats'):
            nn.Linear._print_fp8_stats()
        from accelerate.utils import offload as accel_offload
        if hasattr(accel_offload, 'print_fp8_offload_stats'):
            accel_offload.print_fp8_offload_stats()

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

            # Embed PNG metadata for webui compatibility
            model_name = Path(self.current_model_path).stem if self.current_model_path else None
            infotext = create_hunyuan_infotext(
                prompt=prompt,
                seed=actual_seed,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
                flow_shift=flow_shift,
                image_size=f"{width}x{height}",
                system_prompt_type=system_prompt_type,
                custom_system_prompt=custom_system_prompt,
                bot_task=bot_task,
                use_taylor_cache=use_taylor_cache,
                model_name=model_name,
                cot_summary=cot_text if cot_text else None,
                generation_time=total_time,
            )
            # Save to disk with PNG metadata
            filepath = save_hunyuan_image(image, infotext, actual_seed)

            yield cot_text, final_stats, filepath

    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL image to temporary file and return path, preserving metadata."""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vlm_img_{int(time.time())}_{id(image)}.png")

        # Preserve PNG metadata if present
        pnginfo = None
        if image.info:
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(value, str):
                    pnginfo.add_text(key, value)

        image.save(temp_path, pnginfo=pnginfo)
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
    keep_attention_on_gpu: bool = False,
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
        keep_attention_on_gpu=keep_attention_on_gpu,
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
                            type="filepath",
                            format="png",
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
                        choices=[
                            "bfloat16", "float16",
                            "fp8", "q4_nf4", "q8_fp16", "quanto_int4",
                            "sdnq_int8", "sdnq_fp8", "sdnq_int4", "sdnq_uint4",
                        ],
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
                    keep_attn_gpu_check = gr.Checkbox(
                        value=saved_settings.get("keep_attention_on_gpu", False),
                        label="Keep Attention on GPU",
                        info="Faster FP8 inference with CPU offload",
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
                keep_attn_gpu_check,
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
                keep_attn_gpu_check,
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
