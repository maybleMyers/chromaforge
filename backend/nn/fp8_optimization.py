"""
FP8 optimization for ChromaDCT/Radiance models.
Supports both pre-scaled FP8 weights and on-the-fly conversion.
"""

import torch
import torch.nn as nn
from backend import memory_management


def fp8_linear_forward(cls, original_dtype, input):
    """FP8 linear forward using torch._scaled_mm for efficient computation."""
    weight = cls.weight
    weight_dtype = weight.dtype

    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        scale_weight = getattr(cls, 'scale_weight', None)

        if len(input.shape) == 3:
            target_dtype = torch.float8_e5m2 if weight_dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
            inn = input.reshape(-1, input.shape[2]).to(target_dtype)
            w = weight.t()

            if scale_weight is not None:
                scale_a = torch.ones((1), device=input.device, dtype=torch.float32)
                scale_b = scale_weight.to(torch.float32)
            else:
                scale_a = torch.ones((1), device=input.device, dtype=torch.float32)
                scale_b = scale_a

            bias = cls.bias.to(original_dtype) if cls.bias is not None else None

            if bias is not None:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, bias=bias, scale_a=scale_a, scale_b=scale_b)
            else:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, scale_a=scale_a, scale_b=scale_b)

            if isinstance(o, tuple):
                o = o[0]

            return o.reshape((-1, input.shape[1], weight.shape[0]))
        elif len(input.shape) == 2:
            target_dtype = torch.float8_e5m2 if weight_dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
            inn = input.to(target_dtype)
            w = weight.t()

            if scale_weight is not None:
                scale_a = torch.ones((1), device=input.device, dtype=torch.float32)
                scale_b = scale_weight.to(torch.float32)
            else:
                scale_a = torch.ones((1), device=input.device, dtype=torch.float32)
                scale_b = scale_a

            bias = cls.bias.to(original_dtype) if cls.bias is not None else None

            if bias is not None:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, bias=bias, scale_a=scale_a, scale_b=scale_b)
            else:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, scale_a=scale_a, scale_b=scale_b)

            if isinstance(o, tuple):
                o = o[0]

            return o
        else:
            return cls.original_forward(input.to(original_dtype))
    else:
        return cls.original_forward(input)


def convert_to_fp8_scaled(module, original_dtype=None, exclude_patterns=None):
    """
    Convert model linear layers to FP8 with per-tensor scaling.

    Args:
        module: The model to convert
        original_dtype: The original dtype for computation (default: bf16)
        exclude_patterns: List of name patterns to exclude from conversion
    """
    if original_dtype is None:
        original_dtype = torch.bfloat16

    if exclude_patterns is None:
        exclude_patterns = []

    converted_count = 0

    for name, child in module.named_modules():
        if any(pattern in name for pattern in exclude_patterns):
            continue

        if isinstance(child, nn.Linear) and not hasattr(child, 'fp8_converted'):
            weight = child.weight.data

            if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                continue

            abs_max = weight.abs().max()
            scale = abs_max / 448.0  # FP8 E4M3 max value

            fp8_weight = (weight / scale).to(torch.float8_e4m3fn)

            child.weight = nn.Parameter(fp8_weight, requires_grad=False)
            child.scale_weight = nn.Parameter(scale.unsqueeze(0), requires_grad=False)

            original_forward = child.forward
            child.original_forward = original_forward
            child.forward = lambda input, m=child, od=original_dtype: fp8_linear_forward(m, od, input)
            child.fp8_converted = True

            converted_count += 1

    if converted_count > 0:
        print(f"[FP8] Converted {converted_count} linear layers to FP8 scaled format")

    return module


def detect_fp8_scaled_weights(state_dict):
    """
    Detect if state dict contains pre-scaled FP8 weights.

    Returns:
        tuple: (is_fp8, has_scale_weights)
    """
    has_fp8_weights = False
    has_scale_weights = False

    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                has_fp8_weights = True
            if 'scale_weight' in key:
                has_scale_weights = True

    return has_fp8_weights, has_scale_weights


def enable_fp8_for_chromadct(model, computation_dtype=None):
    """
    Enable FP8 optimizations for ChromaDCT/Radiance model.

    If weights are already FP8, just enables the optimized forward.
    If weights are FP16/BF16, converts them to FP8 with scaling.

    Args:
        model: ChromaDCT model
        computation_dtype: Output dtype for matmul (default: model's computation_dtype)
    """
    if computation_dtype is None:
        computation_dtype = getattr(model, 'computation_dtype', torch.bfloat16)

    exclude_nerf = ['nerf_blocks', 'nerf_image_embedder', 'nerf_final_layer']

    for name, child in model.named_modules():
        if any(pattern in name for pattern in exclude_nerf):
            continue

        if isinstance(child, nn.Linear):
            weight = child.weight
            if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                if not hasattr(child, 'original_forward'):
                    original_forward = child.forward
                    child.original_forward = original_forward
                    child.forward = lambda input, m=child, cd=computation_dtype: fp8_linear_forward(m, cd, input)
                    child.fp8_enabled = True

    setattr(model, 'fp8_enabled', True)
    print(f"[FP8] Enabled FP8 optimizations for ChromaDCT model")
    return model


def convert_chromadct_to_fp8(model, computation_dtype=None):
    """
    Convert ChromaDCT model weights to FP8 format with scaling.

    Args:
        model: ChromaDCT model
        computation_dtype: Output dtype (default: bf16)
    """
    if computation_dtype is None:
        computation_dtype = getattr(model, 'computation_dtype', torch.bfloat16)

    exclude_nerf = ['nerf_blocks', 'nerf_image_embedder', 'nerf_final_layer']

    model = convert_to_fp8_scaled(model, computation_dtype, exclude_nerf)
    setattr(model, 'fp8_enabled', True)

    return model


# ============================================================================
# VLM FP8 Optimization (adapted from musubi-tuner)
# ============================================================================

import gc
import json
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default exclude patterns for VLM models
VLM_FP8_EXCLUDE_KEYS = [
    "visual",           # Vision encoder - very sensitive
    "embed",            # Embeddings
    "lm_head",          # Output head
    "norm",             # Normalization layers
    "rotary",           # Rotary embeddings
    "merger",           # Vision-language merger
    "deepstack",        # Deepstack merger
]


def quantize_fp8_vlm(tensor, scale, fp8_dtype, max_value, min_value):
    """Quantize a tensor to FP8 format."""
    tensor = tensor.to(torch.float32)
    tensor = torch.div(tensor, scale).nan_to_num_(0.0)
    tensor = tensor.clamp_(min=min_value, max=max_value)
    tensor = tensor.to(fp8_dtype)
    return tensor


def fp8_linear_forward_vlm(self, x):
    """
    Patched forward method for VLM Linear layers with FP8 weights.
    Uses torch._scaled_mm for native FP8 tensor core computation.

    BF16 layers (visual, embed, norm) are NOT patched and use regular forward.
    Only quantized layers get this fast FP8 path.
    """
    weight = self.weight
    weight_dtype = weight.dtype

    # Get computation dtype from scale_weight (this is the original model dtype, usually bf16)
    computation_dtype = self.scale_weight.dtype

    # Only use FP8 fast path if weight is actually FP8
    if weight_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        # Fallback to regular forward for non-FP8 weights
        return nn.functional.linear(x, weight, self.bias)

    # FP8 matmul requires complementary dtypes: e4m3fn weights with e5m2 inputs (or vice versa)
    input_fp8_dtype = torch.float8_e5m2 if weight_dtype == torch.float8_e4m3fn else torch.float8_e4m3fn

    # Get scales - ensure they're on the same device as input
    scale_weight = self.scale_weight.to(device=x.device, dtype=torch.float32)
    scale_input = torch.ones(1, device=x.device, dtype=torch.float32)

    # Cache transposed weight for efficiency (avoid transpose every forward)
    if not hasattr(self, '_weight_t_cached') or self._weight_t_cached is None:
        self._weight_t_cached = weight.t().contiguous()
    w_t = self._weight_t_cached

    # Handle 3D input (batch, seq, hidden) - common for transformers
    if x.dim() == 3:
        batch, seq_len, hidden = x.shape
        # Reshape to 2D for matmul
        x_2d = x.reshape(-1, hidden)

        # Convert input to FP8
        x_fp8 = x_2d.to(input_fp8_dtype)

        # Native FP8 matmul using tensor cores
        output = torch._scaled_mm(
            x_fp8, w_t,
            out_dtype=computation_dtype,
            scale_a=scale_input,
            scale_b=scale_weight,
        )

        # Handle tuple return (some PyTorch versions)
        if isinstance(output, tuple):
            output = output[0]

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.to(computation_dtype)

        # Reshape back to 3D
        return output.reshape(batch, seq_len, -1)

    # Handle 2D input (batch, hidden)
    elif x.dim() == 2:
        # Convert input to FP8
        x_fp8 = x.to(input_fp8_dtype)

        # Native FP8 matmul
        output = torch._scaled_mm(
            x_fp8, w_t,
            out_dtype=computation_dtype,
            scale_a=scale_input,
            scale_b=scale_weight,
        )

        if isinstance(output, tuple):
            output = output[0]

        if self.bias is not None:
            output = output + self.bias.to(computation_dtype)

        return output

    else:
        # Fallback for other shapes - use dequantization path
        dequantized_weight = weight.to(computation_dtype) * scale_weight
        return nn.functional.linear(x.to(computation_dtype), dequantized_weight, self.bias)


def optimize_vlm_state_dict_fp8(
    state_dict: dict,
    calc_device,
    target_layer_keys=None,
    exclude_layer_keys=None,
    move_to_device: bool = False,
) -> dict:
    """
    Optimize VLM state dict with FP8 quantization (in-place).

    Args:
        state_dict: Model state dict
        calc_device: Device for quantization
        target_layer_keys: Layer patterns to target (None = all)
        exclude_layer_keys: Layer patterns to exclude
        move_to_device: Keep weights on calc_device

    Returns:
        Optimized state dict
    """
    fp8_dtype = torch.float8_e4m3fn
    max_value = torch.finfo(fp8_dtype).max
    min_value = -max_value

    if exclude_layer_keys is None:
        exclude_layer_keys = VLM_FP8_EXCLUDE_KEYS

    optimized_count = 0
    skipped_count = 0

    # Find target keys
    target_keys = []
    for key in state_dict.keys():
        is_target = (target_layer_keys is None or any(p in key for p in target_layer_keys)) and key.endswith(".weight")
        is_excluded = any(p in key for p in exclude_layer_keys)
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            tensor = state_dict[key]
            # Only quantize 2D weights (Linear layers) with reasonable size
            if tensor.ndim == 2 and tensor.numel() > 1000:
                target_keys.append(key)
            else:
                skipped_count += 1

    print(f"[FP8-VLM] Found {len(target_keys)} layers to quantize, {skipped_count} skipped")

    # Process each target
    for key in tqdm(target_keys, desc="[FP8-VLM] Quantizing"):
        value = state_dict[key]

        original_device = value.device
        original_dtype = value.dtype

        if calc_device is not None:
            value = value.to(calc_device)

        # Calculate scale (per-tensor)
        abs_max = value.abs().max()
        scale = (abs_max / max_value).clamp(min=1e-8).to(torch.float32)

        # Quantize
        fp8_weight = quantize_fp8_vlm(value, scale, fp8_dtype, max_value, min_value)

        # Store
        scale_key = key.replace(".weight", ".scale_weight")

        if not move_to_device:
            fp8_weight = fp8_weight.to(original_device)

        scale = scale.unsqueeze(0).to(dtype=original_dtype, device=fp8_weight.device)

        state_dict[key] = fp8_weight
        state_dict[scale_key] = scale

        optimized_count += 1

        if calc_device is not None and optimized_count % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"[FP8-VLM] Optimized {optimized_count} layers to FP8")
    return state_dict


def apply_fp8_vlm_monkey_patch(model, optimized_state_dict):
    """
    Apply monkey patches for FP8 VLM model.

    Args:
        model: The model to patch
        optimized_state_dict: State dict with FP8 weights and scales

    Returns:
        Patched model
    """
    # Find scale keys
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    patched_paths = set()
    scale_shapes = {}
    for scale_key in scale_keys:
        path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_paths.add(path)
        scale_shapes[path] = optimized_state_dict[scale_key].shape

    patched_count = 0

    for name, module in model.named_modules():
        if name in patched_paths and isinstance(module, nn.Linear):
            scale_shape = scale_shapes[name]
            module.register_buffer("scale_weight", torch.ones(scale_shape, dtype=module.weight.dtype))

            def new_forward(self, x):
                return fp8_linear_forward_vlm(self, x)

            module.forward = new_forward.__get__(module, type(module))
            patched_count += 1

    print(f"[FP8-VLM] Patched {patched_count} Linear layers for FP8")
    return model


def load_vlm_with_fp8(
    model,
    model_path: str,
    exclude_keys=None,
    device="cuda",
):
    """
    Load a VLM model with FP8 optimization from safetensors.

    Args:
        model: Initialized model (can be on meta device)
        model_path: Path to model directory
        exclude_keys: Layer patterns to exclude from FP8
        device: Device for quantization

    Returns:
        Model with FP8 weights
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")

    model_dir = Path(model_path)

    # Find safetensor files
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        safetensor_files = [str(model_dir / f) for f in shard_files]
    else:
        single_file = model_dir / "model.safetensors"
        if single_file.exists():
            safetensor_files = [str(single_file)]
        else:
            safetensor_files = sorted([str(f) for f in model_dir.glob("*.safetensors")])

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors in {model_path}")

    print(f"[FP8-VLM] Loading from {len(safetensor_files)} shard(s)")

    # Load all weights to CPU
    state_dict = {}
    for shard in tqdm(safetensor_files, desc="[FP8-VLM] Loading"):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    print(f"[FP8-VLM] Loaded {len(state_dict)} tensors")

    # FP8 optimization
    state_dict = optimize_vlm_state_dict_fp8(
        state_dict,
        calc_device=device,
        exclude_layer_keys=exclude_keys or VLM_FP8_EXCLUDE_KEYS,
        move_to_device=True,
    )

    # Apply patches
    apply_fp8_vlm_monkey_patch(model, state_dict)

    # Load state dict
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[FP8-VLM] Loaded: {len(info.missing_keys)} missing, {len(info.unexpected_keys)} unexpected")

    return model
