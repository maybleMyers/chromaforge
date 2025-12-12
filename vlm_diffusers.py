"""
VLM Backend using Diffusers/Transformers for Vision Language Models
Supports Qwen-VL (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) and GLM-4V models.
Supports loading unquantized models across multiple GPUs.

Requirements:
- pip install transformers>=4.51.0 accelerate qwen-vl-utils torch torchvision
- For multi-GPU: pip install accelerate

Usage:
    python vlm_diffusers.py --models-dir models/LLM --port 7863
"""

import os
import gc
import time
import json
import argparse
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from io import BytesIO
import base64

import torch
import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
from PIL import Image

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
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )
    TRANSFORMERS_AVAILABLE = True
    print(f"Transformers version: {transformers.__version__}")

    # Try to import Qwen3 VL MoE class (requires newer transformers)
    try:
        from transformers import Qwen3VLMoeForConditionalGeneration
        QWEN3_VL_AVAILABLE = True
    except ImportError:
        QWEN3_VL_AVAILABLE = False
        print("Note: Qwen3VLMoeForConditionalGeneration not available, will use fallback classes")

    # Use native transformers GLM4V classes (GLM-4.6V-Flash is compatible with GLM-4.1V architecture)
    try:
        from transformers import (
            Glm4vForConditionalGeneration,
            Glm4vImageProcessor,
            Glm4vVideoProcessor,
            Glm4vProcessor,
        )
        GLM4V_AVAILABLE = True
        print("GLM-4V/4.6V support: available (native transformers)")
    except ImportError as glm_err:
        GLM4V_AVAILABLE = False
        Glm4vForConditionalGeneration = None
        Glm4vImageProcessor = None
        Glm4vVideoProcessor = None
        Glm4vProcessor = None
        print(f"Note: GLM-4V not available: {glm_err}")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    QWEN3_VL_AVAILABLE = False
    GLM4V_AVAILABLE = False
    print(f"Error: transformers not installed or incompatible version: {e}")

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
    print(f"Accelerate version: {accelerate.__version__}")
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not installed. Multi-GPU support will be limited.")

# Try to import qwen_vl_utils for image/video processing
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen-vl-utils not installed. Install with: pip install qwen-vl-utils")


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


def find_vlm_models(models_dir: str) -> List[Dict[str, str]]:
    """
    Find VLM models in a directory.
    Looks for directories containing config.json (HuggingFace model format).
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
        model_type = "unknown"
        try:
            import json
            with open(config_file, "r") as f:
                config = json.load(f)
                arch = config.get("architectures", [""])[0].lower()
                model_type_str = config.get("model_type", "").lower()

                # Detect Qwen models
                if "qwen" in arch or "qwen" in model_type_str:
                    if "vl" in arch or "vision" in model_type_str or config.get("vision_config"):
                        model_type = "qwen-vl"
                    else:
                        model_type = "qwen"
                # Detect GLM models
                elif "glm" in arch or "glm" in model_type_str:
                    if config.get("vision_config") or "4v" in model_type_str or "vl" in arch:
                        model_type = "glm-vl"
                    else:
                        model_type = "glm"
        except Exception:
            pass

        # Include Qwen and GLM models
        if model_type in ["qwen", "qwen-vl", "glm", "glm-vl"]:
            models.append({
                "name": model_dir.name,
                "path": str(model_dir),
                "type": model_type,
            })

    return sorted(models, key=lambda x: x["name"])


class Qwen3VLMBackend:
    """
    Diffusers/Transformers-based VLM backend for Vision Language Models.
    Supports Qwen-VL (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) and GLM-4V models.
    Supports multi-GPU inference with automatic device mapping.
    """

    def __init__(self, models_dir: str = "models/LLM"):
        self.models_dir = models_dir
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_path: Optional[str] = None
        self.current_model_type: Optional[str] = None
        self.device_map = None

        # Check GPU availability
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Detected {self.num_gpus} GPU(s)")
        print_gpu_status()

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available VLM models."""
        return find_vlm_models(self.models_dir)

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
        num_gpus: int = 2,
        max_memory_per_gpu: Optional[int] = None,
        cpu_offload: bool = False,
        cpu_offload_ram: Optional[int] = None,
        progress=gr.Progress(),
    ) -> str:
        """
        Load a Qwen VLM model with multi-GPU support.

        Args:
            model_name: Name of the model to load
            dtype: Data type (bfloat16, float16, float32)
            num_gpus: Number of GPUs to use
            max_memory_per_gpu: Max memory per GPU in GB (None = auto)
            cpu_offload: Enable CPU offloading for large models
            cpu_offload_ram: Max CPU RAM to use for offloading in GB
            progress: Gradio progress callback

        Returns:
            Status message
        """
        if not TRANSFORMERS_AVAILABLE:
            return "Error: transformers not installed"

        # Find the model
        models = self.get_available_models()
        model_info = next((m for m in models if m["name"] == model_name), None)

        if model_info is None:
            return f"Error: Model '{model_name}' not found"

        model_path = model_info["path"]
        model_type = model_info["type"]

        # Unload current model if any
        if self.model is not None:
            self.unload_model()

        try:
            progress(0.1, desc="Configuring device map...")

            # Determine torch dtype and quantization mode
            # Quantization options:
            # - q8_partial: INT8 with bf16 non-quantized layers (SLOW - bf16->fp16 casting)
            # - q8_fp16: INT8 with fp16 non-quantized layers (FAST - native bitsandbytes)
            # - q4_nf4: 4-bit NF4 with bf16 compute (smallest memory, good quality)
            # - fp8_scaled: Manual FP8 conversion (native on RTX 40/50 series)
            use_fp8 = dtype == "fp8_scaled"
            use_q8_partial = dtype in ["q8_partial", "q8_fp16"]
            use_q4_nf4 = dtype == "q4_nf4"

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "fp8_scaled": None,  # Use native dtype, convert after
                "q8_partial": torch.bfloat16,  # Non-quantized in bf16 (slow - casting overhead)
                "q8_fp16": torch.float16,  # Non-quantized in fp16 (fast - native bnb support)
                "q4_nf4": torch.bfloat16,  # 4-bit NF4 supports bf16 compute dtype
            }
            torch_dtype = dtype_map.get(dtype, torch.bfloat16)

            # Create device map for multi-GPU or CPU offloading
            actual_gpus = min(num_gpus, self.num_gpus) if self.num_gpus > 0 else 0
            max_memory = None

            # Use device_map="auto" when:
            # 1. Multiple GPUs, OR
            # 2. CPU offloading enabled, OR
            # 3. max_memory_per_gpu is set (to respect the limit)
            use_auto_device_map = (actual_gpus > 1) or cpu_offload or (max_memory_per_gpu is not None and max_memory_per_gpu > 0)

            if use_auto_device_map and self.num_gpus > 0:
                device_map = "auto"
                max_memory = {}

                # Configure GPU memory constraints
                if max_memory_per_gpu is not None and max_memory_per_gpu > 0:
                    # User specified a limit
                    for i in range(actual_gpus):
                        max_memory[i] = f"{max_memory_per_gpu}GiB"
                elif cpu_offload:
                    # CPU offload enabled but no GPU limit set - auto-detect and use most of available VRAM
                    for i in range(actual_gpus):
                        if torch.cuda.is_available():
                            free_mem, total_mem = torch.cuda.mem_get_info(i)
                            # Use 90% of free VRAM to leave headroom
                            gpu_mem_gb = int((free_mem / (1024**3)) * 0.9)
                            max_memory[i] = f"{gpu_mem_gb}GiB"
                            print(f"GPU {i}: auto-detected {gpu_mem_gb}GB available for model")

                # Add CPU memory allowance for offloading
                if cpu_offload and cpu_offload_ram and cpu_offload_ram > 0:
                    max_memory["cpu"] = f"{cpu_offload_ram}GiB"
                    print(f"CPU offloading enabled: allowing up to {max_memory['cpu']} CPU RAM")
                elif max_memory:
                    # Even without explicit offload, allow some CPU as fallback
                    max_memory["cpu"] = "16GiB"

                if not max_memory:
                    max_memory = None

                offload_str = " (CPU offload enabled)" if cpu_offload else ""
                print(f"Using device_map='auto' for {actual_gpus} GPU(s){offload_str}")
            else:
                device_map = {"": 0} if self.num_gpus > 0 else "cpu"
                print(f"Using single device: {device_map}")

            progress(0.2, desc="Loading processor/tokenizer...")

            # Load processor or tokenizer based on model type
            if model_type in ["qwen-vl", "glm-vl"]:
                if model_type == "glm-vl" and GLM4V_AVAILABLE:
                    # Use native Glm4vProcessor with manual tokenizer construction
                    # (tokenizer_config.json has incompatible "TokenizersBackend" class)
                    from transformers import PreTrainedTokenizerFast
                    from tokenizers import Tokenizer as TokenizerFast

                    # Load tokenizer directly from tokenizer.json (bypasses tokenizer_config issues)
                    tokenizer_file = os.path.join(model_path, "tokenizer.json")
                    tokenizer_config_file = os.path.join(model_path, "tokenizer_config.json")

                    tokenizer_object = TokenizerFast.from_file(tokenizer_file)

                    # Load config for special tokens
                    with open(tokenizer_config_file, 'r') as f:
                        tokenizer_config = json.load(f)

                    # Create fast tokenizer with proper special tokens
                    tokenizer = PreTrainedTokenizerFast(
                        tokenizer_object=tokenizer_object,
                        eos_token=tokenizer_config.get("eos_token", "<|endoftext|>"),
                        pad_token=tokenizer_config.get("pad_token", "<|endoftext|>"),
                        model_max_length=tokenizer_config.get("model_max_length", 128000),
                        padding_side=tokenizer_config.get("padding_side", "left"),
                    )

                    # Add additional special tokens
                    additional_special = tokenizer_config.get("additional_special_tokens", [])
                    if additional_special:
                        tokenizer.add_special_tokens({"additional_special_tokens": additional_special})

                    # Load native image processor
                    image_processor = Glm4vImageProcessor.from_pretrained(model_path)

                    # Load native video processor (REQUIRED by Glm4vProcessor)
                    video_processor = Glm4vVideoProcessor.from_pretrained(model_path)

                    # Load chat template
                    chat_template_path = os.path.join(model_path, "chat_template.jinja")
                    chat_template = None
                    if os.path.exists(chat_template_path):
                        with open(chat_template_path, "r") as f:
                            chat_template = f.read()

                    # Use native Glm4vProcessor (requires image_processor, tokenizer, AND video_processor)
                    self.processor = Glm4vProcessor(
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                        video_processor=video_processor,
                        chat_template=chat_template,
                    )
                    print(f"Loaded GLM processor for {model_name}")
                else:
                    # Use AutoProcessor for Qwen
                    self.processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
                    print(f"Loaded processor for {model_name}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                print(f"Loaded tokenizer for {model_name}")

            # Build progress/status strings
            quant_str = ""
            if use_fp8:
                quant_str = " (FP8 scaled)"
            elif use_q4_nf4:
                quant_str = " (4-bit NF4)"
            elif dtype == "q8_fp16":
                quant_str = " (INT8 + fp16 vision)"
            elif dtype == "q8_partial":
                quant_str = " (INT8 + bf16 vision - slow)"

            progress(0.3, desc=f"Loading {model_name} across {actual_gpus} GPU(s)...{quant_str}")
            print(f"Loading model from: {model_path}")
            print(f"Dtype: {torch_dtype if torch_dtype else 'native (for FP8)'}, Device map: {device_map}")

            # Load the model
            model_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            # For Q8 partial: use BitsAndBytesConfig to quantize language model to INT8
            # but skip vision encoder layers which are sensitive to quantization
            quantization_config = None
            if use_q8_partial:
                # Based on quantization sensitivity analysis:
                # - Visual encoder layers (model.visual.*) are CRITICAL and should NOT be quantized
                # - This includes pos_embed, attention blocks, MLP layers, merger layers
                # - Language model layers can safely be quantized to INT8
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=['visual'],  # Skip entire visual encoder
                    llm_int8_threshold=6.0,  # Default threshold for outlier handling
                )
                is_fp16_mode = dtype == "q8_fp16"
                print(f"[Q8] Using partial INT8 quantization (skipping visual encoder)")
                print(f"[Q8] Non-quantized layers dtype: {'fp16 (fast)' if is_fp16_mode else 'bf16 (slow - casting)'}")
                print("[Q8] Expected memory: ~30GB (from ~62GB for bf16)")
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = device_map
                if max_memory is not None:
                    model_kwargs["max_memory"] = max_memory

            # For Q4 NF4: 4-bit quantization with bf16 compute (smallest memory footprint)
            elif use_q4_nf4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # NormalFloat4 - better for normally distributed weights
                    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 compute is supported for 4-bit!
                    bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
                    llm_int8_skip_modules=['visual'],  # Skip visual encoder (uses bnb_4bit_skip_modules internally)
                )
                print("[Q4] Using 4-bit NF4 quantization (skipping visual encoder)")
                print("[Q4] Compute dtype: bf16 (native support)")
                print("[Q4] Expected memory: ~20GB (from ~62GB for bf16)")
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = device_map
                if max_memory is not None:
                    model_kwargs["max_memory"] = max_memory

            # For FP8: Use new musubi-tuner style loading
            # 1. Create model with meta tensors
            # 2. Load weights from safetensors with FP8 optimization
            # 3. Apply monkey patches and load state dict
            elif use_fp8:
                # We'll handle FP8 separately after model creation
                model_kwargs["device_map"] = "meta"  # Create with meta tensors
                print("[FP8] Creating model structure with meta tensors...")
                print("[FP8] Will load and quantize weights from safetensors...")
            else:
                model_kwargs["device_map"] = device_map
                if max_memory is not None:
                    model_kwargs["max_memory"] = max_memory

            # Only set dtype if not using FP8 (FP8 loads native dtype first)
            # For Q8 partial, torch_dtype is used for non-quantized layers
            if torch_dtype is not None:
                model_kwargs["dtype"] = torch_dtype

            # Use appropriate model class based on type
            if model_type in ["qwen-vl", "glm-vl"]:
                # Detect model architecture from config to choose the right class
                config_path = os.path.join(model_path, "config.json")
                model_type_from_config = None
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                        model_type_from_config = config_data.get("model_type", "")
                        print(f"Detected model_type from config: {model_type_from_config}")

                loaded = False

                # For GLM-4V/4.6V models (GLM-4.6V-Flash is compatible with GLM-4.1V architecture)
                # Uses native transformers Glm4vForConditionalGeneration
                if model_type_from_config in ["glm4v", "glm46v"]:
                    if not GLM4V_AVAILABLE:
                        return "Error: GLM-4V not available. Requires transformers >= 4.57"

                    # CRITICAL: Patch config for transformers 4.57 compatibility
                    # GLM-4.6V config uses "rope_parameters" (5.0.0rc0 format)
                    # but transformers 4.57.3 expects "rope_scaling" (old format)
                    from transformers import AutoConfig
                    glm_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

                    # Check if text_config has rope_parameters but no rope_scaling
                    if hasattr(glm_config, 'text_config'):
                        text_cfg = glm_config.text_config
                        if hasattr(text_cfg, 'rope_parameters') and text_cfg.rope_parameters is not None:
                            if not hasattr(text_cfg, 'rope_scaling') or text_cfg.rope_scaling is None:
                                # Copy rope_parameters to rope_scaling for 4.57 compatibility
                                text_cfg.rope_scaling = text_cfg.rope_parameters
                                print(f"[GLM] Patched config: rope_parameters -> rope_scaling")

                    print("Loading as GLM-4V model using native Glm4vForConditionalGeneration...")
                    model_kwargs["config"] = glm_config
                    self.model = Glm4vForConditionalGeneration.from_pretrained(**model_kwargs)
                    print("Loaded as GLM-4V model")
                    loaded = True

                # For Qwen3-VL models, use AutoModelForVision2Seq which loads model's custom code
                # This is critical because Qwen3-VL has different architecture than Qwen2.5-VL
                if not loaded and model_type_from_config == "qwen3_vl":
                    print("Loading as Qwen3-VL model using AutoModelForVision2Seq...")
                    self.model = AutoModelForVision2Seq.from_pretrained(**model_kwargs)
                    print("Loaded as Qwen3-VL model (via AutoModelForVision2Seq)")
                    loaded = True

                # For Qwen3-VL MoE models
                if not loaded and QWEN3_VL_AVAILABLE and model_type_from_config == "qwen3_vl_moe":
                    try:
                        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(**model_kwargs)
                        print("Loaded as Qwen3-VL MoE model")
                        loaded = True
                    except Exception as e:
                        print(f"Qwen3-VL MoE load failed: {e}, trying fallback...")

                # For Qwen2.5-VL models
                if not loaded and model_type_from_config in ["qwen2_5_vl", "qwen2_vl"]:
                    try:
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**model_kwargs)
                        print("Loaded as Qwen2.5-VL model")
                        loaded = True
                    except Exception:
                        pass

                # Legacy Qwen2-VL fallback
                if not loaded:
                    try:
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(**model_kwargs)
                        print("Loaded as Qwen2-VL model")
                        loaded = True
                    except Exception:
                        pass

                # Final fallback
                if not loaded:
                    self.model = AutoModelForVision2Seq.from_pretrained(**model_kwargs)
                    print("Loaded as generic Vision2Seq model")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                print("Loaded as CausalLM model")

            self.model.eval()
            self.current_model_path = model_path
            self.current_model_type = model_type
            self.device_map = device_map

            # Apply FP8 optimization if requested (musubi-tuner style)
            if use_fp8:
                from backend.nn.fp8_optimization import (
                    optimize_vlm_state_dict_fp8,
                    apply_fp8_vlm_monkey_patch,
                    VLM_FP8_EXCLUDE_KEYS,
                )
                from safetensors import safe_open
                from accelerate import dispatch_model, infer_auto_device_map
                from accelerate.utils import get_balanced_memory

                progress(0.5, desc="Loading weights from safetensors...")
                print("[FP8] Loading weights directly from safetensors...")

                # Find safetensor files
                index_file = os.path.join(model_path, "model.safetensors.index.json")
                if os.path.exists(index_file):
                    with open(index_file, "r") as f:
                        index = json.load(f)
                    shard_files = sorted(set(index["weight_map"].values()))
                    safetensor_files = [os.path.join(model_path, f) for f in shard_files]
                else:
                    single_file = os.path.join(model_path, "model.safetensors")
                    if os.path.exists(single_file):
                        safetensor_files = [single_file]
                    else:
                        safetensor_files = sorted([str(f) for f in Path(model_path).glob("*.safetensors")])

                print(f"[FP8] Found {len(safetensor_files)} safetensor shard(s)")

                # Load state dict from safetensors
                state_dict = {}
                for shard_file in safetensor_files:
                    print(f"[FP8] Loading {os.path.basename(shard_file)}...")
                    with safe_open(shard_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)

                print(f"[FP8] Loaded {len(state_dict)} tensors from safetensors")

                # Apply FP8 quantization to state dict
                progress(0.7, desc="Quantizing to FP8...")
                print("[FP8] Applying FP8 quantization (skipping visual/embed/norm)...")
                state_dict = optimize_vlm_state_dict_fp8(
                    state_dict,
                    calc_device="cuda:0" if torch.cuda.is_available() else "cpu",
                    exclude_layer_keys=VLM_FP8_EXCLUDE_KEYS,
                    move_to_device=False,  # Keep on CPU for now
                )

                # Apply monkey patches to model
                print("[FP8] Applying forward patches for FP8 dequantization...")
                apply_fp8_vlm_monkey_patch(self.model, state_dict)

                # Load state dict with assign=True (replaces meta tensors)
                progress(0.85, desc="Loading quantized weights...")
                print("[FP8] Loading quantized state dict into model...")
                info = self.model.load_state_dict(state_dict, strict=False, assign=True)
                if info.missing_keys:
                    print(f"[FP8] Missing keys: {len(info.missing_keys)}")
                if info.unexpected_keys:
                    print(f"[FP8] Unexpected keys: {len(info.unexpected_keys)}")

                # Free state dict memory
                del state_dict
                gc.collect()

                # Initialize any remaining meta tensors (like inv_freq for rotary embeddings)
                # These are computed buffers not stored in checkpoints
                print("[FP8] Initializing remaining meta tensors...")
                meta_params = [(n, p) for n, p in self.model.named_parameters() if p.device.type == 'meta']
                meta_buffers = [(n, b) for n, b in self.model.named_buffers() if b.device.type == 'meta']

                for name, param in meta_params:
                    print(f"[FP8] Warning: param {name} still on meta device")

                for name, buf in meta_buffers:
                    # Get the parent module
                    parts = name.split('.')
                    buf_name = parts[-1]
                    module = self.model
                    for part in parts[:-1]:
                        module = getattr(module, part)

                    # Create initialized tensor
                    if 'inv_freq' in name:
                        # Rotary embedding inv_freq: 1/(base^(2i/dim))
                        dim = buf.shape[0] * 2
                        base = getattr(module, 'base', 10000.0) if hasattr(module, 'base') else 10000.0
                        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                        module._buffers[buf_name] = inv_freq
                        print(f"[FP8] Initialized {name} (rotary inv_freq, dim={dim})")
                    else:
                        # Generic: initialize with zeros on CPU
                        new_buf = torch.zeros(buf.shape, dtype=torch.float32, device='cpu')
                        module._buffers[buf_name] = new_buf
                        print(f"[FP8] Initialized {name} with zeros")

                print(f"[FP8] Initialized {len(meta_buffers)} meta buffers")

                # Dispatch to GPU
                progress(0.9, desc="Moving to GPU...")
                print("[FP8] Dispatching model to GPU...")

                # Calculate available GPU memory
                gpu_memory = {}
                for i in range(actual_gpus):
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    free_gb = int(free_mem / (1024**3) * 0.95)
                    gpu_memory[i] = f"{free_gb}GiB"
                    print(f"[FP8] GPU {i}: {free_gb}GB available")

                fp8_max_memory = get_balanced_memory(
                    self.model,
                    max_memory=gpu_memory,
                    no_split_module_classes=getattr(self.model, "_no_split_modules", None),
                )

                fp8_device_map = infer_auto_device_map(
                    self.model,
                    max_memory=fp8_max_memory,
                    no_split_module_classes=getattr(self.model, "_no_split_modules", None),
                )

                unique_devices = set(fp8_device_map.values())
                print(f"[FP8] Device map: {unique_devices}")

                self.model = dispatch_model(self.model, device_map=fp8_device_map)
                self.device_map = fp8_device_map
                print(f"[FP8] Model dispatched successfully!")

                # Apply torch.compile for optimized kernels
                print("[FP8] Applying torch.compile for optimized inference...")
                try:
                    # Use reduce-overhead mode for CUDA graphs (best for autoregressive)
                    # fullgraph=False allows for graph breaks with dynamic control flow
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    print("[FP8] torch.compile applied successfully!")
                except Exception as e:
                    print(f"[FP8] torch.compile failed (will use eager mode): {e}")

            progress(1.0, desc="Model loaded!")

            # Print device distribution
            if hasattr(self.model, "hf_device_map"):
                unique_devices = set(self.model.hf_device_map.values())
                print(f"Model distributed across devices: {unique_devices}")

            print_gpu_status()

            # Build status string based on quantization mode
            if use_fp8:
                dtype_str = "fp8_scaled"
            elif use_q4_nf4:
                dtype_str = "q4_nf4 (bf16 compute)"
            elif dtype == "q8_fp16":
                dtype_str = "q8_fp16 (fast)"
            elif dtype == "q8_partial":
                dtype_str = "q8_partial (bf16 - slow)"
            else:
                dtype_str = dtype
            return f"Loaded: {model_name} ({model_type}, {dtype_str}) on {actual_gpus} GPU(s)"

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
            self.tokenizer = None
            return f"Error loading model: {str(e)}"

    def unload_model(self) -> str:
        """Unload the current model and free GPU memory."""
        if self.model is None:
            return "No model loaded"

        model_name = Path(self.current_model_path).stem if self.current_model_path else "model"

        try:
            # Delete model and related objects
            del self.model
            if self.processor is not None:
                del self.processor
            if self.tokenizer is not None:
                del self.tokenizer

            self.model = None
            self.processor = None
            self.tokenizer = None
            self.current_model_path = None
            self.current_model_type = None
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
        if hasattr(self.model, "hf_device_map"):
            unique_devices = set(self.model.hf_device_map.values())
            device_str = f"devices: {unique_devices}"
        else:
            device_str = "single device"

        return f"Loaded: {model_name} ({self.current_model_type}, {device_str})"

    def generate(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        video_max_frames: int = 8,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from the VLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            images: Optional list of PIL Images
            video_path: Optional path to video file
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            video_max_frames: Max frames for video processing

        Returns:
            Tuple of (response string, stats dict)
        """
        empty_stats = {"tokens": 0, "time": 0.0, "tokens_per_sec": 0.0}

        if self.model is None:
            return "Error: No model loaded. Please load a model first.", empty_stats

        try:
            # Handle vision-language model (Qwen-VL, GLM-VL, etc.)
            if self.current_model_type in ["qwen-vl", "glm-vl"] and self.processor is not None:
                return self._generate_vl(
                    messages, images, video_path,
                    max_new_tokens, temperature, top_p, top_k, video_max_frames
                )
            # Handle text-only model
            elif self.tokenizer is not None:
                return self._generate_text(
                    messages, max_new_tokens, temperature, top_p, top_k
                )
            else:
                return "Error: No processor or tokenizer available", empty_stats

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during generation: {str(e)}", empty_stats

    def _generate_vl(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]],
        video_path: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        video_max_frames: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response from vision-language model."""
        start_time = time.perf_counter()

        # Build conversation in Qwen VL format
        conversation = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                conversation.append({"role": "system", "content": content})
            elif role == "user":
                if isinstance(content, list):
                    # Handle multimodal content
                    qwen_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                qwen_content.append({"type": "text", "text": item.get("text", "")})
                            elif item.get("type") == "image" and "image" in item:
                                img = item["image"]
                                if isinstance(img, Image.Image):
                                    # Save to temp file for Qwen processing
                                    temp_path = self._save_temp_image(img)
                                    qwen_content.append({"type": "image", "image": temp_path})
                            elif item.get("type") == "video" and "video" in item:
                                vid_path = item["video"]
                                if isinstance(vid_path, str) and os.path.exists(vid_path):
                                    qwen_content.append({
                                        "type": "video",
                                        "video": vid_path,
                                        "max_pixels": 360 * 420,
                                        "nframes": video_max_frames,
                                    })
                    if qwen_content:
                        conversation.append({"role": "user", "content": qwen_content})
                else:
                    conversation.append({"role": "user", "content": str(content)})
            elif role == "assistant":
                conversation.append({"role": "assistant", "content": str(content)})

        # Add direct images if provided
        if images:
            img_content = []
            for img in images:
                temp_path = self._save_temp_image(img)
                img_content.append({"type": "image", "image": temp_path})
            if conversation and conversation[-1]["role"] == "user":
                # Append images to last user message
                last_content = conversation[-1]["content"]
                if isinstance(last_content, str):
                    img_content.append({"type": "text", "text": last_content})
                    conversation[-1]["content"] = img_content
                elif isinstance(last_content, list):
                    conversation[-1]["content"] = img_content + last_content
            else:
                img_content.append({"type": "text", "text": "Describe this image."})
                conversation.append({"role": "user", "content": img_content})

        # Add video if provided
        if video_path and os.path.exists(video_path):
            vid_content = [{
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "nframes": video_max_frames,
            }]
            if conversation and conversation[-1]["role"] == "user":
                last_content = conversation[-1]["content"]
                if isinstance(last_content, str):
                    vid_content.append({"type": "text", "text": last_content})
                    conversation[-1]["content"] = vid_content
                elif isinstance(last_content, list):
                    conversation[-1]["content"] = vid_content + last_content
            else:
                vid_content.append({"type": "text", "text": "Describe this video."})
                conversation.append({"role": "user", "content": vid_content})

        # Check if this is a GLM model
        is_glm_model = self.current_model_type == "glm-vl"

        if is_glm_model:
            # GLM processor requires ALL content to be list of dicts with "type" keys
            # Images must be embedded in content as {"type": "image", "image": <PIL>}
            # NOT passed separately (causes "multiple values for keyword argument 'images'")
            glm_conversation = []

            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if isinstance(content, str):
                    # Convert string to list format
                    glm_content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    # Already list format - normalize and embed images directly
                    glm_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image":
                                img_data = item.get("image")
                                if img_data:
                                    if isinstance(img_data, str):
                                        # Load image from path and embed directly
                                        glm_content.append({"type": "image", "image": Image.open(img_data)})
                                    elif isinstance(img_data, Image.Image):
                                        # Already PIL image, embed directly
                                        glm_content.append({"type": "image", "image": img_data})
                                else:
                                    # No image data, just placeholder
                                    glm_content.append({"type": "image"})
                            elif item.get("type") == "text":
                                glm_content.append({"type": "text", "text": item.get("text", "")})
                            elif item.get("type") == "video":
                                glm_content.append(item)
                else:
                    glm_content = [{"type": "text", "text": str(content)}]

                glm_conversation.append({"role": role, "content": glm_content})

            # GLM's apply_chat_template - images embedded in content, NOT passed separately
            inputs = self.processor.apply_chat_template(
                glm_conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Remove token_type_ids if present (GLM-specific)
            inputs.pop("token_type_ids", None)
        else:
            # Qwen and other models: two-step process
            # Apply chat template (text only)
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process images/videos
            # Use qwen_vl_utils for Qwen models, manual extraction for others
            if QWEN_VL_UTILS_AVAILABLE:
                image_inputs, video_inputs = process_vision_info(conversation)
            else:
                # Manual extraction fallback
                image_inputs = []
                video_inputs = []
                for msg in conversation:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "image":
                                    img_path = item.get("image")
                                    if img_path:
                                        if isinstance(img_path, str):
                                            image_inputs.append(Image.open(img_path))
                                        elif isinstance(img_path, Image.Image):
                                            image_inputs.append(img_path)
                                elif item.get("type") == "video":
                                    vid_path = item.get("video")
                                    if vid_path:
                                        frames = extract_video_frames(vid_path, max_frames=video_max_frames)
                                        video_inputs.append(frames)

            # Prepare inputs via processor
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            )

        # Move inputs to the appropriate device
        if hasattr(self.model, "hf_device_map"):
            # Get the device of the first layer
            first_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(first_device, int):
                inputs = inputs.to(f"cuda:{first_device}")
            else:
                inputs = inputs.to(first_device)
        else:
            inputs = inputs.to(self.model.device)

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                top_k=top_k if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Decode output, removing input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        num_tokens = len(generated_ids_trimmed[0])
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0

        stats = {
            "tokens": num_tokens,
            "time": elapsed_time,
            "tokens_per_sec": tokens_per_sec,
        }
        print(f"Generated {num_tokens} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tok/s)")

        return output_text, stats

    def _generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response from text-only model."""
        start_time = time.perf_counter()

        # Build conversation
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)
            conversation.append({"role": role, "content": str(content)})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")

        # Move to appropriate device
        if hasattr(self.model, "hf_device_map"):
            first_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(first_device, int):
                inputs = inputs.to(f"cuda:{first_device}")
            else:
                inputs = inputs.to(first_device)
        else:
            inputs = inputs.to(self.model.device)

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                top_k=top_k if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Decode output
        generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
        output_text = self.tokenizer.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        num_tokens = len(generated_ids_trimmed)
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0

        stats = {
            "tokens": num_tokens,
            "time": elapsed_time,
            "tokens_per_sec": tokens_per_sec,
        }
        print(f"Generated {num_tokens} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tok/s)")

        return output_text, stats

    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL image to temporary file and return path."""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vlm_img_{int(time.time())}_{id(image)}.png")
        image.save(temp_path)
        return temp_path


# Global backend instance
vlm_backend: Optional[Qwen3VLMBackend] = None


def initialize_backend(models_dir: str = "models/LLM"):
    """Initialize the global VLM backend."""
    global vlm_backend
    vlm_backend = Qwen3VLMBackend(models_dir)


def refresh_models_handler():
    """Refresh the list of available models."""
    if vlm_backend is None:
        return gr.update(choices=["Backend not initialized"])
    models = vlm_backend.get_model_names()
    return gr.update(choices=models, value=models[0] if models else None)


def load_model_handler(
    model_name: str,
    dtype: str,
    num_gpus: int,
    max_memory_per_gpu: Optional[int],
    cpu_offload: bool = False,
    cpu_offload_ram: Optional[int] = None,
    progress=gr.Progress()
):
    """Handle model loading."""
    if vlm_backend is None:
        return "Backend not initialized"

    # Convert 0 to None for auto
    max_mem = None if max_memory_per_gpu == 0 else max_memory_per_gpu
    cpu_ram = None if cpu_offload_ram == 0 else cpu_offload_ram

    return vlm_backend.load_model(
        model_name=model_name,
        dtype=dtype,
        num_gpus=num_gpus,
        max_memory_per_gpu=max_mem,
        cpu_offload=cpu_offload,
        cpu_offload_ram=cpu_ram,
        progress=progress,
    )


def unload_model_handler():
    """Handle model unloading."""
    if vlm_backend is None:
        return "Backend not initialized"
    return vlm_backend.unload_model()


def status_handler():
    """Handle status request."""
    if vlm_backend is None:
        return "Backend not initialized"
    return vlm_backend.get_status()


def chat_handler(
    message: str,
    history: List[Dict[str, Any]],
    system_prompt: str,
    image,
    video,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    video_max_frames: int,
):
    """Handle chat messages from UI."""
    empty_stats = "Tokens: 0 | Time: 0.00s | Speed: 0.00 tok/s"

    if vlm_backend is None or vlm_backend.model is None:
        error_history = list(history)
        error_history.append({"role": "user", "content": message})
        error_history.append({"role": "assistant", "content": "Error: No model loaded."})
        return error_history, "", empty_stats

    # Build messages list
    messages = []

    # Add system prompt if provided
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # Add chat history
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        messages.append({"role": role, "content": " ".join(text_parts)})

    # Build current message content
    model_content = []

    if image is not None:
        model_content.append({"type": "image", "image": image})

    if video is not None:
        model_content.append({"type": "video", "video": video})

    if message.strip():
        model_content.append({"type": "text", "text": message})
    elif not model_content:
        model_content.append({"type": "text", "text": "Describe this."})

    messages.append({"role": "user", "content": model_content})

    # Generate response
    response, stats = vlm_backend.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        video_max_frames=video_max_frames,
    )

    # Format stats string
    stats_str = f"Tokens: {stats['tokens']} | Time: {stats['time']:.2f}s | Speed: {stats['tokens_per_sec']:.2f} tok/s"

    # Build display content
    new_history = list(history)

    if image is not None:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vlm_chat_{int(time.time())}_{id(image)}.png")
        image.save(temp_path)
        display_text = message if message else "Describe this image"
        new_history.append({"role": "user", "content": display_text})
        new_history.append({"role": "user", "content": gr.Image(temp_path)})
        new_history.append({"role": "assistant", "content": response})
    elif video is not None:
        display_text = message if message else "Describe this video"
        new_history.append({"role": "user", "content": display_text})
        new_history.append({"role": "user", "content": gr.Video(video)})
        new_history.append({"role": "assistant", "content": response})
    else:
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": response})

    return new_history, "", stats_str


def clear_chat_handler():
    """Clear chat history."""
    return []


def batch_caption_handler(
    folder_path: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    progress=gr.Progress(),
):
    """Process a folder of images and generate captions."""
    if vlm_backend is None or vlm_backend.model is None:
        return "Error: No model loaded."

    if not folder_path or not os.path.isdir(folder_path):
        return f"Error: Invalid folder path: {folder_path}"

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    image_files = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            image_files.append(f)

    if not image_files:
        return f"No images found in {folder_path}"

    results = []
    total = len(image_files)

    for i, filename in enumerate(image_files):
        progress((i / total), desc=f"Processing {filename}...")

        image_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(image_path).convert("RGB")

            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})

            content = [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]
            messages.append({"role": "user", "content": content})

            caption, stats = vlm_backend.generate(
                messages=messages,
                images=[img],
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(folder_path, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            results.append(f"[OK] {filename} -> {base_name}.txt ({stats['tokens']} tokens, {stats['tokens_per_sec']:.1f} tok/s)")

        except Exception as e:
            results.append(f"[ERROR] {filename}: {str(e)}")

    progress(1.0, desc="Complete!")
    return f"Processed {total} images:\n\n" + "\n".join(results)


def create_ui():
    """Create the Gradio interface."""
    # Theme
    vlm_theme = themes.Default(
        primary_hue=colors.Color(
            name="custom",
            c50="#E6F0FF",
            c100="#CCE0FF",
            c200="#99C1FF",
            c300="#66A3FF",
            c400="#3384FF",
            c500="#0060df",
            c600="#0052C2",
            c700="#003D91",
            c800="#002961",
            c900="#001430",
            c950="#000A18"
        )
    )

    vlm_css = """
    .green-btn {
        background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
        color: white !important;
        border: none !important;
    }
    .green-btn:hover {
        background: linear-gradient(to bottom right, #27ae60, #219651) !important;
    }
    .stats-display {
        font-family: monospace;
        font-size: 14px;
        padding: 8px 12px;
        background: #1a1a2e;
        border-radius: 5px;
        color: #4ade80;
    }
    """

    # Get initial model list
    initial_models = vlm_backend.get_model_names() if vlm_backend else ["Initialize backend first"]
    num_gpus = vlm_backend.num_gpus if vlm_backend else 0

    with gr.Blocks(title="Chromaforge VLM (Diffusers)", theme=vlm_theme, css=vlm_css) as demo:
        gr.Markdown("# Chromaforge VLM Chat (Diffusers/Transformers Backend)")

        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("Chat"):
                # Chat interface at top - full width
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
                )

                # Media inputs row - max 300px
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image (optional)",
                            type="pil",
                            height=300,
                        )
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video (optional)",
                            height=300,
                        )

                # Message input row
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # Clear and stats row
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", scale=1)
                    stats_display = gr.Textbox(
                        label="Generation Stats",
                        value="Tokens: 0 | Time: 0.00s | Speed: 0.00 tok/s",
                        interactive=False,
                        scale=3,
                        elem_classes=["stats-display"],
                    )

            # Batch Caption Tab
            with gr.TabItem("Batch Caption"):
                gr.Markdown("Generate captions for all images in a folder.")

                batch_folder = gr.Textbox(
                    label="Folder Path",
                    placeholder="Enter path to folder...",
                    lines=1,
                )

                batch_system_prompt = gr.Textbox(
                    label="System Prompt",
                    lines=3,
                    value="You are an image captioning assistant. Provide detailed, accurate descriptions.",
                )

                batch_prompt = gr.Textbox(
                    label="Caption Prompt",
                    lines=2,
                    value="Describe this image in detail.",
                )

                batch_start_btn = gr.Button(
                    "Start Batch Captioning",
                    variant="primary",
                    elem_classes=["green-btn"],
                )

                batch_output = gr.Textbox(
                    label="Output",
                    lines=15,
                    interactive=False,
                )

        # Settings section below chat - in accordions
        with gr.Accordion("Model Settings", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Select Model",
                            choices=initial_models,
                            value=initial_models[0] if initial_models else None,
                            interactive=True,
                            scale=4,
                        )
                        refresh_models_btn = gr.Button("Refresh", scale=1, min_width=80)

                with gr.Column(scale=1):
                    dtype_dropdown = gr.Dropdown(
                        label="Precision",
                        choices=[
                            "q8_fp16",      # INT8 + fp16 vision (fast, ~30GB)
                            "q4_nf4",       # 4-bit NF4 + bf16 compute (~20GB)
                            "fp8_scaled",   # FP8 manual (RTX 40/50 native, ~31GB)
                            "q8_partial",   # INT8 + bf16 vision (slow - casting)
                            "bfloat16",     # Full bf16 (~62GB)
                            "float16",      # Full fp16 (~62GB)
                        ],
                        value="q8_fp16",
                        info="q8_fp16: fast INT8 (~30GB) | q4_nf4: smallest (~20GB)",
                    )

                with gr.Column(scale=1):
                    num_gpus_slider = gr.Slider(
                        minimum=1,
                        maximum=max(8, num_gpus),
                        value=min(2, num_gpus) if num_gpus > 0 else 1,
                        step=1,
                        label="Number of GPUs",
                        info=f"Detected {num_gpus} GPU(s)",
                    )

                with gr.Column(scale=1):
                    max_memory_slider = gr.Slider(
                        minimum=0,
                        maximum=48,
                        value=0,
                        step=1,
                        label="Max Memory/GPU (GB)",
                        info="0 = Auto",
                    )

                with gr.Column(scale=1):
                    cpu_offload_checkbox = gr.Checkbox(
                        label="CPU Offload",
                        value=False,
                        info="Offload layers to CPU RAM",
                    )
                    cpu_ram_slider = gr.Slider(
                        minimum=0,
                        maximum=256,
                        value=64,
                        step=8,
                        label="CPU RAM (GB)",
                        info="Max RAM for offloading",
                    )

            with gr.Row():
                load_model_btn = gr.Button("Load Model", variant="primary")
                unload_model_btn = gr.Button("Unload", variant="secondary")
                model_status = gr.Textbox(
                    label="Status",
                    value="No model loaded",
                    interactive=False,
                    scale=3,
                )

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter a system prompt...",
                    lines=2,
                    value="You are a helpful AI assistant that can understand and describe images and videos in detail.",
                    scale=3,
                )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    value=512,
                    step=64,
                    label="Max New Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K",
                )
                video_max_frames = gr.Slider(
                    minimum=1,
                    maximum=64,
                    value=16,
                    step=1,
                    label="Max Video Frames",
                )

        # Event handlers
        refresh_models_btn.click(
            fn=refresh_models_handler,
            outputs=[model_dropdown],
        )

        load_model_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, dtype_dropdown, num_gpus_slider, max_memory_slider, cpu_offload_checkbox, cpu_ram_slider],
            outputs=[model_status],
        )

        unload_model_btn.click(
            fn=unload_model_handler,
            outputs=[model_status],
        )

        def send_message(msg, history, sys_prompt, img, vid, max_tok, temp, tp, tk, vid_frames):
            if not msg.strip() and img is None and vid is None:
                return history, "", None, None, "Tokens: 0 | Time: 0.00s | Speed: 0.00 tok/s"

            new_history, _, stats_str = chat_handler(
                msg, history, sys_prompt, img, vid,
                max_tok, temp, tp, tk, vid_frames
            )
            return new_history, "", None, None, stats_str

        send_btn.click(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, top_p, top_k, video_max_frames
            ],
            outputs=[chatbot, msg_input, image_input, video_input, stats_display],
        )

        msg_input.submit(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, top_p, top_k, video_max_frames
            ],
            outputs=[chatbot, msg_input, image_input, video_input, stats_display],
        )

        clear_btn.click(
            fn=clear_chat_handler,
            outputs=[chatbot],
        )

        batch_start_btn.click(
            fn=batch_caption_handler,
            inputs=[
                batch_folder, batch_prompt, batch_system_prompt,
                max_tokens, temperature
            ],
            outputs=[batch_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Chromaforge VLM (Diffusers Backend)")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/LLM",
        help="Directory containing VLM models (default: models/LLM)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7863,
        help="Port to run the server on (default: 7863)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Listen on 0.0.0.0 to enable LAN access",
    )

    args = parser.parse_args()

    host = "0.0.0.0" if args.listen else args.host

    print("=" * 60)
    print("Chromaforge VLM Chat (Diffusers/Transformers Backend)")
    print("Supported: Qwen-VL, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V")
    print("=" * 60)
    print(f"Transformers: {'available' if TRANSFORMERS_AVAILABLE else 'NOT INSTALLED'}")
    print(f"Accelerate: {'available' if ACCELERATE_AVAILABLE else 'NOT INSTALLED'}")
    print(f"qwen-vl-utils: {'available' if QWEN_VL_UTILS_AVAILABLE else 'NOT INSTALLED (optional, only for Qwen)'}")
    print(f"Models directory: {args.models_dir}")
    print(f"Server: http://{host}:{args.port}")
    if args.listen:
        print("LAN access: enabled")
    print("=" * 60)

    if not TRANSFORMERS_AVAILABLE:
        print("\nERROR: transformers not installed!")
        print("\nInstall with:")
        print("  pip install transformers>=4.51.0 accelerate qwen-vl-utils")
        return

    # Initialize the backend
    initialize_backend(args.models_dir)

    # List found models
    models = vlm_backend.get_available_models()
    if models:
        print(f"\nFound {len(models)} model(s):")
        for m in models:
            print(f"  - {m['name']} ({m['type']})")
    else:
        print(f"\nNo models found in {args.models_dir}")
        print("Download Qwen-VL or GLM-4V models and place them in this directory.")

    print("=" * 60)

    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_name=host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
