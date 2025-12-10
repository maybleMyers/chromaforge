"""
VLM Backend using Diffusers/Transformers for Qwen3-VL Models
Supports loading unquantized models across multiple GPUs.

Requirements:
- pip install transformers>=4.51.0 accelerate qwen-vl-utils torch torchvision
- For multi-GPU: pip install accelerate

Usage:
    python vlm_diffusers.py --models-dir models/VLM --port 7863
"""

import os
import gc
import time
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
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    QWEN3_VL_AVAILABLE = False
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
        except Exception:
            pass

        # Only include Qwen models
        if model_type in ["qwen", "qwen-vl"]:
            models.append({
                "name": model_dir.name,
                "path": str(model_dir),
                "type": model_type,
            })

    return sorted(models, key=lambda x: x["name"])


class Qwen3VLMBackend:
    """
    Diffusers/Transformers-based VLM backend for Qwen3-VL models.
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

            # Determine torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
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
            if model_type == "qwen-vl":
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

            progress(0.3, desc=f"Loading {model_name} across {actual_gpus} GPU(s)...")
            print(f"Loading model from: {model_path}")
            print(f"Dtype: {torch_dtype}, Device map: {device_map}")

            # Load the model
            model_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "dtype": torch_dtype,
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory

            # Use appropriate model class based on type
            if model_type == "qwen-vl":
                # Try Qwen3-VL MoE first (for newer models), then Qwen2.5-VL, then Qwen2-VL
                loaded = False

                if QWEN3_VL_AVAILABLE:
                    try:
                        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(**model_kwargs)
                        print("Loaded as Qwen3-VL MoE model")
                        loaded = True
                    except Exception as e:
                        print(f"Qwen3-VL MoE load failed: {e}, trying fallback...")

                if not loaded:
                    try:
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**model_kwargs)
                        print("Loaded as Qwen2.5-VL model")
                        loaded = True
                    except Exception:
                        pass

                if not loaded:
                    try:
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(**model_kwargs)
                        print("Loaded as Qwen2-VL model")
                        loaded = True
                    except Exception:
                        pass

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

            progress(1.0, desc="Model loaded!")

            # Print device distribution
            if hasattr(self.model, "hf_device_map"):
                unique_devices = set(self.model.hf_device_map.values())
                print(f"Model distributed across devices: {unique_devices}")

            print_gpu_status()

            return f"Loaded: {model_name} ({model_type}) on {actual_gpus} GPU(s)"

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
            # Handle vision-language model
            if self.current_model_type == "qwen-vl" and self.processor is not None:
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

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process images/videos with qwen_vl_utils if available
        if QWEN_VL_UTILS_AVAILABLE:
            image_inputs, video_inputs = process_vision_info(conversation)
        else:
            # Fallback: extract images manually
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
                                    image_inputs.append(Image.open(img_path))
                            elif item.get("type") == "video":
                                vid_path = item.get("video")
                                if vid_path:
                                    frames = extract_video_frames(vid_path, max_frames=video_max_frames)
                                    video_inputs.append(frames)

        # Prepare inputs
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
                        choices=["bfloat16", "float16", "float32"],
                        value="bfloat16",
                        info="bfloat16 recommended for Qwen models",
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
    print("=" * 60)
    print(f"Transformers: {'available' if TRANSFORMERS_AVAILABLE else 'NOT INSTALLED'}")
    print(f"Accelerate: {'available' if ACCELERATE_AVAILABLE else 'NOT INSTALLED'}")
    print(f"qwen-vl-utils: {'available' if QWEN_VL_UTILS_AVAILABLE else 'NOT INSTALLED'}")
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
        print("Download Qwen3-VL models and place them in this directory.")

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
