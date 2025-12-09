"""
VLM Chat Interface for Qwen Vision-Language Models
A standalone GUI for interacting with Qwen VL models using images, videos, and text.
"""

import os
import sys
import gc
import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
from PIL import Image

# Try to import video processing utilities
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Video support will be limited.")

# Try to import vLLM for high-performance inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("vLLM loaded successfully")
except ImportError as e:
    VLLM_AVAILABLE = False
    print(f"Warning: vLLM not available. Install with: pip install vllm>=0.11.0")
    print(f"  Import error: {e}")
except Exception as e:
    VLLM_AVAILABLE = False
    print(f"Warning: vLLM import failed: {e}")

# Try to import qwen-vl-utils for image processing
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen-vl-utils not installed. Install with: pip install qwen-vl-utils")

# Default model paths (relative to models/LLM)
DEFAULT_MODELS = {
    "Qwen3-VL-8B-Caption-V4.5": "models/LLM/Qwen3-VL-8B-Caption-V4.5",
    "Qwen3-VL-4B-Instruct": "models/LLM/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-30B-A3B-Instruct": "models/LLM/Qwen3-VL-30B-A3B-Instruct",
}


def extract_video_frames(video_path: str, max_frames: int = 8, target_size: Tuple[int, int] = (448, 448)) -> List[Image.Image]:
    """
    Extract frames from a video file for VLM processing.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        target_size: Target size for frames (width, height)

    Returns:
        List of PIL Images
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv-python is required for video processing. Install with: pip install opencv-python")

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
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                # Resize to target size
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
                frames.append(pil_image)
    finally:
        cap.release()

    return frames


class VLMManager:
    """Manages Qwen VL model loading, inference, and memory."""

    def __init__(self, low_vram: bool = False, backend: str = "auto"):
        """
        Initialize VLM Manager.

        Args:
            low_vram: Enable low VRAM mode for transformers backend
            backend: "vllm", "transformers", or "auto" (vLLM if available, else transformers)
        """
        self.model = None
        self.processor = None
        self.model_name = None
        self.low_vram = low_vram
        self.device = self._get_device()

        # vLLM specific attributes
        self.vllm_model = None
        self.model_path = None

        # Determine backend
        if backend == "auto":
            self.backend = "vllm" if VLLM_AVAILABLE else "transformers"
        else:
            self.backend = backend

        if self.backend == "vllm" and not VLLM_AVAILABLE:
            print("Warning: vLLM requested but not available. Falling back to transformers.")
            self.backend = "transformers"

        print(f"VLM Backend: {self.backend}")

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_available_models(self) -> List[str]:
        """Scan for available models in the models/LLM directory."""
        models = []
        llm_dir = Path("models/LLM")

        if llm_dir.exists():
            for item in llm_dir.iterdir():
                if item.is_dir():
                    # Check if it looks like a valid model directory
                    if (item / "config.json").exists() or (item / "model.safetensors").exists():
                        models.append(item.name)

        # Also check default paths
        for name, path in DEFAULT_MODELS.items():
            if Path(path).exists() and name not in models:
                models.append(name)

        return sorted(models) if models else ["No models found"]

    def _detect_model_type(self, model_path: str) -> str:
        """Detect the model type from config.json."""
        import json
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                # Check for MoE models first
                if "moe" in model_type:
                    if "qwen3" in model_type:
                        return "qwen3_vl_moe"
                # Then standard models
                if "qwen3" in model_type:
                    return "qwen3_vl"
                elif "qwen2_5" in model_type or "qwen2.5" in model_type:
                    return "qwen2_5_vl"
                elif "qwen2" in model_type:
                    return "qwen2_vl"
            except Exception as e:
                print(f"Warning: Could not read config.json: {e}")

        # Fallback to name-based detection
        model_name_lower = Path(model_path).name.lower()
        if "qwen3" in model_name_lower:
            return "qwen3_vl"
        elif "qwen2.5" in model_name_lower or "qwen2_5" in model_name_lower:
            return "qwen2_5_vl"

        return "qwen2_5_vl"  # Default fallback

    def _load_with_vllm(self, model_name: str, quantization: str = "none", progress=gr.Progress()) -> str:
        """Load model using vLLM backend for high-performance inference."""
        if not VLLM_AVAILABLE:
            return "vLLM is not available. Please install with: pip install vllm>=0.11.0"

        # Check if already loaded
        if self.vllm_model is not None and self.model_name == model_name:
            return f"Model '{model_name}' is already loaded (vLLM)."

        # Unload existing model first
        if self.vllm_model is not None:
            self.unload_model()

        progress(0.1, desc="Loading model with vLLM...")

        # Determine model path
        if model_name in DEFAULT_MODELS:
            model_path = DEFAULT_MODELS[model_name]
        else:
            model_path = f"models/LLM/{model_name}"

        if not Path(model_path).exists():
            return f"Model path not found: {model_path}"

        try:
            # Detect model type
            model_type = self._detect_model_type(model_path)
            print(f"Detected model type: {model_type}")

            progress(0.3, desc="Initializing vLLM engine...")

            # Configure vLLM loading options
            vllm_kwargs = {
                "model": model_path,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "max_model_len": 4096,  # Adjust based on your VRAM
                "gpu_memory_utilization": 0.9,
            }

            # Handle quantization
            if quantization == "4bit":
                vllm_kwargs["quantization"] = "awq"  # or "gptq" depending on model
                print("Using AWQ 4-bit quantization with vLLM")
            elif quantization == "8bit":
                vllm_kwargs["quantization"] = "fp8"
                print("Using FP8 quantization with vLLM")

            # Enable multimodal for VL models
            vllm_kwargs["limit_mm_per_prompt"] = {"image": 10, "video": 2}

            progress(0.5, desc=f"Loading {model_type} with vLLM...")

            self.vllm_model = LLM(**vllm_kwargs)
            self.model_path = model_path
            self.model_name = model_name

            # Also load processor for chat template
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path)

            progress(1.0, desc="Model loaded with vLLM!")

            quant_info = f", {quantization}" if quantization != "none" else ""
            return f"Successfully loaded '{model_name}' with vLLM ({model_type}{quant_info})"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Failed to load model with vLLM: {str(e)}"

    def load_model(self, model_name: str, quantization: str = "none", use_flash_attn: bool = False, vram_buffer: int = 0, progress=gr.Progress()) -> str:
        """Load a Qwen VL model.

        Args:
            model_name: Name of the model to load
            quantization: "none", "4bit", or "8bit"
            use_flash_attn: Whether to use Flash Attention 2
            vram_buffer: GB of VRAM to reserve (for loading large models)
            progress: Gradio progress callback
        """
        if model_name == "No models found":
            return "No models available. Please download a model first."

        # Use vLLM backend if selected
        if self.backend == "vllm":
            return self._load_with_vllm(model_name, quantization, progress)

        # Check if already loaded
        if self.model is not None and self.model_name == model_name:
            return f"Model '{model_name}' is already loaded."

        # Unload existing model first
        if self.model is not None:
            self.unload_model()

        progress(0.1, desc="Loading model...")

        # Determine model path
        if model_name in DEFAULT_MODELS:
            model_path = DEFAULT_MODELS[model_name]
        else:
            model_path = f"models/LLM/{model_name}"

        if not Path(model_path).exists():
            return f"Model path not found: {model_path}"

        try:
            # Detect model type from config
            model_type = self._detect_model_type(model_path)
            print(f"Detected model type: {model_type}")

            from transformers import AutoProcessor

            progress(0.3, desc="Loading processor...")
            self.processor = AutoProcessor.from_pretrained(model_path)

            progress(0.5, desc=f"Loading model weights ({model_type})...")

            # Select the correct model class based on detected type
            if model_type == "qwen3_vl_moe":
                from transformers import Qwen3VLMoeForConditionalGeneration as ModelClass
            elif model_type == "qwen3_vl":
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

            # Build loading kwargs
            load_kwargs = {
                "low_cpu_mem_usage": True,
            }

            # Configure quantization for faster inference with less VRAM
            if quantization == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    # 4-bit models must use device_map for proper loading
                    load_kwargs["device_map"] = "auto"
                    print("Using 4-bit quantization (NF4)")
                except ImportError:
                    print("Warning: bitsandbytes not installed, falling back to bfloat16")
                    load_kwargs["torch_dtype"] = torch.bfloat16
                    load_kwargs["device_map"] = "auto"
            elif quantization == "8bit":
                try:
                    from transformers import BitsAndBytesConfig

                    # Clear GPU memory before loading
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )

                    # Apply VRAM buffer if specified
                    if vram_buffer > 0 and torch.cuda.is_available():
                        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        max_gpu = int(total_mem_gb - vram_buffer)
                        if max_gpu > 0:
                            load_kwargs["device_map"] = "auto"
                            load_kwargs["max_memory"] = {0: f"{max_gpu}GiB", "cpu": "100GiB"}
                            offload_dir = Path(tempfile.gettempdir()) / "vlm_offload"
                            offload_dir.mkdir(exist_ok=True)
                            load_kwargs["offload_folder"] = str(offload_dir)
                            print(f"Using 8-bit quantization (max GPU: {max_gpu}GB, buffer: {vram_buffer}GB)")
                        else:
                            load_kwargs["device_map"] = "auto"
                            print("Using 8-bit quantization...")
                    else:
                        load_kwargs["device_map"] = "auto"
                        print("Using 8-bit quantization...")

                except ImportError as e:
                    print(f"Warning: bitsandbytes not installed, falling back to bfloat16")
                    load_kwargs["torch_dtype"] = torch.bfloat16
                    load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16
                # Set device map based on low_vram setting
                if self.low_vram:
                    load_kwargs["device_map"] = {"": self.device}
                else:
                    load_kwargs["device_map"] = "auto"

            # Use Flash Attention 2 if requested
            if use_flash_attn:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")

            self.model = ModelClass.from_pretrained(model_path, **load_kwargs)

            self.model_name = model_name
            progress(1.0, desc="Model loaded!")

            quant_info = f", {quantization}" if quantization != "none" else ""
            flash_info = ", flash_attn" if use_flash_attn else ""
            return f"Successfully loaded '{model_name}' ({model_type}{quant_info}{flash_info})"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Failed to load model: {str(e)}"

    def unload_model(self) -> str:
        """Unload the current model to free memory."""
        # Check if any model is loaded (either vLLM or transformers)
        if self.model is None and self.vllm_model is None:
            return "No model is currently loaded."

        model_name = self.model_name
        backend_used = "vLLM" if self.vllm_model is not None else "transformers"

        # Unload vLLM model
        if self.vllm_model is not None:
            del self.vllm_model
            self.vllm_model = None
            self.model_path = None

        # Unload transformers model
        if self.model is not None:
            del self.model
            self.model = None

        # Clean up processor
        if self.processor is not None:
            del self.processor
            self.processor = None

        self.model_name = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return f"Unloaded '{model_name}' ({backend_used}) and freed memory."

    def get_memory_info(self) -> str:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return "CUDA not available"

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total"

    def _generate_with_vllm(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        video_max_frames: int = 8,
    ) -> str:
        """Generate a response using vLLM backend."""
        if self.vllm_model is None:
            return "Error: No vLLM model loaded."

        try:
            # Process messages to extract images and prepare for vLLM
            images = []
            processed_messages = []

            for msg in messages:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    for item in msg["content"]:
                        if item.get("type") == "image" and "image" in item:
                            img = item["image"]
                            images.append(img)
                            # For vLLM, use placeholder in text
                            new_content.append({"type": "image"})
                        elif item.get("type") == "video" and "video" in item:
                            # Process video into frames
                            video_path = item["video"]
                            if isinstance(video_path, str) and os.path.exists(video_path):
                                try:
                                    frames = extract_video_frames(video_path, max_frames=video_max_frames)
                                    for frame in frames:
                                        images.append(frame)
                                        new_content.append({"type": "image"})
                                    if frames:
                                        new_content.append({"type": "text", "text": f"[The above {len(frames)} images are frames extracted from a video]"})
                                except Exception as e:
                                    new_content.append({"type": "text", "text": f"[Video processing error: {str(e)}]"})
                        elif item.get("type") == "text":
                            new_content.append(item)
                        else:
                            new_content.append(item)
                    processed_messages.append({"role": msg["role"], "content": new_content})
                else:
                    processed_messages.append(msg)

            # Apply chat template using processor
            text_input = self.processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            print(f"[vLLM Debug] Prompt preview: {text_input[:500]}...")
            print(f"[vLLM Debug] Number of images: {len(images)}")

            # Configure sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.001,
                top_p=top_p,
                top_k=top_k if top_k > 0 else -1,
                repetition_penalty=repetition_penalty,
            )

            # Prepare multimodal inputs for vLLM
            if images:
                # Convert PIL images to format vLLM expects
                mm_data = {"image": images}
                inputs = {
                    "prompt": text_input,
                    "multi_modal_data": mm_data,
                }
            else:
                inputs = {"prompt": text_input}

            # Generate with timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            outputs = self.vllm_model.generate([inputs], sampling_params=sampling_params)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Extract response
            response = outputs[0].outputs[0].text

            # Calculate throughput
            num_generated_tokens = len(outputs[0].outputs[0].token_ids)
            generation_time = end_time - start_time
            tokens_per_sec = num_generated_tokens / generation_time if generation_time > 0 else 0

            print(f"[vLLM Inference] Generated {num_generated_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.2f} tok/s)")

            # Clean up thinking tags if present
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during vLLM generation: {str(e)}"

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        video_max_frames: int = 8,
    ) -> str:
        """Generate a response from the model."""
        # Use vLLM backend if loaded
        if self.vllm_model is not None:
            return self._generate_with_vllm(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                video_max_frames=video_max_frames,
            )

        if self.model is None:
            return "Error: No model loaded. Please load a model first."

        try:
            # Extract images and videos from messages, process videos into frames
            images = []
            processed_messages = []

            for msg in messages:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    for item in msg["content"]:
                        if item.get("type") == "image" and "image" in item:
                            images.append(item["image"])
                            new_content.append({"type": "image", "image": item["image"]})
                        elif item.get("type") == "video" and "video" in item:
                            # Process video into frames
                            video_path = item["video"]
                            if isinstance(video_path, str) and os.path.exists(video_path):
                                try:
                                    frames = extract_video_frames(video_path, max_frames=video_max_frames)
                                    # Add each frame as an image
                                    for frame in frames:
                                        images.append(frame)
                                        new_content.append({"type": "image", "image": frame})
                                    # Add a note about video frames
                                    if frames:
                                        new_content.append({"type": "text", "text": f"[The above {len(frames)} images are frames extracted from a video]"})
                                except Exception as e:
                                    new_content.append({"type": "text", "text": f"[Video processing error: {str(e)}]"})
                            else:
                                new_content.append({"type": "text", "text": "[Video file not found]"})
                        else:
                            new_content.append(item)
                    processed_messages.append({"role": msg["role"], "content": new_content})
                else:
                    processed_messages.append(msg)

            # Apply chat template
            print(f"[Debug] Messages being sent to model: {len(processed_messages)} messages")
            for i, msg in enumerate(processed_messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    preview = content[:100] + "..." if len(content) > 100 else content
                else:
                    preview = f"[{len(content)} content items]"
                print(f"  [{i}] {role}: {preview}")

            text_input = self.processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            print(f"[Debug] Formatted prompt preview: {text_input[:500]}...")

            # Process inputs
            process_kwargs = {"text": [text_input], "padding": True, "return_tensors": "pt"}

            if images:
                process_kwargs["images"] = images

            inputs = self.processor(**process_kwargs)

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate with timing
            input_len = inputs['input_ids'].shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Calculate throughput
            generated_ids = outputs[0][input_len:]
            num_generated_tokens = len(generated_ids)
            generation_time = end_time - start_time
            tokens_per_sec = num_generated_tokens / generation_time if generation_time > 0 else 0

            print(f"[Inference] Generated {num_generated_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.2f} tok/s)")

            response = self.processor.decode(generated_ids, skip_special_tokens=True)

            # Clean up thinking tags if present
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during generation: {str(e)}"


# Global manager instance
vlm_manager: Optional[VLMManager] = None


def initialize_manager(low_vram: bool = False, backend: str = "auto"):
    """Initialize the global VLM manager."""
    global vlm_manager
    vlm_manager = VLMManager(low_vram=low_vram, backend=backend)


def switch_backend_handler(backend: str):
    """Handle backend switching from UI."""
    global vlm_manager
    if vlm_manager is not None:
        # Unload current model first
        vlm_manager.unload_model()

    # Get current low_vram setting
    low_vram = vlm_manager.low_vram if vlm_manager else False

    # Reinitialize with new backend
    vlm_manager = VLMManager(low_vram=low_vram, backend=backend)
    return f"Switched to {vlm_manager.backend} backend"


def load_model_handler(model_name: str, quantization: str, use_flash_attn: bool, vram_buffer: int, progress=gr.Progress()):
    """Handle model loading from UI."""
    if vlm_manager is None:
        return "Manager not initialized"
    return vlm_manager.load_model(model_name, quantization, use_flash_attn, int(vram_buffer), progress)


def unload_model_handler():
    """Handle model unloading from UI."""
    if vlm_manager is None:
        return "Manager not initialized"
    return vlm_manager.unload_model()


def get_memory_handler():
    """Handle memory info request from UI."""
    if vlm_manager is None:
        return "Manager not initialized"
    return vlm_manager.get_memory_info()


def chat_handler(
    message: str,
    history: List[Tuple[str, str]],
    system_prompt: str,
    image,
    video,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    video_max_frames: int = 8,
    auto_unload: bool = False,
):
    """Handle chat messages from UI."""
    # Check if any model is loaded (either transformers or vLLM)
    if vlm_manager is None or (vlm_manager.model is None and vlm_manager.vllm_model is None):
        return history + [(message, "Error: No model loaded. Please load a model first.")], ""

    # Build messages list for the model
    messages = []

    # Add system prompt if provided
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # Add chat history (convert from Gradio tuple format to model format)
    for user_msg, assistant_msg in history:
        # Skip image-only entries (where assistant_msg is None)
        if assistant_msg is None:
            continue

        # Extract text from user message (may contain tuples for files)
        if isinstance(user_msg, tuple):
            # Tuple format is (filepath,) for images - skip these in history
            continue
        elif isinstance(user_msg, str):
            user_text = user_msg
        else:
            user_text = str(user_msg)

        if user_text:  # Only add non-empty messages
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_msg})

    # Build current message content for the model
    model_content = []

    if image is not None:
        model_content.append({"type": "image", "image": image})

    if video is not None:
        model_content.append({"type": "video", "video": video})

    model_content.append({"type": "text", "text": message})

    messages.append({"role": "user", "content": model_content})

    # Generate response
    response = vlm_manager.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        video_max_frames=video_max_frames,
    )

    # Build display content for chatbot (classic tuple format)
    # Format: [(user_msg, bot_msg), ...] where user_msg can be (filepath,) for files
    new_history = list(history)

    if image is not None:
        # Save image to temp file for display
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vlm_chat_{id(image)}.png")
        image.save(temp_path)
        # Add image as separate message, then text+response
        new_history.append(((temp_path,), None))
        if message:
            new_history.append((message, response))
        else:
            # If no text, attach response to a placeholder
            new_history.append(("[Describe this image]", response))
    elif video is not None:
        new_history.append(((video,), None))
        if message:
            new_history.append((message, response))
        else:
            new_history.append(("[Describe this video]", response))
    else:
        new_history.append((message, response))

    # Auto-unload if requested
    status_msg = ""
    if auto_unload and vlm_manager.model is not None:
        status_msg = vlm_manager.unload_model()

    return new_history, status_msg


def clear_chat_handler():
    """Clear chat history."""
    return []


def batch_caption_handler(
    folder_path: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    progress=gr.Progress(),
):
    """Process a folder of images and generate captions."""
    if vlm_manager is None or vlm_manager.model is None:
        return "Error: No model loaded. Please load a model first."

    if not folder_path or not os.path.isdir(folder_path):
        return f"Error: Invalid folder path: {folder_path}"

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    # Find all images in folder
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
            # Load image
            img = Image.open(image_path).convert("RGB")

            # Build messages
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})

            content = [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]
            messages.append({"role": "user", "content": content})

            # Generate caption
            caption = vlm_manager.generate(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            # Save caption to .txt file
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(folder_path, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            results.append(f"[OK] {filename} -> {base_name}.txt")

        except Exception as e:
            results.append(f"[ERROR] {filename}: {str(e)}")

    progress(1.0, desc="Complete!")
    return f"Processed {total} images:\n\n" + "\n".join(results)


def create_ui():
    """Create the Gradio interface."""
    available_models = vlm_manager.get_available_models() if vlm_manager else ["Manager not initialized"]

    # Theme for Gradio 5.x
    global vlm_theme, vlm_css
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
    .gallery-item:first-child { border: 2px solid #4CAF50 !important; }
    .gallery-item:first-child:hover { border-color: #45a049 !important; }
    .green-btn {
        background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
        color: white !important;
        border: none !important;
    }
    .green-btn:hover {
        background: linear-gradient(to bottom right, #27ae60, #219651) !important;
    }
    .refresh-btn {
        max-width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .light-blue-btn {
        background: linear-gradient(to bottom right, #AEC6CF, #9AB8C4) !important;
        color: #333 !important;
        border: 1px solid #9AB8C4 !important;
    }
    .light-blue-btn:hover {
        background: linear-gradient(to bottom right, #9AB8C4, #8AA9B5) !important;
        border-color: #8AA9B5 !important;
    }
    """

    with gr.Blocks(title="Chromaforge VLM") as demo:
        with gr.Row():
            # Left column - Settings (shared across tabs)
            with gr.Column(scale=1):
                gr.Markdown("### Model Settings")

                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Select Model",
                    interactive=True,
                )

                refresh_models_btn = gr.Button("Refresh Model List", size="sm")

                # Backend selection (vLLM or transformers)
                backend_choices = ["auto", "vllm", "transformers"] if VLLM_AVAILABLE else ["transformers"]
                backend_dropdown = gr.Dropdown(
                    choices=backend_choices,
                    value="auto" if VLLM_AVAILABLE else "transformers",
                    label="Backend",
                    info="vLLM: faster inference, transformers: more compatible",
                    interactive=True,
                )
                backend_status = gr.Textbox(
                    label="Backend Status",
                    value=f"Current: {vlm_manager.backend if vlm_manager else 'not initialized'}",
                    interactive=False,
                )

                quantization_dropdown = gr.Dropdown(
                    choices=["none", "4bit", "8bit"],
                    value="none",
                    label="Quantization",
                    info="4-bit: ~4GB VRAM, 8-bit: ~8GB VRAM, none: ~16GB for 8B model",
                    interactive=True,
                )

                use_flash_attn = gr.Checkbox(
                    label="Use Flash Attention 2",
                    value=False,
                    info="Faster attention, requires flash-attn package",
                )

                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="primary")
                    unload_btn = gr.Button("Unload Model", variant="secondary")

                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded",
                    interactive=False,
                )

                refresh_btn = gr.Button("Refresh Memory Info")
                memory_info = gr.Textbox(
                    label="Memory Info",
                    value=vlm_manager.get_memory_info() if vlm_manager else "N/A",
                    interactive=False,
                )

                gr.Markdown("### System Prompt")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter a system prompt to guide the model's behavior...",
                    lines=4,
                    value="You are a helpful AI assistant that can understand and describe images and videos in detail.",
                )

                gr.Markdown("### Generation Settings")
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
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
                    label="Top P",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top K",
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                    label="Repetition Penalty",
                )

                gr.Markdown("### Video Settings")
                video_max_frames = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=8,
                    step=1,
                    label="Max Video Frames",
                    info="Number of frames to extract from videos",
                )

                gr.Markdown("### Memory Settings")
                vram_buffer = gr.Slider(
                    minimum=0,
                    maximum=32,
                    value=0,
                    step=1,
                    label="VRAM Buffer (GB)",
                    info="Reserve GPU memory during loading. Useful for large models with 8-bit quantization.",
                )
                auto_unload = gr.Checkbox(
                    label="Auto-unload after generation",
                    value=False,
                    info="Unload model after each response (saves VRAM but slower)",
                )

            # Right column - Tabs
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Chat Tab
                    with gr.TabItem("Chat"):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400,
                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                image_input = gr.Image(
                                    label="Upload Image (optional)",
                                    type="pil",
                                    height=150,
                                )
                            with gr.Column(scale=1):
                                video_input = gr.Video(
                                    label="Upload Video (optional)",
                                    height=150,
                                )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Message",
                                placeholder="Type your message here...",
                                lines=2,
                                scale=4,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        clear_btn = gr.Button("Clear Chat")

                    # Batch Caption Tab
                    with gr.TabItem("Batch Caption"):
                        gr.Markdown("Generate captions for all images in a folder. Outputs .txt files with matching names.")

                        batch_folder = gr.Textbox(
                            label="Folder Path",
                            placeholder="Enter the full path to folder containing images...",
                            lines=1,
                        )

                        batch_system_prompt = gr.Textbox(
                            label="System Prompt",
                            placeholder="System instructions for captioning...",
                            lines=3,
                            value="You are an image captioning assistant. Provide detailed, accurate descriptions suitable for training image generation models.",
                        )

                        batch_prompt = gr.Textbox(
                            label="Caption Prompt",
                            placeholder="Describe this image in detail.",
                            lines=2,
                            value="Describe this image in detail, including the subject, style, composition, colors, lighting, and any notable features.",
                        )

                        batch_start_btn = gr.Button("Start Batch Captioning", variant="primary", elem_classes=["green-btn"])

                        batch_output = gr.Textbox(
                            label="Output",
                            lines=15,
                            interactive=False,
                        )

        # Event handlers
        def refresh_models():
            """Refresh the list of available models."""
            if vlm_manager is None:
                return gr.update(choices=["Manager not initialized"])
            models = vlm_manager.get_available_models()
            return gr.update(choices=models, value=models[0] if models else None)

        refresh_models_btn.click(
            fn=refresh_models,
            outputs=[model_dropdown],
        )

        load_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, quantization_dropdown, use_flash_attn, vram_buffer],
            outputs=[model_status],
        )

        # Backend switching handler
        backend_dropdown.change(
            fn=switch_backend_handler,
            inputs=[backend_dropdown],
            outputs=[backend_status],
        )

        unload_btn.click(
            fn=unload_model_handler,
            outputs=[model_status],
        )

        refresh_btn.click(
            fn=get_memory_handler,
            outputs=[memory_info],
        )

        def send_message(msg, history, sys_prompt, img, vid, max_tok, temp, tp, tk, rep, vid_frames, auto_unl):
            if not msg.strip() and img is None and vid is None:
                return history, "", None, None, gr.update()

            new_history, status = chat_handler(
                msg, history, sys_prompt, img, vid,
                max_tok, temp, tp, tk, rep, vid_frames, auto_unl
            )
            # Update status only if auto-unload happened
            if status:
                return new_history, "", None, None, status
            return new_history, "", None, None, gr.update()

        send_btn.click(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, top_p, top_k, repetition_penalty, video_max_frames, auto_unload
            ],
            outputs=[chatbot, msg_input, image_input, video_input, model_status],
        )

        msg_input.submit(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, top_p, top_k, repetition_penalty, video_max_frames, auto_unload
            ],
            outputs=[chatbot, msg_input, image_input, video_input, model_status],
        )

        clear_btn.click(
            fn=clear_chat_handler,
            outputs=[chatbot],
        )

        batch_start_btn.click(
            fn=batch_caption_handler,
            inputs=[
                batch_folder, batch_prompt, batch_system_prompt,
                max_tokens, temperature, top_p, top_k, repetition_penalty
            ],
            outputs=[batch_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Chromaforge VLM Chat Interface")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7862,
        help="Port to run the server on (default: 7862)",
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
    parser.add_argument(
        "--lowvram",
        action="store_true",
        help="Enable low VRAM mode for smaller GPUs",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "vllm", "transformers"],
        default="auto",
        help="Backend for model inference (default: auto - uses vLLM if available)",
    )

    args = parser.parse_args()

    # Override host if --listen is specified
    host = "0.0.0.0" if args.listen else args.host

    print("=" * 60)
    print("Chromaforge VLM Chat Interface")
    print("=" * 60)
    print(f"Low VRAM mode: {'enabled' if args.lowvram else 'disabled'}")
    print(f"Backend: {args.backend}" + (" (vLLM available)" if VLLM_AVAILABLE else " (vLLM not available)"))
    print(f"Server: http://{host}:{args.port}")
    if args.listen:
        print("LAN access: enabled (listening on 0.0.0.0)")
    print("=" * 60)

    # Initialize the manager
    initialize_manager(low_vram=args.lowvram, backend=args.backend)

    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_name=host,
        server_port=args.port,
        share=args.share,
        pwa=False,
    )


if __name__ == "__main__":
    main()
