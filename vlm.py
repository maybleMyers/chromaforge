"""
VLM Chat Interface using llama-cpp-python Backend
A standalone GUI for interacting with Vision-Language Models via llama.cpp.

Requirements:
- pip install llama-cpp-python (with CUDA support)
- GGUF vision model + mmproj (clip) model

Installation with CUDA (Linux):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

Installation with CUDA (Windows prebuilt):
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
"""

import os
import gc
import re
import json
import base64
import argparse
import tempfile
import time
import subprocess
import signal
import threading
import requests
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Generator

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

# Import llama-cpp-python
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import (
        Llava15ChatHandler,
        Llava16ChatHandler,
    )
    LLAMA_CPP_AVAILABLE = True

    # Try to import Qwen3VLChatHandler first (for Qwen3-VL models), then fall back to Qwen25VLChatHandler
    Qwen3VLChatHandler = None
    Qwen25VLChatHandler = None
    QWEN3_VL_AVAILABLE = False
    QWEN_VL_AVAILABLE = False

    try:
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        QWEN3_VL_AVAILABLE = True
        print("Qwen3VLChatHandler: available")
    except ImportError:
        print("Note: Qwen3VLChatHandler not available. Install JamePeng's fork for Qwen3-VL support.")

    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        QWEN_VL_AVAILABLE = True
        print("Qwen25VLChatHandler: available")
    except ImportError:
        print("Note: Qwen25VLChatHandler not available.")

except ImportError:
    LLAMA_CPP_AVAILABLE = False
    QWEN_VL_AVAILABLE = False
    QWEN3_VL_AVAILABLE = False
    Llama = None
    Qwen25VLChatHandler = None
    Qwen3VLChatHandler = None
    print("Error: llama-cpp-python not installed.")
    print("Install with CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 data URL."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{b64_data}"


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


def find_gguf_models(models_dir: str) -> List[Dict[str, str]]:
    """
    Find GGUF models in a directory.
    Returns list of dicts with 'name', 'model_path', and optional 'mmproj_path'.
    Groups by folder - each subfolder is treated as one model.
    For split models, selects the first shard as the model_path.
    """
    models = []
    models_path = Path(models_dir)

    if not models_path.exists():
        return models

    # Track folders we've already processed
    processed_folders = set()

    # First, find all subdirectories containing .gguf files
    for gguf_file in models_path.rglob("*.gguf"):
        parent = gguf_file.parent

        # Skip if we've already processed this folder
        if parent in processed_folders:
            continue

        # Skip files directly in models_dir (no subfolder)
        if parent == models_path:
            # Handle loose files in root - treat each as its own model
            file_stem = gguf_file.stem

            # Skip mmproj/clip files as main models
            if any(x in file_stem.lower() for x in ["mmproj", "clip", "vision-encoder", "image-encoder"]):
                continue

            # Skip split model shards (not the first one)
            if "-0000" in file_stem and not file_stem.endswith("-00001-of"):
                # Check if this is a shard that's not the first
                import re
                shard_match = re.search(r'-(\d{5})-of-(\d{5})', file_stem)
                if shard_match and shard_match.group(1) != "00001":
                    continue

            models.append({
                "name": file_stem,
                "model_path": str(gguf_file),
                "mmproj_path": None,
            })
            continue

        # Mark this folder as processed
        processed_folders.add(parent)

        # Use folder name as display name
        name = parent.name

        # Find all gguf files in this folder
        all_ggufs = list(parent.glob("*.gguf"))

        # Separate mmproj/clip files from model files
        mmproj_path = None
        model_files = []

        mmproj_patterns = ["mmproj", "clip", "vision-encoder", "image-encoder"]

        for gf in all_ggufs:
            stem_lower = gf.stem.lower()
            if any(p in stem_lower for p in mmproj_patterns):
                if mmproj_path is None:
                    mmproj_path = str(gf)
            else:
                model_files.append(gf)

        if not model_files:
            continue

        # Sort model files to get consistent ordering
        # For split models, this ensures we get the first shard
        model_files.sort(key=lambda x: x.name)

        # Use the first model file (or first shard for split models)
        model_path = str(model_files[0])

        models.append({
            "name": name,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
        })

    return sorted(models, key=lambda x: x["name"])


class LlamaCppVLM:
    """Manages VLM inference via llama-cpp-python or native llama-server subprocess."""

    # Default path to llama-server executable
    DEFAULT_LLAMA_SERVER_PATH = "llama-server"

    def __init__(self, models_dir: str = "models/LLM"):
        """Initialize llama.cpp VLM Manager."""
        self.models_dir = models_dir
        self.model: Optional[Llama] = None
        self.current_model_path: Optional[str] = None
        self.current_mmproj_path: Optional[str] = None
        self.chat_handler = None
        self.is_text_only_model = False  # Flag for text-only models like GPT-OSS
        self.model_type: Optional[str] = None  # Track model type (e.g., "gpt-oss", "qwen", "llava")

        # llama-server subprocess mode
        self.server_process: Optional[subprocess.Popen] = None
        self.server_url: Optional[str] = None
        self.use_server_backend = False
        self.llama_server_path = self.DEFAULT_LLAMA_SERVER_PATH

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available GGUF models."""
        return find_gguf_models(self.models_dir)

    def get_model_names(self) -> List[str]:
        """Get list of model names for dropdown."""
        models = self.get_available_models()
        if not models:
            return ["No GGUF models found"]
        return [m["name"] for m in models]

    def load_model(
        self,
        model_name: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        tensor_split: Optional[str] = None,
        flash_attn: bool = False,
        main_gpu: int = 0,
        type_k: Optional[str] = None,
        type_v: Optional[str] = None,
        progress=gr.Progress(),
    ) -> str:
        """
        Load a GGUF model.

        Args:
            model_name: Name of the model to load
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context length
            tensor_split: Comma-separated GPU memory split ratios (e.g., "0.5,0.5" for 2 GPUs)
            flash_attn: Enable Flash Attention for faster inference
            main_gpu: Index of the main GPU for small tensors (default: 0)
            type_k: KV cache quantization type for keys (e.g., "q8_0", "q4_0", "f16")
            type_v: KV cache quantization type for values (e.g., "q8_0", "q4_0", "f16")
            progress: Gradio progress callback

        Returns:
            Status message
        """
        if not LLAMA_CPP_AVAILABLE:
            return "Error: llama-cpp-python not installed"

        # Find the model
        models = self.get_available_models()
        model_info = next((m for m in models if m["name"] == model_name), None)

        if model_info is None:
            return f"Error: Model '{model_name}' not found"

        model_path = model_info["model_path"]
        mmproj_path = model_info.get("mmproj_path")

        # Unload current model if any
        if self.model is not None:
            self.unload_model()

        try:
            progress(0.1, desc="Initializing...")

            # Detect model type from name/path
            model_name_lower = model_name.lower()
            model_path_lower = model_path.lower()
            is_qwen_vl = any(x in model_name_lower or x in model_path_lower for x in ["qwen", "qwen2-vl", "qwen3-vl", "qwen2.5-vl"])
            is_llava = any(x in model_name_lower or x in model_path_lower for x in ["llava", "llava-v1"])

            # Detect GPT-OSS models (text-only with Harmony format)
            is_gpt_oss = any(x in model_name_lower or x in model_path_lower for x in ["gpt-oss", "gptoss", "gpt_oss", "huihui-gpt-oss"])

            # More specific detection
            is_qwen3_specific = any(x in model_name_lower or x in model_path_lower for x in ["qwen3"])
            is_qwen25_specific = any(x in model_name_lower or x in model_path_lower for x in ["qwen2.5", "qwen25"])

            print(f"[llama.cpp] Model type detection: GPT-OSS={is_gpt_oss}, Qwen-VL={is_qwen_vl}, Qwen3={is_qwen3_specific}, Qwen2.5={is_qwen25_specific}, LLaVA={is_llava}")

            # Set model type tracking
            self.is_text_only_model = is_gpt_oss or (mmproj_path is None)
            if is_gpt_oss:
                self.model_type = "gpt-oss"
            elif is_qwen_vl:
                self.model_type = "qwen-vl"
            elif is_llava:
                self.model_type = "llava"
            else:
                self.model_type = "generic"

            if is_gpt_oss:
                print("[llama.cpp] GPT-OSS model detected - using text-only mode with Harmony format")

            # Set up chat handler for vision models
            self.chat_handler = None
            if mmproj_path and os.path.exists(mmproj_path):
                progress(0.2, desc="Loading vision encoder...")
                print(f"[llama.cpp] Loading mmproj from: {mmproj_path}")

                # Select chat handler based on model type
                # Check for Qwen3-VL first (uses Qwen3VLChatHandler)
                is_qwen3_vl = any(x in model_name_lower or x in model_path_lower for x in ["qwen3-vl", "qwen3vl", "qwen3"])
                is_qwen25_vl = any(x in model_name_lower or x in model_path_lower for x in ["qwen2.5-vl", "qwen25vl", "qwen2-vl"])

                if is_qwen3_vl and QWEN3_VL_AVAILABLE and Qwen3VLChatHandler is not None:
                    print("[llama.cpp] Using Qwen3VLChatHandler (for Qwen3-VL)")
                    try:
                        self.chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, verbose=False)
                    except Exception as e:
                        print(f"Warning: Qwen3VLChatHandler failed: {e}")
                        self.chat_handler = None
                elif is_qwen25_vl and QWEN_VL_AVAILABLE and Qwen25VLChatHandler is not None:
                    print("[llama.cpp] Using Qwen25VLChatHandler (for Qwen2.5-VL)")
                    try:
                        self.chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)
                    except Exception as e:
                        print(f"Warning: Qwen25VLChatHandler failed: {e}")
                        self.chat_handler = None
                elif is_llava:
                    print("[llama.cpp] Using LLaVA chat handler")
                    try:
                        self.chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path, verbose=False)
                    except Exception:
                        try:
                            self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
                        except Exception as e:
                            print(f"Warning: LLaVA handler failed: {e}")
                            self.chat_handler = None
                else:
                    # Try handlers in order of likelihood based on model name
                    print("[llama.cpp] Trying chat handlers in order...")
                    handlers_to_try = []

                    # If it looks like a Qwen3 model, try Qwen3VL first
                    if is_qwen3_specific and QWEN3_VL_AVAILABLE and Qwen3VLChatHandler is not None:
                        handlers_to_try.append(("Qwen3VL", Qwen3VLChatHandler))

                    # Then try other handlers
                    if QWEN_VL_AVAILABLE and Qwen25VLChatHandler is not None and ("Qwen25VL", Qwen25VLChatHandler) not in handlers_to_try:
                        handlers_to_try.append(("Qwen25VL", Qwen25VLChatHandler))
                    if QWEN3_VL_AVAILABLE and Qwen3VLChatHandler is not None and ("Qwen3VL", Qwen3VLChatHandler) not in handlers_to_try:
                        handlers_to_try.append(("Qwen3VL", Qwen3VLChatHandler))

                    # Try Llama3VisionAlphaChatHandler for Qwen models
                    try:
                        from llama_cpp.llama_chat_format import Llama3VisionAlphaChatHandler
                        if is_qwen_vl:
                            handlers_to_try.insert(0, ("Llama3VisionAlpha", Llama3VisionAlphaChatHandler))
                        else:
                            handlers_to_try.append(("Llama3VisionAlpha", Llama3VisionAlphaChatHandler))
                    except ImportError:
                        pass

                    handlers_to_try.extend([
                        ("Llava16", Llava16ChatHandler),
                        ("Llava15", Llava15ChatHandler),
                    ])

                    for handler_name, handler_class in handlers_to_try:
                        try:
                            print(f"[llama.cpp] Trying {handler_name}ChatHandler...")
                            self.chat_handler = handler_class(clip_model_path=mmproj_path, verbose=False)
                            print(f"[llama.cpp] {handler_name}ChatHandler loaded successfully")
                            break
                        except Exception as e:
                            print(f"[llama.cpp] {handler_name}ChatHandler failed: {e}")
                            continue

            progress(0.3, desc=f"Loading {model_name}...")
            print(f"[llama.cpp] Loading model from: {model_path}")
            print(f"[llama.cpp] GPU layers: {n_gpu_layers}, Context: {n_ctx}, Main GPU: {main_gpu}")

            # Parse tensor_split if provided
            tensor_split_list = None
            if tensor_split and tensor_split.strip():
                try:
                    tensor_split_list = [float(x.strip()) for x in tensor_split.split(",")]
                    print(f"[llama.cpp] Tensor split: {tensor_split_list}")
                except ValueError:
                    print(f"[llama.cpp] Warning: Invalid tensor_split format '{tensor_split}', ignoring")
                    tensor_split_list = None

            if flash_attn:
                print("[llama.cpp] Flash Attention: enabled")

            if type_k:
                print(f"[llama.cpp] KV cache key type: {type_k}")
            if type_v:
                print(f"[llama.cpp] KV cache value type: {type_v}")

            # Build kwargs for Llama constructor
            llama_kwargs = {
                "model_path": model_path,
                "chat_handler": self.chat_handler,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "main_gpu": main_gpu,
                "verbose": True,
            }

            # Add optional parameters
            if tensor_split_list:
                llama_kwargs["tensor_split"] = tensor_split_list

            if flash_attn:
                llama_kwargs["flash_attn"] = True

            # Add KV cache type parameters if specified
            if type_k:
                llama_kwargs["type_k"] = type_k
            if type_v:
                llama_kwargs["type_v"] = type_v

            # Load the model
            self.model = Llama(**llama_kwargs)

            self.current_model_path = model_path
            self.current_mmproj_path = mmproj_path

            progress(1.0, desc="Model loaded!")

            vision_status = "with vision" if self.chat_handler else "text-only"
            return f"Loaded: {model_name} ({vision_status})"

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.model = None
            self.chat_handler = None
            return f"Error loading model: {str(e)}"

    def unload_model(self) -> str:
        """Unload the current model."""
        if self.model is None:
            return "No model loaded"

        model_name = Path(self.current_model_path).stem if self.current_model_path else "model"

        try:
            # Kill server subprocess if running
            if self.server_process is not None:
                print("[llama.cpp] Stopping llama-server subprocess...")
                try:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                self.server_process = None
                self.server_url = None
                self.use_server_backend = False

            del self.model
            del self.chat_handler
            self.model = None
            self.chat_handler = None
            self.current_model_path = None
            self.current_mmproj_path = None
            self.is_text_only_model = False
            self.model_type = None

            # Force garbage collection
            gc.collect()

            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            return f"Unloaded: {model_name}"
        except Exception as e:
            return f"Error unloading: {str(e)}"

    def get_status(self) -> str:
        """Get current status."""
        if self.server_process is not None and self.server_url:
            model_name = Path(self.current_model_path).stem if self.current_model_path else "Unknown"
            return f"Server: {model_name} @ {self.server_url}"

        if self.model is None:
            return "No model loaded"

        model_name = Path(self.current_model_path).stem if self.current_model_path else "Unknown"
        vision_status = "vision" if self.chat_handler else "text-only"
        model_type_str = f", {self.model_type}" if self.model_type and self.model_type != "generic" else ""
        return f"Loaded: {model_name} ({vision_status}{model_type_str})"

    def load_model_server(
        self,
        model_name: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        tensor_split: Optional[str] = None,
        flash_attn: bool = False,
        main_gpu: int = 0,
        type_k: Optional[str] = None,
        type_v: Optional[str] = None,
        override_tensor: Optional[str] = None,
        server_port: int = 8080,
        extra_args: Optional[str] = None,
        progress=gr.Progress(),
    ) -> str:
        """
        Load a GGUF model using native llama-server subprocess.

        This allows using advanced options like --override-tensor for MoE optimization.
        """
        # Find the model
        models = self.get_available_models()
        model_info = next((m for m in models if m["name"] == model_name), None)

        if model_info is None:
            return f"Error: Model '{model_name}' not found"

        model_path = model_info["model_path"]

        # Unload current model if any
        if self.model is not None or self.server_process is not None:
            self.unload_model()

        progress(0.1, desc="Building server command...")

        # Build llama-server command
        cmd = [
            self.llama_server_path,
            "-m", model_path,
            "--port", str(server_port),
            "--host", "127.0.0.1",
            "-ngl", str(n_gpu_layers),
            "-c", str(n_ctx),
            "--main-gpu", str(main_gpu),
        ]

        # Add tensor split if provided
        if tensor_split and tensor_split.strip():
            cmd.extend(["--tensor-split", tensor_split.strip()])

        # Add flash attention
        if flash_attn:
            cmd.append("-fa")

        # Add KV cache types
        if type_k:
            cmd.extend(["--cache-type-k", type_k])
        if type_v:
            cmd.extend(["--cache-type-v", type_v])

        # Add override tensor (the key MoE optimization!)
        if override_tensor and override_tensor.strip():
            # Support multiple patterns separated by semicolons
            patterns = [p.strip() for p in override_tensor.split(";") if p.strip()]
            for pattern in patterns:
                cmd.extend(["-ot", pattern])

        # Add any extra arguments
        if extra_args and extra_args.strip():
            cmd.extend(extra_args.strip().split())

        print(f"[llama-server] Command: {' '.join(cmd)}")
        progress(0.2, desc="Starting llama-server...")

        try:
            # Start the server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Start a thread to read and print server output
            def read_output():
                if self.server_process and self.server_process.stdout:
                    for line in self.server_process.stdout:
                        print(f"[llama-server] {line.rstrip()}")

            output_thread = threading.Thread(target=read_output, daemon=True)
            output_thread.start()

            # Wait for server to be ready (poll health endpoint)
            self.server_url = f"http://127.0.0.1:{server_port}"
            health_url = f"{self.server_url}/health"

            progress(0.3, desc="Waiting for server to load model...")

            max_wait = 300  # 5 minutes max wait
            start_time = time.time()
            server_ready = False

            while time.time() - start_time < max_wait:
                # Check if process died
                if self.server_process.poll() is not None:
                    return f"Error: llama-server exited with code {self.server_process.returncode}"

                try:
                    resp = requests.get(health_url, timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data.get("status", "")
                        print(f"[llama-server] Health check: {resp.status_code}, status='{status}'")
                        if status == "ok":
                            server_ready = True
                            print("[llama-server] Server is ready!")
                            break
                        elif status == "loading model":
                            progress(0.5, desc="Server loading model...")
                    else:
                        print(f"[llama-server] Health check: {resp.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[llama-server] Health check failed: {e}")

                time.sleep(1)

            if not server_ready:
                self.server_process.terminate()
                self.server_process = None
                return "Error: Server failed to become ready within timeout"

            self.current_model_path = model_path
            self.use_server_backend = True
            self.is_text_only_model = True  # Server mode is text-only for now

            print(f"[llama-server] Flags set: use_server_backend={self.use_server_backend}, server_url={self.server_url}, server_process={self.server_process is not None}")

            progress(1.0, desc="Server ready!")
            return f"Server started: {model_name} @ {self.server_url}"

        except FileNotFoundError:
            return f"Error: llama-server not found at '{self.llama_server_path}'. Set the correct path."
        except Exception as e:
            if self.server_process:
                self.server_process.terminate()
                self.server_process = None
            return f"Error starting server: {str(e)}"

    def generate_via_api(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ):
        """Generate response via llama-server OpenAI-compatible API."""
        if not self.server_url:
            return "Error: Server not running"

        api_url = f"{self.server_url}/v1/chat/completions"

        # Convert messages to simple text format for API
        api_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract text from complex content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)

            api_messages.append({"role": role, "content": str(content)})

        payload = {
            "messages": api_messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.001,
            "stream": stream,
        }

        start_time = time.perf_counter()

        if stream:
            # Streaming mode
            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    timeout=300,
                )
                response.raise_for_status()

                accumulated = ""
                token_count = 0

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    accumulated += content
                                    token_count += 1
                                    elapsed = time.perf_counter() - start_time
                                    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                                    yield accumulated, accumulated, f"{tokens_per_sec:.1f} tok/s"
                            except json.JSONDecodeError:
                                pass

                end_time = time.perf_counter()
                generation_time = end_time - start_time
                final_speed = token_count / generation_time if generation_time > 0 else 0
                print(f"[llama-server] Streamed {token_count} tokens in {generation_time:.2f}s ({final_speed:.1f} tok/s)")

            except Exception as e:
                yield f"Error: {str(e)}", f"Error: {str(e)}", "0 tok/s"
        else:
            # Non-streaming mode
            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
                result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return result
            except Exception as e:
                return f"Error: {str(e)}"

    def generate(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        video_max_frames: int = 8,
        stream: bool = False,
    ):
        """
        Generate a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            images: Optional list of PIL Images
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            video_max_frames: Max frames for video processing
            stream: If True, yields partial responses as a generator

        Returns:
            Generated response string, or generator if stream=True
        """
        # Debug: show current state
        print(f"[vlm.py] generate() called: use_server_backend={self.use_server_backend}, server_process={self.server_process is not None}, model={self.model is not None}")

        # Route to API if using server backend
        if self.use_server_backend and self.server_process is not None:
            print(f"[vlm.py] Using server backend at {self.server_url}")
            return self.generate_via_api(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=stream,
            )

        if self.model is None and not self.use_server_backend:
            return "Error: No model loaded. Please load a model first."

        if self.model is None and self.use_server_backend:
            # Server mode but server_process is None - server may have crashed
            return f"Error: Server backend enabled but server not running. server_process={self.server_process}, url={self.server_url}"

        try:
            # Build messages in llama.cpp format
            llama_messages = []

            # For text-only models (like GPT-OSS), we need to convert all content to plain text
            # The chat templates for these models expect string content, not list content
            use_text_only = self.is_text_only_model

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    # System messages should always be strings
                    if isinstance(content, list):
                        # Extract text from list content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        content = " ".join(text_parts) if text_parts else str(content)
                    llama_messages.append({"role": "system", "content": str(content)})

                elif role == "user":
                    if isinstance(content, list):
                        if use_text_only:
                            # For text-only models, extract only text content
                            text_parts = []
                            has_images = False
                            has_video = False
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))
                                    elif item.get("type") == "image":
                                        has_images = True
                                    elif item.get("type") == "video":
                                        has_video = True
                                elif isinstance(item, str):
                                    text_parts.append(item)

                            # Build text-only content
                            final_text = " ".join(text_parts) if text_parts else ""
                            if has_images and not final_text:
                                final_text = "Please respond to this message."
                            elif has_images:
                                final_text = f"[Note: Images were provided but this model is text-only] {final_text}"
                            if has_video:
                                final_text = f"[Note: Video was provided but this model is text-only] {final_text}"

                            llama_messages.append({"role": "user", "content": final_text if final_text else "Hello"})
                        else:
                            # Handle multimodal content for vision models
                            parts = []
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        parts.append({
                                            "type": "text",
                                            "text": item.get("text", "")
                                        })
                                    elif item.get("type") == "image" and "image" in item:
                                        img = item["image"]
                                        if isinstance(img, Image.Image):
                                            b64_url = image_to_base64(img)
                                            parts.append({
                                                "type": "image_url",
                                                "image_url": {"url": b64_url}
                                            })
                                    elif item.get("type") == "video" and "video" in item:
                                        video_path = item["video"]
                                        if isinstance(video_path, str) and os.path.exists(video_path):
                                            try:
                                                frames = extract_video_frames(video_path, max_frames=video_max_frames)
                                                for frame in frames:
                                                    b64_url = image_to_base64(frame)
                                                    parts.append({
                                                        "type": "image_url",
                                                        "image_url": {"url": b64_url}
                                                    })
                                                parts.append({
                                                    "type": "text",
                                                    "text": f"[Video with {len(frames)} frames]"
                                                })
                                            except Exception as e:
                                                parts.append({
                                                    "type": "text",
                                                    "text": f"[Video error: {e}]"
                                                })

                            if parts:
                                llama_messages.append({"role": "user", "content": parts})
                            else:
                                llama_messages.append({"role": "user", "content": "Describe this."})
                    else:
                        llama_messages.append({"role": "user", "content": str(content)})

                elif role == "assistant":
                    llama_messages.append({"role": "assistant", "content": str(content)})

            # Add any direct images (only for vision models)
            if images and not use_text_only:
                image_parts = []
                for img in images:
                    b64_url = image_to_base64(img)
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": b64_url}
                    })
                image_parts.append({"type": "text", "text": "Describe this image."})
                llama_messages.append({"role": "user", "content": image_parts})
            elif images and use_text_only:
                # For text-only models, just add a note about images
                llama_messages.append({"role": "user", "content": "[Note: Images were provided but this model is text-only. Please respond to the previous message.]"})

            # Debug logging for GPT-OSS and other text-only models
            if self.model_type == "gpt-oss":
                print(f"[llama.cpp] GPT-OSS: Building request with {len(llama_messages)} messages")
                for i, msg in enumerate(llama_messages):
                    content_preview = str(msg.get('content', ''))[:100]
                    print(f"[llama.cpp]   [{i}] {msg['role']}: {content_preview}...")

            # Generate response
            start_time = time.perf_counter()

            if stream:
                # Streaming mode - yield partial responses
                response_stream = self.model.create_chat_completion(
                    messages=llama_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.001,
                    stream=True,
                )

                accumulated = ""
                token_count = 0
                for chunk in response_stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated += content
                        token_count += 1
                        # Calculate current speed
                        elapsed = time.perf_counter() - start_time
                        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                        # Clean up thinking tags and GPT-OSS Harmony format tags for display
                        display_text = accumulated
                        raw_for_thinking = accumulated  # Keep a version with thinking preserved

                        if "</think>" in display_text:
                            display_text = display_text.split("</think>")[-1].strip()

                        # Clean up GPT-OSS Harmony format tags if present
                        if self.model_type == "gpt-oss":
                            # For display_text (no thinking): extract only the final channel content
                            # First, try to get just the final channel
                            final_match = re.search(r'<\|channel\|>final<\|message\|>(.*?)(?:<\|(?:end|return)\|>|$)', display_text, re.DOTALL)
                            if final_match:
                                display_text = final_match.group(1).strip()
                            else:
                                # If no final channel yet, just clean up any tags
                                display_text = re.sub(r'<\|channel\|>(analysis|commentary|final)<\|message\|>', '', display_text)
                                display_text = re.sub(r'<\|(start|end|return|call)\|>', '', display_text)
                                display_text = display_text.strip()

                            # For raw_for_thinking (with thinking): make the analysis channel readable
                            # Convert channel markers to readable labels
                            raw_for_thinking = re.sub(r'<\|channel\|>analysis<\|message\|>', '\n[Thinking]\n', raw_for_thinking)
                            raw_for_thinking = re.sub(r'<\|channel\|>final<\|message\|>', '\n[Response]\n', raw_for_thinking)
                            raw_for_thinking = re.sub(r'<\|channel\|>commentary<\|message\|>', '\n[Commentary]\n', raw_for_thinking)
                            raw_for_thinking = re.sub(r'<\|(start|end|return|call)\|>', '', raw_for_thinking)
                            raw_for_thinking = raw_for_thinking.strip()

                        # Yield display_text, raw_text (with thinking formatted), and stats
                        yield display_text, raw_for_thinking, f"{tokens_per_sec:.1f} tok/s"

                end_time = time.perf_counter()
                generation_time = end_time - start_time
                final_speed = token_count / generation_time if generation_time > 0 else 0
                print(f"[llama.cpp] Streamed {token_count} tokens in {generation_time:.2f}s ({final_speed:.1f} tok/s)")

            else:
                # Non-streaming mode
                response = self.model.create_chat_completion(
                    messages=llama_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.001,
                )

                end_time = time.perf_counter()
                generation_time = end_time - start_time
                print(f"[llama.cpp] Generated response in {generation_time:.2f}s")

                # Extract response content
                result = response["choices"][0]["message"]["content"]

                # Clean up thinking tags if present
                if "</think>" in result:
                    result = result.split("</think>")[-1].strip()

                # Clean up GPT-OSS Harmony format tags if present
                if self.model_type == "gpt-oss":
                    # For non-streaming, extract only the final channel content
                    final_match = re.search(r'<\|channel\|>final<\|message\|>(.*?)(?:<\|(?:end|return)\|>|$)', result, re.DOTALL)
                    if final_match:
                        result = final_match.group(1).strip()
                    else:
                        # If no final channel, just clean up any tags
                        result = re.sub(r'<\|channel\|>(analysis|commentary|final)<\|message\|>', '', result)
                        result = re.sub(r'<\|(start|end|return|call)\|>', '', result)
                        result = result.strip()

                return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during generation: {str(e)}"


# Global manager instance
vlm_manager: Optional[LlamaCppVLM] = None
# Global stop flag for generation
stop_generation: bool = False


def initialize_manager(models_dir: str = "models/LLM"):
    """Initialize the global VLM manager."""
    global vlm_manager
    vlm_manager = LlamaCppVLM(models_dir)


def refresh_models_handler():
    """Refresh the list of available models."""
    if vlm_manager is None:
        return gr.update(choices=["Manager not initialized"])
    models = vlm_manager.get_model_names()
    return gr.update(choices=models, value=models[0] if models else None)


def load_model_handler(
    model_name: str,
    n_gpu_layers: int,
    n_ctx: int,
    tensor_split: str,
    flash_attn: bool,
    main_gpu: int,
    kv_cache_type: str,
    progress=gr.Progress()
):
    """Handle model loading."""
    if vlm_manager is None:
        return "Manager not initialized"

    # Parse KV cache type (same type for both k and v)
    type_k = kv_cache_type if kv_cache_type and kv_cache_type != "f16" else None
    type_v = kv_cache_type if kv_cache_type and kv_cache_type != "f16" else None

    return vlm_manager.load_model(
        model_name,
        n_gpu_layers,
        n_ctx,
        tensor_split,
        flash_attn,
        main_gpu,
        type_k,
        type_v,
        progress
    )


def load_model_server_handler(
    model_name: str,
    n_gpu_layers: int,
    n_ctx: int,
    tensor_split: str,
    flash_attn: bool,
    main_gpu: int,
    kv_cache_type: str,
    override_tensor: str,
    server_port: int,
    llama_server_path: str,
    progress=gr.Progress()
):
    """Handle model loading via llama-server subprocess."""
    if vlm_manager is None:
        return "Manager not initialized"

    # Set server path if provided
    if llama_server_path and llama_server_path.strip():
        vlm_manager.llama_server_path = llama_server_path.strip()

    # Parse KV cache type
    type_k = kv_cache_type if kv_cache_type and kv_cache_type != "f16" else None
    type_v = kv_cache_type if kv_cache_type and kv_cache_type != "f16" else None

    return vlm_manager.load_model_server(
        model_name=model_name,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        tensor_split=tensor_split,
        flash_attn=flash_attn,
        main_gpu=main_gpu,
        type_k=type_k,
        type_v=type_v,
        override_tensor=override_tensor,
        server_port=server_port,
        progress=progress,
    )


def unload_model_handler():
    """Handle model unloading."""
    if vlm_manager is None:
        return "Manager not initialized"
    return vlm_manager.unload_model()


def status_handler():
    """Handle status request."""
    if vlm_manager is None:
        return "Manager not initialized"
    return vlm_manager.get_status()


def chat_handler(
    message: str,
    history: List[Dict[str, Any]],
    system_prompt: str,
    image1,
    image2,
    image3,
    image4,
    video,
    max_tokens: int,
    temperature: float,
    video_max_frames: int = 8,
    show_thinking: bool = False,
):
    """Handle chat messages from UI with streaming support."""
    # Check if any model is loaded (either local or server mode)
    model_ready = (
        vlm_manager is not None and
        (vlm_manager.model is not None or vlm_manager.use_server_backend)
    )
    if not model_ready:
        error_history = list(history)
        error_history.append({"role": "user", "content": message})
        error_history.append({"role": "assistant", "content": "Error: No model loaded. Please load a model first."})
        yield error_history, "", ""
        return

    # Build messages list for the model
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
                # Extract text content for the model
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    # Extract text from multimodal content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        messages.append({"role": role, "content": " ".join(text_parts)})

    # Build current message content for model
    model_content = []

    # Add all provided images
    images = [img for img in [image1, image2, image3, image4] if img is not None]
    for img in images:
        model_content.append({"type": "image", "image": img})

    if video is not None:
        model_content.append({"type": "video", "video": video})

    if message.strip():
        model_content.append({"type": "text", "text": message})
    elif not model_content:
        model_content.append({"type": "text", "text": "Describe this image."})

    messages.append({"role": "user", "content": model_content})

    # Build initial display content for chatbot
    new_history = list(history)

    if images:
        display_text = message if message else f"Describe {'these images' if len(images) > 1 else 'this image'}"
        new_history.append({"role": "user", "content": display_text})

        # Add each image to chat display
        for idx, img in enumerate(images):
            # Resize image to 150px height for chat display
            orig_width, orig_height = img.size
            new_height = 150
            new_width = int(orig_width * (new_height / orig_height))
            display_image = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save resized image to temp file for display
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"vlm_chat_{int(time.time())}_{idx}_{id(img)}.png")
            display_image.save(temp_path)

            # Use dict with "path" key for file content in chatbot
            new_history.append({"role": "user", "content": {"path": temp_path}})

    elif video is not None:
        display_text = message if message else "Describe this video"
        new_history.append({"role": "user", "content": display_text})
        # Use dict with "path" key for file content in chatbot
        new_history.append({"role": "user", "content": {"path": video}})

    else:
        new_history.append({"role": "user", "content": message})

    # Add empty assistant message that we'll stream into
    new_history.append({"role": "assistant", "content": ""})

    # Reset stop flag before starting generation
    global stop_generation
    stop_generation = False

    # Stream the response
    stats = ""
    for display_text, raw_text, stats in vlm_manager.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        video_max_frames=video_max_frames,
        stream=True,
    ):
        # Check stop flag
        if stop_generation:
            new_history[-1]["content"] += "\n\n[Generation stopped]"
            stop_generation = False
            yield new_history, "", stats
            return

        # Show thinking if enabled, otherwise show cleaned text
        if show_thinking:
            new_history[-1]["content"] = raw_text
        else:
            new_history[-1]["content"] = display_text
        yield new_history, "", stats


def clear_chat_handler():
    """Clear chat history."""
    return []


def stop_generation_handler():
    """Set the stop flag to interrupt generation."""
    global stop_generation
    stop_generation = True
    return


def batch_caption_handler(
    folder_path: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
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
    .red-btn {
        background: linear-gradient(to bottom right, #e74c3c, #c0392b) !important;
        color: white !important;
        border: none !important;
    }
    .red-btn:hover {
        background: linear-gradient(to bottom right, #c0392b, #a93226) !important;
    }
    .resizable-chatbot {
        resize: vertical;
        overflow: auto;
        min-height: 200px;
        max-height: 90vh;
    }
    """

    # Get initial model list
    initial_models = vlm_manager.get_model_names() if vlm_manager else ["Initialize manager first"]

    with gr.Blocks(title="Chromaforge VLM (llama.cpp)", theme=vlm_theme, css=vlm_css) as demo:

        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("Chat"):
                # Chat interface at top - full width, user-resizable
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
                    elem_classes=["resizable-chatbot"],
                )

                # Media inputs row - 4 image inputs
                with gr.Row():
                    image_input_1 = gr.Image(
                        label="Image 1",
                        type="pil",
                        height=150,
                    )
                    image_input_2 = gr.Image(
                        label="Image 2",
                        type="pil",
                        height=150,
                    )
                    image_input_3 = gr.Image(
                        label="Image 3",
                        type="pil",
                        height=150,
                    )
                    image_input_4 = gr.Image(
                        label="Image 4",
                        type="pil",
                        height=150,
                    )
                    video_input = gr.Video(
                        label="Upload Video (optional)",
                        height=150,
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

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", scale=1)
                    stop_btn = gr.Button("Stop", variant="stop", scale=1, elem_classes=["red-btn"])
                    stats_display = gr.Textbox(
                        label="Speed",
                        value="",
                        interactive=False,
                        scale=2,
                    )

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
                    n_gpu_layers = gr.Slider(
                        minimum=-1,
                        maximum=100,
                        value=-1,
                        step=1,
                        label="GPU Layers (-1 = all)",
                        info="Layers to offload to GPU",
                    )

                with gr.Column(scale=1):
                    n_ctx = gr.Slider(
                        minimum=512,
                        maximum=200000,
                        value=32768,
                        step=512,
                        label="Context Length",
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    backend_type = gr.Dropdown(
                        label="Backend",
                        choices=["llama-cpp-python", "llama-server"],
                        value="llama-cpp-python",
                        info="llama-server enables MoE --override-tensor",
                    )
                with gr.Column(scale=2):
                    tensor_split = gr.Textbox(
                        label="Tensor Split (Multi-GPU)",
                        placeholder="e.g., 2,1 for 48GB+24GB GPUs (ratio-based distribution)",
                        value="",
                        info="Comma-separated ratios for distributing layers across GPUs",
                    )
                with gr.Column(scale=1):
                    main_gpu = gr.Slider(
                        minimum=0,
                        maximum=7,
                        value=0,
                        step=1,
                        label="Main GPU",
                        info="GPU for small tensors/scratch buffer",
                    )
                with gr.Column(scale=1):
                    kv_cache_type = gr.Dropdown(
                        label="KV Cache Type",
                        choices=["f16", "q8_0", "q4_0"],
                        value="q8_0",
                        info="Quantize KV cache to save VRAM",
                    )
                with gr.Column(scale=1):
                    flash_attn = gr.Checkbox(
                        label="Flash Attention",
                        value=True,
                        info="Faster attention (requires layers on GPU)",
                    )

            # Server-mode specific options
            with gr.Row(visible=False) as server_options_row:
                with gr.Column(scale=3):
                    override_tensor = gr.Textbox(
                        label="Override Tensor (-ot)",
                        placeholder=r"\.ffn_.*_exps\.weight=CPU",
                        value=r"\.ffn_.*_exps\.weight=CPU",
                        info="MoE optimization: offload expert FFN to CPU. Use ; for multiple patterns.",
                    )
                with gr.Column(scale=1):
                    server_port = gr.Number(
                        label="Server Port",
                        value=8080,
                        precision=0,
                        info="Port for llama-server",
                    )
                with gr.Column(scale=2):
                    llama_server_path = gr.Textbox(
                        label="llama-server Path",
                        placeholder="llama-server",
                        value="llama-server",
                        info="Path to llama-server executable",
                    )

            with gr.Row():
                load_model_btn = gr.Button("Load Model", variant="primary")
                unload_model_btn = gr.Button("Unload", variant="secondary")
                status_display = gr.Textbox(
                    label="Status",
                    value="No model loaded",
                    interactive=False,
                    scale=3,
                )

            # Toggle server options visibility based on backend selection
            def toggle_server_options(backend):
                return gr.update(visible=(backend == "llama-server"))

            backend_type.change(
                fn=toggle_server_options,
                inputs=[backend_type],
                outputs=[server_options_row],
            )

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter a system prompt to guide the model's behavior...",
                    lines=2,
                    value="You are a helpful AI assistant that can understand and describe images and videos in detail.",
                    scale=3,
                )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=262048,
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
                video_max_frames = gr.Slider(
                    minimum=1,
                    maximum=201,
                    value=8,
                    step=1,
                    label="Max Video Frames",
                    info="Frames to extract from videos",
                )

            with gr.Row():
                show_thinking = gr.Checkbox(
                    label="Show Thinking",
                    value=True,
                )

        # Event handlers
        refresh_models_btn.click(
            fn=refresh_models_handler,
            outputs=[model_dropdown],
        )

        def load_model_dispatcher(
            backend, model_name, n_gpu_layers, n_ctx, tensor_split, flash_attn,
            main_gpu, kv_cache_type, override_tensor, server_port, llama_server_path,
            progress=gr.Progress()
        ):
            """Route to appropriate load handler based on backend selection."""
            if backend == "llama-server":
                return load_model_server_handler(
                    model_name, n_gpu_layers, n_ctx, tensor_split, flash_attn,
                    main_gpu, kv_cache_type, override_tensor, int(server_port),
                    llama_server_path, progress
                )
            else:
                return load_model_handler(
                    model_name, n_gpu_layers, n_ctx, tensor_split, flash_attn,
                    main_gpu, kv_cache_type, progress
                )

        load_model_btn.click(
            fn=load_model_dispatcher,
            inputs=[
                backend_type, model_dropdown, n_gpu_layers, n_ctx, tensor_split,
                flash_attn, main_gpu, kv_cache_type, override_tensor, server_port,
                llama_server_path
            ],
            outputs=[status_display],
        )

        unload_model_btn.click(
            fn=unload_model_handler,
            outputs=[status_display],
        )

        def send_message(msg, history, sys_prompt, img1, img2, img3, img4, vid, max_tok, temp, vid_frames, thinking):
            if not msg.strip() and img1 is None and img2 is None and img3 is None and img4 is None and vid is None:
                yield history, "", None, None, None, None, None, ""
                return

            # Stream responses from chat_handler generator
            for new_history, _, stats in chat_handler(
                msg, history, sys_prompt, img1, img2, img3, img4, vid,
                max_tok, temp, vid_frames, thinking
            ):
                yield new_history, "", None, None, None, None, None, stats

        send_btn.click(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt,
                image_input_1, image_input_2, image_input_3, image_input_4,
                video_input, max_tokens, temperature, video_max_frames, show_thinking
            ],
            outputs=[chatbot, msg_input, image_input_1, image_input_2, image_input_3, image_input_4, video_input, stats_display],
        )

        msg_input.submit(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt,
                image_input_1, image_input_2, image_input_3, image_input_4,
                video_input, max_tokens, temperature, video_max_frames, show_thinking
            ],
            outputs=[chatbot, msg_input, image_input_1, image_input_2, image_input_3, image_input_4, video_input, stats_display],
        )

        clear_btn.click(
            fn=clear_chat_handler,
            outputs=[chatbot],
        )

        stop_btn.click(
            fn=stop_generation_handler,
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
    parser = argparse.ArgumentParser(description="Chromaforge VLM Chat Interface (llama.cpp Backend)")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/LLM",
        help="Directory containing GGUF models (default: models/LLM)",
    )
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

    args = parser.parse_args()

    # Override host if --listen is specified
    host = "0.0.0.0" if args.listen else args.host

    print("=" * 60)
    print("Chromaforge VLM Chat Interface (llama.cpp Backend)")
    print("=" * 60)
    print(f"llama-cpp-python: {'available' if LLAMA_CPP_AVAILABLE else 'NOT INSTALLED'}")
    print(f"Models directory: {args.models_dir}")
    print(f"Server: http://{host}:{args.port}")
    if args.listen:
        print("LAN access: enabled (listening on 0.0.0.0)")
    print("=" * 60)

    if not LLAMA_CPP_AVAILABLE:
        print("\nERROR: llama-cpp-python not installed!")
        print("\nInstall with CUDA support:")
        print("  Linux: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python")
        print("  Windows: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        return

    # Initialize the manager
    initialize_manager(args.models_dir)

    # List found models
    models = vlm_manager.get_available_models()
    if models:
        print(f"\nFound {len(models)} GGUF model(s):")
        for m in models:
            vision = " [+vision]" if m.get("mmproj_path") else ""
            print(f"  - {m['name']}{vision}")
    else:
        print(f"\nNo GGUF models found in {args.models_dir}")
        print("Download GGUF vision models and place them in this directory.")

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
