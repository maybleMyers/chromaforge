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
import base64
import argparse
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

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
    Uses the folder name as the display name instead of the GGUF filename.
    """
    models = []
    models_path = Path(models_dir)

    if not models_path.exists():
        return models

    # Look for .gguf files
    for gguf_file in models_path.rglob("*.gguf"):
        file_stem = gguf_file.stem
        model_path = str(gguf_file)

        # Skip mmproj/clip files as main models
        if any(x in file_stem.lower() for x in ["mmproj", "clip", "vision-encoder", "image-encoder"]):
            continue

        # Use folder name as display name instead of file name
        parent = gguf_file.parent
        if parent == models_path:
            # File is directly in models_dir, use file stem
            name = file_stem
        else:
            # File is in a subdirectory, use folder name
            name = parent.name

        # Try to find matching mmproj file for vision models
        mmproj_path = None

        # Common mmproj naming patterns
        mmproj_patterns = [
            "*mmproj*.gguf",
            "*clip*.gguf",
            "*vision*.gguf",
            "*image-encoder*.gguf",
        ]

        for pattern in mmproj_patterns:
            mmproj_files = list(parent.glob(pattern))
            if mmproj_files:
                mmproj_path = str(mmproj_files[0])
                break

        models.append({
            "name": name,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
        })

    return sorted(models, key=lambda x: x["name"])


class LlamaCppVLM:
    """Manages VLM inference via llama-cpp-python."""

    def __init__(self, models_dir: str = "models/LLM"):
        """Initialize llama.cpp VLM Manager."""
        self.models_dir = models_dir
        self.model: Optional[Llama] = None
        self.current_model_path: Optional[str] = None
        self.current_mmproj_path: Optional[str] = None
        self.chat_handler = None

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
        progress=gr.Progress(),
    ) -> str:
        """
        Load a GGUF model.

        Args:
            model_name: Name of the model to load
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context length
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

            # More specific detection
            is_qwen3_specific = any(x in model_name_lower or x in model_path_lower for x in ["qwen3"])
            is_qwen25_specific = any(x in model_name_lower or x in model_path_lower for x in ["qwen2.5", "qwen25"])

            print(f"[llama.cpp] Model type detection: Qwen-VL={is_qwen_vl}, Qwen3={is_qwen3_specific}, Qwen2.5={is_qwen25_specific}, LLaVA={is_llava}")

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
            print(f"[llama.cpp] GPU layers: {n_gpu_layers}, Context: {n_ctx}")

            # Load the model
            self.model = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=True,  # Enable verbose to see any loading issues
            )

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
            del self.model
            del self.chat_handler
            self.model = None
            self.chat_handler = None
            self.current_model_path = None
            self.current_mmproj_path = None

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
        if self.model is None:
            return "No model loaded"

        model_name = Path(self.current_model_path).stem if self.current_model_path else "Unknown"
        vision_status = "vision" if self.chat_handler else "text-only"
        return f"Loaded: {model_name} ({vision_status})"

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
        if self.model is None:
            return "Error: No model loaded. Please load a model first."

        try:
            # Build messages in llama.cpp format
            llama_messages = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    llama_messages.append({"role": "system", "content": content})

                elif role == "user":
                    if isinstance(content, list):
                        # Handle multimodal content
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

            # Add any direct images
            if images:
                image_parts = []
                for img in images:
                    b64_url = image_to_base64(img)
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": b64_url}
                    })
                image_parts.append({"type": "text", "text": "Describe this image."})
                llama_messages.append({"role": "user", "content": image_parts})

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
                for chunk in response_stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated += content
                        # Yield both raw (with thinking) and cleaned versions
                        # Clean up thinking tags for display
                        if "</think>" in accumulated:
                            display_text = accumulated.split("</think>")[-1].strip()
                        else:
                            display_text = accumulated
                        yield display_text, accumulated

                end_time = time.perf_counter()
                generation_time = end_time - start_time
                print(f"[llama.cpp] Streamed response in {generation_time:.2f}s")

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

                return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during generation: {str(e)}"


# Global manager instance
vlm_manager: Optional[LlamaCppVLM] = None


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


def load_model_handler(model_name: str, n_gpu_layers: int, n_ctx: int, progress=gr.Progress()):
    """Handle model loading."""
    if vlm_manager is None:
        return "Manager not initialized"
    return vlm_manager.load_model(model_name, n_gpu_layers, n_ctx, progress)


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
    image,
    video,
    max_tokens: int,
    temperature: float,
    video_max_frames: int = 8,
    show_thinking: bool = False,
):
    """Handle chat messages from UI with streaming support."""
    if vlm_manager is None or vlm_manager.model is None:
        error_history = list(history)
        error_history.append({"role": "user", "content": message})
        error_history.append({"role": "assistant", "content": "Error: No model loaded. Please load a model first."})
        yield error_history, ""
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

    if image is not None:
        model_content.append({"type": "image", "image": image})

    if video is not None:
        model_content.append({"type": "video", "video": video})

    if message.strip():
        model_content.append({"type": "text", "text": message})
    elif not model_content:
        model_content.append({"type": "text", "text": "Describe this image."})

    messages.append({"role": "user", "content": model_content})

    # Build initial display content for chatbot
    new_history = list(history)

    if image is not None:
        # Save image to temp file for display
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vlm_chat_{int(time.time())}_{id(image)}.png")
        image.save(temp_path)

        display_text = message if message else "Describe this image"
        new_history.append({"role": "user", "content": display_text})
        new_history.append({"role": "user", "content": gr.Image(temp_path)})

    elif video is not None:
        display_text = message if message else "Describe this video"
        new_history.append({"role": "user", "content": display_text})
        new_history.append({"role": "user", "content": gr.Video(video)})

    else:
        new_history.append({"role": "user", "content": message})

    # Add empty assistant message that we'll stream into
    new_history.append({"role": "assistant", "content": ""})

    # Stream the response
    for display_text, raw_text in vlm_manager.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        video_max_frames=video_max_frames,
        stream=True,
    ):
        # Show thinking if enabled, otherwise show cleaned text
        if show_thinking:
            new_history[-1]["content"] = raw_text
        else:
            new_history[-1]["content"] = display_text
        yield new_history, ""


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

                # Media inputs row
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image (optional)",
                            type="pil",
                            height=100,
                        )
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video (optional)",
                            height=100,
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
                    clear_btn = gr.Button("Clear Chat")
                    show_thinking = gr.Checkbox(
                        label="Show Thinking",
                        value=False,
                        info="Display model's reasoning process",
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
                load_model_btn = gr.Button("Load Model", variant="primary")
                unload_model_btn = gr.Button("Unload", variant="secondary")
                status_display = gr.Textbox(
                    label="Status",
                    value="No model loaded",
                    interactive=False,
                    scale=3,
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

        # Event handlers
        refresh_models_btn.click(
            fn=refresh_models_handler,
            outputs=[model_dropdown],
        )

        load_model_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, n_gpu_layers, n_ctx],
            outputs=[status_display],
        )

        unload_model_btn.click(
            fn=unload_model_handler,
            outputs=[status_display],
        )

        def send_message(msg, history, sys_prompt, img, vid, max_tok, temp, vid_frames, thinking):
            if not msg.strip() and img is None and vid is None:
                yield history, "", None, None
                return

            # Stream responses from chat_handler generator
            for new_history, _ in chat_handler(
                msg, history, sys_prompt, img, vid,
                max_tok, temp, vid_frames, thinking
            ):
                yield new_history, "", None, None

        send_btn.click(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, video_max_frames, show_thinking
            ],
            outputs=[chatbot, msg_input, image_input, video_input],
        )

        msg_input.submit(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, video_max_frames, show_thinking
            ],
            outputs=[chatbot, msg_input, image_input, video_input],
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
