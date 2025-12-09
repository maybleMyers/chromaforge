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
    from llama_cpp.llama_chat_format import Llava15ChatHandler, Llava16ChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
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
    """
    models = []
    models_path = Path(models_dir)

    if not models_path.exists():
        return models

    # Look for .gguf files
    for gguf_file in models_path.rglob("*.gguf"):
        name = gguf_file.stem
        model_path = str(gguf_file)

        # Skip mmproj/clip files as main models
        if any(x in name.lower() for x in ["mmproj", "clip", "vision-encoder", "image-encoder"]):
            continue

        # Try to find matching mmproj file for vision models
        mmproj_path = None
        parent = gguf_file.parent

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

            # Set up chat handler for vision models
            self.chat_handler = None
            if mmproj_path and os.path.exists(mmproj_path):
                progress(0.2, desc="Loading vision encoder...")
                print(f"[llama.cpp] Loading mmproj from: {mmproj_path}")

                # Try different chat handlers
                try:
                    self.chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path, verbose=False)
                except Exception:
                    try:
                        self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
                    except Exception as e:
                        print(f"Warning: Could not load vision encoder: {e}")
                        self.chat_handler = None

            progress(0.3, desc=f"Loading {model_name}...")
            print(f"[llama.cpp] Loading model from: {model_path}")
            print(f"[llama.cpp] GPU layers: {n_gpu_layers}, Context: {n_ctx}")

            # Load the model
            self.model = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
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
    ) -> str:
        """
        Generate a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            images: Optional list of PIL Images
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            video_max_frames: Max frames for video processing

        Returns:
            Generated response string
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
):
    """Handle chat messages from UI."""
    if vlm_manager is None or vlm_manager.model is None:
        error_history = list(history)
        error_history.append({"role": "user", "content": message})
        error_history.append({"role": "assistant", "content": "Error: No model loaded. Please load a model first."})
        return error_history, ""

    # Build messages list for the model
    messages = []

    # Add system prompt if provided
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # Add chat history
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            content = msg.get("content", "")
            if isinstance(content, str):
                messages.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                # Handle mixed content - extract text parts
                text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                if text_parts:
                    messages.append({"role": msg["role"], "content": " ".join(text_parts)})

    # Build current message content
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

    # Generate response
    response = vlm_manager.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        video_max_frames=video_max_frames,
    )

    # Build display content for chatbot
    new_history = list(history)

    if image is not None:
        # Save image to temp file for display
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vlm_chat_{id(image)}.png")
        image.save(temp_path)
        display_text = message if message else "[Describe this image]"
        new_history.append({"role": "user", "content": [{"type": "text", "text": display_text}, {"type": "image", "path": temp_path}]})
        new_history.append({"role": "assistant", "content": response})
    elif video is not None:
        display_text = message if message else "[Describe this video]"
        new_history.append({"role": "user", "content": [{"type": "text", "text": display_text}, {"type": "video", "path": video}]})
        new_history.append({"role": "assistant", "content": response})
    else:
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": response})

    return new_history, ""


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
    """

    # Get initial model list
    initial_models = vlm_manager.get_model_names() if vlm_manager else ["Initialize manager first"]

    with gr.Blocks(title="Chromaforge VLM (llama.cpp)", theme=vlm_theme, css=vlm_css) as demo:
        gr.Markdown("# Chromaforge VLM Chat (llama.cpp Backend)")
        gr.Markdown("Load GGUF vision models and chat with images/video. Fully local, GPU accelerated.")

        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=initial_models,
                        value=initial_models[0] if initial_models else None,
                        interactive=True,
                        scale=4,
                    )
                    refresh_models_btn = gr.Button("🔄", scale=1, min_width=40)

                gr.Markdown("### GPU Settings")
                n_gpu_layers = gr.Slider(
                    minimum=-1,
                    maximum=100,
                    value=-1,
                    step=1,
                    label="GPU Layers (-1 = all)",
                    info="Number of layers to offload to GPU",
                )
                n_ctx = gr.Slider(
                    minimum=512,
                    maximum=32768,
                    value=4096,
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

                gr.Markdown("### Video Settings")
                video_max_frames = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=8,
                    step=1,
                    label="Max Video Frames",
                    info="Number of frames to extract from videos",
                )

            # Right column - Chat/Batch tabs
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Chat Tab
                    with gr.TabItem("Chat"):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400,
                            type="messages",
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

        def send_message(msg, history, sys_prompt, img, vid, max_tok, temp, vid_frames):
            if not msg.strip() and img is None and vid is None:
                return history, "", None, None

            new_history, _ = chat_handler(
                msg, history, sys_prompt, img, vid,
                max_tok, temp, vid_frames
            )
            return new_history, "", None, None

        send_btn.click(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, video_max_frames
            ],
            outputs=[chatbot, msg_input, image_input, video_input],
        )

        msg_input.submit(
            fn=send_message,
            inputs=[
                msg_input, chatbot, system_prompt, image_input, video_input,
                max_tokens, temperature, video_max_frames
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
