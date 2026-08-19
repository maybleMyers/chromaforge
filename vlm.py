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
import shutil
import tempfile
import time
import subprocess
import signal
import threading
import requests
import html
from io import BytesIO
from pathlib import Path
from urllib.parse import quote as url_quote
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
    from llama_cpp import llama_cpp as llama_cpp_lib
    from llama_cpp.llama_chat_format import (
        Llava15ChatHandler,
        Llava16ChatHandler,
    )
    LLAMA_CPP_AVAILABLE = True

    # Mapping from string type names to GGML type constants
    GGML_TYPE_MAP = {
        "f16": 1,   # GGML_TYPE_F16
        "f32": 0,   # GGML_TYPE_F32
        "q4_0": 2,  # GGML_TYPE_Q4_0
        "q4_1": 3,  # GGML_TYPE_Q4_1
        "q5_0": 6,  # GGML_TYPE_Q5_0
        "q5_1": 7,  # GGML_TYPE_Q5_1
        "q8_0": 8,  # GGML_TYPE_Q8_0
    }

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
    GGML_TYPE_MAP = {}
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


def clean_harmony_tags(content: str) -> str:
    """
    Clean GPT-OSS Harmony format tags from content.
    Extracts the final channel content if present, otherwise removes all tags.
    """
    if not content or not isinstance(content, str):
        return content

    # Try to extract just the final channel content
    final_match = re.search(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|(?:end|return)\|>|$)',
        content,
        re.DOTALL
    )
    if final_match:
        return final_match.group(1).strip()

    # If no final channel, remove all harmony tags
    cleaned = re.sub(r'<\|channel\|>(?:analysis|commentary|final)<\|message\|>', '', content)
    cleaned = re.sub(r'<\|(?:start|end|return|call|message)\|>', '', cleaned)
    return cleaned.strip()


def format_think_tags_for_display(text: str) -> str:
    """
    Convert <think>...</think> markers into markdown Gradio renders reliably.
    A literal <think> tag is stripped as unknown HTML by the chatbot's
    sanitizer, which can blank the entire message.
    """
    if not text or "<think>" not in text:
        return text
    formatted = text.replace("<think>", "**[Thinking]**\n\n", 1)
    formatted = formatted.replace("</think>", "\n\n**[Response]**\n\n", 1)
    return formatted


# Muse Glimmer turn markers. llama.cpp b10353+ parses these itself and returns
# clean content, so these should never reach us. They do if llama-server predates
# the muse_glimmer chat parser, in which case content starts with a literal
# "to=self<|message|>" - stripping it turns a baffling transcript into an obvious
# "your llama-server is too old" signal instead of raw markup in the chat window.
# A full header, with or without a leading <|start|> and with or without a
# " to=<recipient>" - e.g. "<|start|>assistant to=self<|message|>" or a bare
# "to=self<|message|>" when the header was consumed as the generation prompt.
_MUSE_HEADER_RE = re.compile(
    r"(?:<\|start\|>\s*\w+)?[ \t]*(?:to=\S+)?[ \t]*<\|message\|>"
)
_MUSE_MARKER_RE = re.compile(r"<\|(?:start|end|message|eom|eot|end_of_text)\|>")


def clean_muse_glimmer_markers(text: str) -> str:
    """Defensive: strip Muse Glimmer channel markup that a pre-b10353 llama-server
    would leave in the response body."""
    if not text or not isinstance(text, str):
        return text
    if not any(t in text for t in ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>")):
        return text
    # Headers become paragraph breaks so consecutive messages don't run together
    cleaned = _MUSE_HEADER_RE.sub("\n\n", text)
    cleaned = _MUSE_MARKER_RE.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# How often the streaming loop is allowed to push a frame to the UI, in seconds.
# Every yield makes Gradio re-serialise the whole chat history and blocks the
# generator while it does, so yielding per token pins generation to the browser's
# redraw rate (~26/s on a long message) no matter how fast the server runs.
# ~15 fps still reads as smooth streaming and costs nothing.
UI_STREAM_INTERVAL = float(os.environ.get("VLM_UI_STREAM_INTERVAL", "0.066"))

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DISPLAY_THINK_RE = re.compile(r"\*\*\[Thinking\]\*\*.*?\*\*\[Response\]\*\*\s*", re.DOTALL)


def strip_reasoning_for_context(text: str) -> str:
    """
    Remove reasoning blocks from a previous assistant response before sending
    it back to the model. Thinking models (DeepSeek, GLM, Qwen) expect prior
    turns without reasoning; resending it wastes context (thousands of tokens
    per turn) and degrades output.
    Handles both raw <think>...</think> and the display form from
    format_think_tags_for_display, including unclosed blocks from stopped
    generations.
    """
    if not text or not isinstance(text, str):
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    text = _DISPLAY_THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    if "**[Thinking]**" in text:
        text = text.split("**[Thinking]**")[0]
    return text.strip()


def format_gpt_oss_system_prompt(user_prompt: str, reasoning_level: str = "medium") -> str:
    """
    Format a system prompt for GPT-OSS models with proper Harmony format requirements.

    Args:
        user_prompt: The user's system prompt content
        reasoning_level: Reasoning level (low, medium, high)

    Returns:
        Properly formatted system prompt for GPT-OSS
    """
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Build the GPT-OSS system prompt with required elements
    harmony_header = f"""Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final. Channel must be included for every message."""

    if user_prompt and user_prompt.strip():
        return f"{user_prompt.strip()}\n\n{harmony_header}"
    else:
        return f"You are a helpful AI assistant.\n\n{harmony_header}"


def is_kimi_model(*values: Optional[str]) -> bool:
    """
    Detect Kimi K2 / K2.6 from a model name or path.
    These GGUFs report the deepseek2 architecture but need their own handling:
    K2.6 ships a vision projector and its chat template emits <think> blocks.
    """
    return any("kimi" in str(v).lower() for v in values if v)


def is_muse_glimmer_model(*values: Optional[str]) -> bool:
    """
    Detect meta-models/Muse-Glimmer-30B from a model name or path.
    The GGUFs report general.architecture "muse-glimmer" (HF model_type is
    "muse_glimmer"), need llama.cpp b10353+, and are llama-server only.
    Both tokens are required: a bare "muse" would over-match unrelated repos,
    while requiring the literal "muse-glimmer" would miss MuseGlimmer/Muse_Glimmer.
    """
    return any("muse" in str(v).lower() and "glimmer" in str(v).lower()
               for v in values if v)


# Folder names that identify a quantization variant rather than a model. The unsloth
# GGUF repos are laid out as <repo>/UD-Q2_K_XL/*.gguf with the vision projector
# (mmproj-*.gguf) sitting one level up in the repo root.
# Anchored so a model folder that merely starts with a quant-ish token isn't mistaken
# for one. The trailing run covers compound names like "UD-Q2_K_XL-MXFP4".
_QUANT_DIR_RE = re.compile(r"^(ud-)?(i?q\d|bf16|f16|f32|mxfp4)[\w.\-]*$", re.IGNORECASE)

# Some repos prefix the quant folder with the model name instead of naming it after
# the quant alone - K2.7-Code-GGUF/k2.7-code-UD-Q2_K_XL-MXFP4 vs Kimi-K2.6-GGUF/UD-Q2_K_XL.
# The projector still sits in the repo root, so a trailing quant token is enough to
# recognise the folder and look one level up.
_QUANT_SUFFIX_RE = re.compile(r"[-_](ud-)?(i?q\d|bf16|f16|f32|mxfp4)[\w.\-]*$", re.IGNORECASE)

# mmproj precision preference - F16 is what unsloth recommends for Kimi K2.6, but
# K2.7-Code ships the projector as fp32 only (mmproj-F32.gguf), so F32 is a first
# class option rather than a fallback.
_MMPROJ_PREFERENCE = ("f16", "bf16", "f32")

_MMPROJ_PATTERNS = ["mmproj", "clip", "vision-encoder", "image-encoder"]

# Speculative-decoding drafters shipped alongside the weights. Muse Glimmer's
# DFlash drafter (dflash-*.gguf) sits in the same folder as the main quants, so
# without this it would be offered in the dropdown as a loadable model.
_DRAFT_PATTERNS = ["dflash", "draft", "drafter"]

# Drafters are tiny and pure overhead on a tight VRAM budget, so prefer the
# smallest useful quant rather than the highest precision.
_DRAFT_PREFERENCE = ("q4_k_m", "q4_0", "q5_k_m", "q6_k", "q8_0", "f16", "bf16", "f32")


def _is_mmproj(gguf_path: Path) -> bool:
    stem_lower = gguf_path.stem.lower()
    return any(p in stem_lower for p in _MMPROJ_PATTERNS)


def _is_draft(gguf_path: Path) -> bool:
    stem_lower = gguf_path.stem.lower()
    return any(p in stem_lower for p in _DRAFT_PATTERNS)


def _pick_by_precision(candidates: List[Path], preference: Tuple[str, ...]) -> Optional[str]:
    """Choose one companion file from a folder, ranked by a precision preference."""
    if not candidates:
        return None

    def rank(path: Path) -> Tuple[int, str]:
        # Match on the separator too: "mmproj-bf16".endswith("f16") is True, which
        # would otherwise let BF16 tie with F16 and win the name tiebreak.
        stem = path.stem.lower()
        for i, tag in enumerate(preference):
            if stem.endswith(f"-{tag}") or stem.endswith(f"_{tag}") or stem == tag:
                return (i, path.name)
        return (len(preference), path.name)

    return str(sorted(candidates, key=rank)[0])


def _pick_mmproj(candidates: List[Path]) -> Optional[str]:
    """Choose one vision projector from a folder, preferring F16 over BF16/F32."""
    return _pick_by_precision(candidates, _MMPROJ_PREFERENCE)


def _pick_draft(candidates: List[Path]) -> Optional[str]:
    """Choose one speculative-decoding drafter from a folder, smallest quant first."""
    return _pick_by_precision(candidates, _DRAFT_PREFERENCE)


# llama-server always hands video to mtmd as an in-memory buffer (handle_media reads
# input_video.data into a raw_buffer before mtmd sees it), so mtmd probes it by piping
# the bytes to ffprobe on *stdin*. mtmd-helper.cpp asks ffmpeg for "cache:pipe:0",
# which can seek backwards for a container header, but asks ffprobe for a bare
# "pipe:0", which cannot. An MP4 written without +faststart keeps its moov atom after
# the mdat payload, so over that non-seekable pipe ffprobe reports width/height/fps
# but "duration=N/A". probe() still returns true, mtmd_helper_video_init succeeds, the
# request comes back 200 - and n_frames lands at -1, so no frames ever reach the model.
# The symptom is a video that is silently ignored while the chat answers instantly.
#
# Remuxing with -movflags +faststart moves the moov atom to the front, which makes the
# duration readable over a pipe. It is a stream copy, so no re-encode and no quality
# loss. Cached per (path, size, mtime) because the same clip is re-sent every turn.
_FASTSTART_CACHE: Dict[Tuple[str, int, float], str] = {}


def _mp4_is_faststart(video_path: str) -> Optional[bool]:
    """
    Walk the top-level ISO-BMFF box list and report whether 'moov' precedes 'mdat'.

    True  = already faststart, nothing to do.
    False = moov sits after the payload, which is what breaks mtmd's pipe probe.
    None  = not an MP4/MOV (no 'ftyp'), so this test does not apply.

    Reads only the box headers, so it costs a handful of seeks regardless of file size.
    """
    try:
        with open(video_path, "rb") as f:
            head = f.read(8)
            if len(head) < 8 or head[4:8] != b"ftyp":
                return None  # not ISO-BMFF (mkv, webm, avi, ...)

            f.seek(0)
            offset = 0
            size = os.path.getsize(video_path)
            while offset + 8 <= size:
                f.seek(offset)
                header = f.read(8)
                if len(header) < 8:
                    break
                box_size = int.from_bytes(header[0:4], "big")
                box_type = header[4:8]

                if box_size == 1:  # 64-bit extended size follows the type
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    box_size = int.from_bytes(ext, "big")
                elif box_size == 0:  # box runs to end of file
                    box_size = size - offset

                if box_type == b"moov":
                    return True
                if box_type == b"mdat":
                    return False
                if box_size <= 0:
                    break
                offset += box_size
    except OSError as e:
        print(f"[video] Could not inspect MP4 box order ({e})")
        return None

    return None


def ensure_video_pipe_safe(video_path: str) -> Tuple[str, bool]:
    """
    Return a path whose container mtmd can probe over a pipe, remuxing if needed.

    Returns (path_to_send, was_remuxed). On any failure the original path is returned
    so the caller still tries - a failed remux should not block the request.
    """
    try:
        stat = os.stat(video_path)
        cache_key = (os.path.abspath(video_path), stat.st_size, stat.st_mtime)
    except OSError:
        return video_path, False

    cached = _FASTSTART_CACHE.get(cache_key)
    if cached and os.path.exists(cached):
        return cached, True

    if _mp4_is_faststart(video_path) is not False:
        # Already faststart, or a container this does not apply to. Send as-is.
        return video_path, False

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        # mtmd shells out to ffmpeg too, so if it is missing the server would not have
        # advertised video support in the first place. Nothing we can do here.
        print("[video] moov atom is at end of file but ffmpeg is not on PATH; "
              "sending as-is (the model will probably not see the video)")
        return video_path, False

    print(f"[video] {os.path.basename(video_path)} has its moov atom after mdat, "
          "which breaks mtmd's pipe probe; remuxing with +faststart")

    out_dir = os.path.join(tempfile.gettempdir(), "vlm_faststart")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_{stat.st_size}_{int(stat.st_mtime)}_faststart.mp4")

    try:
        result = subprocess.run(
            [ffmpeg_bin, "-nostdin", "-y", "-i", video_path,
             "-c", "copy", "-movflags", "+faststart", out_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            tail = result.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            print(f"[video] Remux failed (exit {result.returncode}); sending the original. "
                  f"ffmpeg said: {' | '.join(tail)}")
            return video_path, False
    except Exception as e:
        print(f"[video] Remux failed ({e}); sending the original")
        return video_path, False

    _FASTSTART_CACHE[cache_key] = out_path
    print(f"[video] Remuxed to {out_path}")
    return out_path, True


def extract_video_frames(video_path: str, max_frames: int = 8, target_size: Tuple[int, int] = (448, 448), every_other_frame: bool = False) -> List[Image.Image]:
    """
    Extract frames from a video file for VLM processing.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        target_size: Target size for frames (width, height)
        every_other_frame: If True, only use every other frame from the video (useful for long videos)

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

        # If every_other_frame is enabled, we effectively halve the available frames
        if every_other_frame:
            # Only consider even-indexed frames (0, 2, 4, 6, ...)
            available_frames = (total_frames + 1) // 2
            print(f"[extract_video_frames] Every other frame enabled: {total_frames} total -> {available_frames} available")
        else:
            available_frames = total_frames

        # Calculate frame indices to extract (evenly spaced from available frames)
        if available_frames <= max_frames:
            frame_indices = list(range(available_frames))
        else:
            frame_indices = [int(i * (available_frames - 1) / (max_frames - 1)) for i in range(max_frames)]

        # If every_other_frame is enabled, convert indices back to actual frame numbers
        if every_other_frame:
            frame_indices = [idx * 2 for idx in frame_indices]

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

            # Skip mmproj/clip and speculative drafters as main models
            if _is_mmproj(gguf_file) or _is_draft(gguf_file):
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
                "draft_path": None,
            })
            continue

        # Mark this folder as processed
        processed_folders.add(parent)

        # Use folder name as display name
        name = parent.name

        # Find all gguf files in this folder
        all_ggufs = list(parent.glob("*.gguf"))

        # Separate mmproj/clip files and speculative drafters from model files.
        # Drafters are tested first so a "mmproj-dflash-*" style name can't be
        # misfiled as a projector.
        mmproj_candidates = []
        draft_candidates = []
        model_files = []

        for gf in all_ggufs:
            if _is_draft(gf):
                draft_candidates.append(gf)
            elif _is_mmproj(gf):
                mmproj_candidates.append(gf)
            else:
                model_files.append(gf)

        if not model_files:
            continue

        # Repos that split quants into subfolders (unsloth's Kimi-K2.6-GGUF has
        # UD-Q2_K_XL/*.gguf) keep the vision projector in the repo root, one level
        # up. Qualify the name as well, so several quants of the same repo stay
        # tellable apart in the dropdown.
        is_quant_dir = bool(_QUANT_DIR_RE.match(parent.name))
        if (is_quant_dir or _QUANT_SUFFIX_RE.search(parent.name)) and parent.parent != models_path:
            # A bare quant name ("UD-Q2_K_XL") is meaningless on its own in the
            # dropdown; a model-prefixed one ("k2.7-code-UD-Q2_K_XL-MXFP4") already
            # identifies itself, so it keeps its own name.
            if is_quant_dir:
                name = f"{parent.parent.name}/{parent.name}"
            if not mmproj_candidates:
                mmproj_candidates = [gf for gf in parent.parent.glob("*.gguf") if _is_mmproj(gf)]
            if not draft_candidates:
                draft_candidates = [gf for gf in parent.parent.glob("*.gguf") if _is_draft(gf)]

        mmproj_path = _pick_mmproj(mmproj_candidates)
        draft_path = _pick_draft(draft_candidates)

        # Sort model files to get consistent ordering
        # For split models, this ensures we get the first shard
        model_files.sort(key=lambda x: x.name)

        # Use the first model file (or first shard for split models)
        model_path = str(model_files[0])

        models.append({
            "name": name,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "draft_path": draft_path,
        })

    return sorted(models, key=lambda x: x["name"])


def get_vram_info() -> str:
    """Query VRAM usage for all GPUs via nvidia-smi. Returns '' if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""
        readouts = []
        for i, line in enumerate(result.stdout.strip().splitlines()):
            used, total = (int(x.strip()) for x in line.split(","))
            readouts.append(f"GPU{i}: {used:,}/{total:,} MiB")
        return " | ".join(readouts)
    except Exception:
        return ""


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
        # Set from /props after the server comes up. True only when the binary was
        # built with MTMD_VIDEO, ffmpeg/ffprobe are on PATH, and an mmproj is loaded.
        self.server_supports_video = False

        # Context tracking
        self.n_ctx: int = 0  # Store context size for usage display

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
        use_mmap: bool = True,
        use_mlock: bool = False,
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
            use_mmap: If True (default), use memory-mapped files; if False, load fully into RAM
            use_mlock: If True, lock model in RAM to prevent disk access after initial load
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

            # Detect Muse Glimmer first: its chat format is Harmony-LIKE
            # (<|start|>assistant to=self<|message|> ... <|eom|>) but NOT GPT-OSS's
            # <|channel|> format, so clean_harmony_tags would mangle it into a
            # literal "assistant to=self" in the transcript.
            is_muse_glimmer = is_muse_glimmer_model(model_name, model_path)

            # Detect GPT-OSS models (text-only with Harmony format)
            is_gpt_oss = not is_muse_glimmer and any(x in model_name_lower or x in model_path_lower for x in ["gpt-oss", "gptoss", "gpt_oss", "huihui-gpt-oss"])

            # Detect Kimi K2 / K2.6 before DeepSeek: K2.6 is built on the deepseek2
            # architecture but is multimodal and needs its own handling
            is_kimi = is_kimi_model(model_name, model_path)

            # Detect DeepSeek models (text-only thinking models, e.g. DeepSeek V4 Flash)
            is_deepseek = not is_kimi and not is_muse_glimmer and ("deepseek" in model_name_lower or "deepseek" in model_path_lower)

            # More specific detection
            is_qwen3_specific = any(x in model_name_lower or x in model_path_lower for x in ["qwen3"])
            is_qwen25_specific = any(x in model_name_lower or x in model_path_lower for x in ["qwen2.5", "qwen25"])

            print(f"[llama.cpp] Model type detection: GPT-OSS={is_gpt_oss}, DeepSeek={is_deepseek}, Kimi={is_kimi}, MuseGlimmer={is_muse_glimmer}, Qwen-VL={is_qwen_vl}, Qwen3={is_qwen3_specific}, Qwen2.5={is_qwen25_specific}, LLaVA={is_llava}")

            # Muse Glimmer cannot run in-process at all. llama-cpp-python vendors its
            # own llama.cpp build, so rebuilding ./llama.cpp does not reach it, and
            # anything older than b10353 has no LLM_ARCH_MUSE_GLIMMER - Llama() would
            # raise "unknown model architecture" from deep inside the constructor.
            # Fail here with something actionable instead.
            if is_muse_glimmer:
                self.model_type = "muse-glimmer"
                return ("Error: Muse Glimmer requires the llama-server backend. "
                        "llama-cpp-python vendors its own llama.cpp build, which almost "
                        "certainly predates LLM_ARCH_MUSE_GLIMMER (needs b10353+). "
                        "Set Backend to 'llama-server' and load again.")

            # Set model type tracking. Kimi is text-only here because llama-cpp-python
            # has no handler for its MoonViT projector - vision needs llama-server.
            self.is_text_only_model = is_gpt_oss or is_deepseek or is_kimi or (mmproj_path is None)
            if is_gpt_oss:
                self.model_type = "gpt-oss"
            elif is_kimi:
                self.model_type = "kimi"
            elif is_deepseek:
                self.model_type = "deepseek"
            elif is_qwen_vl:
                self.model_type = "qwen-vl"
            elif is_llava:
                self.model_type = "llava"
            else:
                self.model_type = "generic"

            if is_gpt_oss:
                print("[llama.cpp] GPT-OSS model detected - using text-only mode with Harmony format")

            if is_deepseek:
                print("[llama.cpp] DeepSeek model detected - text-only mode")
                print("[llama.cpp] Note: for large DeepSeek MoE models, the llama-server backend is recommended")
                print("[llama.cpp]       (supports --override-tensor to keep experts on CPU, e.g. \\.ffn_.*_exps\\.weight=CPU)")

            if is_kimi:
                print("[llama.cpp] Kimi K2 model detected - text-only mode")
                if mmproj_path:
                    print("[llama.cpp] Note: llama-cpp-python has no Kimi vision handler, so the mmproj is ignored.")
                    print("[llama.cpp]       Use the llama-server backend for K2.6 image input.")
                print("[llama.cpp] Note: for the 1T MoE weights, the llama-server backend is recommended")
                print("[llama.cpp]       (supports --override-tensor to keep experts on CPU, e.g. \\.ffn_.*_exps\\.weight=CPU)")
                print("[llama.cpp] Recommended sampling: temperature 1.0, top_p 0.95 (thinking mode)")

            # Set up chat handler for vision models
            self.chat_handler = None
            if mmproj_path and os.path.exists(mmproj_path) and not is_kimi:
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
                "use_mmap": use_mmap,  # If False, forces loading into RAM instead of memory-mapping from disk
                "use_mlock": use_mlock,  # Lock model in RAM to prevent repeated disk access
            }

            if not use_mmap:
                print("[llama.cpp] Memory mapping disabled - loading model fully into RAM")
            if use_mlock:
                print("[llama.cpp] Memory locking enabled - model will be locked in RAM after loading")

            # Add optional parameters
            if tensor_split_list:
                llama_kwargs["tensor_split"] = tensor_split_list

            if flash_attn:
                llama_kwargs["flash_attn"] = True

            # Add KV cache type parameters if specified (convert string to GGML type constant)
            if type_k:
                type_k_int = GGML_TYPE_MAP.get(type_k.lower())
                if type_k_int is not None:
                    llama_kwargs["type_k"] = type_k_int
                else:
                    print(f"[llama.cpp] Warning: Unknown type_k '{type_k}', ignoring")
            if type_v:
                type_v_int = GGML_TYPE_MAP.get(type_v.lower())
                if type_v_int is not None:
                    llama_kwargs["type_v"] = type_v_int
                else:
                    print(f"[llama.cpp] Warning: Unknown type_v '{type_v}', ignoring")

            # Load the model
            self.model = Llama(**llama_kwargs)

            # After loading, check the model's actual architecture from metadata
            # This is more reliable than filename-based detection
            try:
                model_metadata = self.model.metadata
                arch = model_metadata.get("general.architecture", "").lower()
                if arch == "gpt-oss":
                    self.model_type = "gpt-oss"
                    self.is_text_only_model = True
                    print(f"[llama.cpp] Detected GPT-OSS architecture from model metadata")
                elif is_kimi and "deepseek" in arch:
                    # Kimi K2/K2.6 ship on the deepseek2 architecture - expected, and
                    # not a reason to treat the model as DeepSeek
                    print(f"[llama.cpp] Kimi model reports architecture '{arch}' (expected)")
                elif "deepseek" in arch:
                    # DeepSeek V3/V4 report deepseek* architectures (e.g. "deepseek2")
                    self.model_type = "deepseek"
                    self.is_text_only_model = True
                    print(f"[llama.cpp] Detected DeepSeek architecture '{arch}' from model metadata")
            except Exception as e:
                print(f"[llama.cpp] Could not read model metadata: {e}")

            self.current_model_path = model_path
            self.current_mmproj_path = mmproj_path
            self.n_ctx = n_ctx  # Store context size for usage tracking

            progress(1.0, desc="Model loaded!")

            vision_status = "with vision" if self.chat_handler else "text-only"
            model_type_info = f", type={self.model_type}" if self.model_type else ""
            return f"Loaded: {model_name} ({vision_status}, ctx={n_ctx}{model_type_info})"

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.model = None
            self.chat_handler = None
            return f"Error loading model: {str(e)}"

    def unload_model(self) -> str:
        """Unload the current model."""
        # Check if anything is loaded (either local model or server)
        if self.model is None and self.server_process is None:
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
                    print("[llama.cpp] Server didn't stop gracefully, killing...")
                    self.server_process.kill()
                print(f"[llama.cpp] Server stopped for {model_name}")
                self.server_process = None
                self.server_url = None
                self.use_server_backend = False
                self.server_supports_video = False

            # Clean up local model if loaded
            if self.model is not None:
                del self.model
            if self.chat_handler is not None:
                del self.chat_handler
            self.model = None
            self.chat_handler = None
            self.current_model_path = None
            self.current_mmproj_path = None
            self.is_text_only_model = False
            self.model_type = None
            self.n_ctx = 0

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
        use_mmap: bool = True,
        use_mlock: bool = False,
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
        mmproj_path = model_info.get("mmproj_path")
        draft_path = model_info.get("draft_path")

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

        # Disable mmap if requested (forces loading into RAM)
        if not use_mmap:
            cmd.append("--no-mmap")
            print("[llama-server] Memory mapping disabled - loading model fully into RAM")

        # Enable mlock if requested (locks model in RAM)
        if use_mlock:
            cmd.append("--mlock")
            print("[llama-server] Memory locking enabled - model will be locked in RAM after loading")

        # Add mmproj/clip model for vision support
        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path])
            print(f"[llama-server] Vision model (mmproj): {mmproj_path}")

        # Add tensor split if provided
        if tensor_split and tensor_split.strip():
            cmd.extend(["--tensor-split", tensor_split.strip()])

        # Add flash attention (new llama.cpp requires value: on/off/auto)
        if flash_attn:
            cmd.extend(["-fa", "on"])

        # Add KV cache types
        if type_k:
            cmd.extend(["--cache-type-k", type_k])
        if type_v:
            cmd.extend(["--cache-type-v", type_v])

        # Thinking models: use the GGUF's embedded jinja chat template and extract
        # <think> reasoning into reasoning_content (handled by generate_via_api's
        # existing reasoning_content -> <think> wrapping).
        # Kimi is tested first - K2/K2.6 are built on deepseek2 but are their own family.
        is_kimi = is_kimi_model(model_name, model_path)
        is_muse_glimmer = is_muse_glimmer_model(model_name, model_path)
        is_deepseek = not is_kimi and not is_muse_glimmer and ("deepseek" in model_name.lower() or "deepseek" in model_path.lower())
        extra_args_str = extra_args or ""

        # --jinja is mandatory for every family whose chat handler llama.cpp only
        # reaches through the GGUF's embedded template. chat_template_kwargs
        # (Kimi's instant mode, Muse Glimmer's reasoning_strength) are inert without it.
        if is_deepseek or is_kimi or is_muse_glimmer:
            if "--jinja" not in extra_args_str:
                cmd.append("--jinja")

        # Only the <think>-emitting families need the deepseek reasoning parser.
        # Muse Glimmer is deliberately excluded: llama.cpp's default
        # --reasoning-format auto already routes its "<|start|>assistant to=self
        # <|message|> ... <|eom|>" turns into reasoning_content via
        # common_chat_params_init_muse_glimmer. Forcing "deepseek" breaks that parse.
        if is_deepseek or is_kimi:
            if "--reasoning-format" not in extra_args_str:
                cmd.extend(["--reasoning-format", "deepseek"])
            family = "Kimi K2" if is_kimi else "DeepSeek"
            print(f"[llama-server] {family} model detected: enabling --jinja and --reasoning-format deepseek")

        if is_kimi:
            print("[llama-server] Kimi K2.6: recommended sampling is temperature 1.0 / top_p 0.95 (thinking),")
            print("[llama-server]   0.6 / 0.95 (instant); recommended context 98304, max 262144")
            if mmproj_path and os.path.exists(mmproj_path):
                print("[llama-server] Kimi K2.6 vision enabled")
            else:
                print("[llama-server] No mmproj found - text only. Kimi vision needs mmproj-F16.gguf")
                print("[llama-server]   (K2.6) or mmproj-F32.gguf (K2.7-Code), which unsloth ships in")
                print("[llama-server]   the repo root next to the quant folder")
            print("[llama-server] The Q2_K_XL weights need ~350GB across RAM+VRAM; keep experts on CPU")
            print("[llama-server]   via Override Tensor (\\.ffn_.*_exps\\.weight=CPU) or Extra Args (--n-cpu-moe)")

        if is_muse_glimmer:
            print("[llama-server] Muse Glimmer detected: enabling --jinja (no --reasoning-format;")
            print("[llama-server]   llama.cpp's muse_glimmer parser handles reasoning_content itself)")
            print("[llama-server] Requires llama.cpp b10353+ - older builds do not register the")
            print("[llama-server]   architecture and will refuse to load these files")
            print("[llama-server] Recommended sampling: temperature 1.0 / top_p 0.95 / top_k 64")
            print("[llama-server] Reasoning cannot be disabled on this model - Thinking Mode has no")
            print("[llama-server]   effect. Use Reasoning Level (low/medium/high/xhigh) instead")
            print("[llama-server] This is a DENSE model: leave Override Tensor blank, the MoE expert")
            print("[llama-server]   pattern matches no tensors here")
            if mmproj_path and os.path.exists(mmproj_path):
                print("[llama-server] Muse Glimmer vision enabled")
            else:
                print("[llama-server] No mmproj found - text only. Vision needs mmproj-*.gguf from the")
                print("[llama-server]   same GGUF repo as the weights")

        # Speculative decoding drafter (Muse Glimmer ships DFlash as dflash-*.gguf).
        # Auto-wired when one is found next to the weights, but an explicit -md in
        # Extra Args wins - that is how you turn drafting off, or retune --draft-max,
        # when the ~1.6GB of extra VRAM is better spent on the vision projector.
        if draft_path and os.path.exists(draft_path):
            if not any(f in extra_args_str for f in ("-md", "--spec-draft-model", "--model-draft")):
                cmd.extend(["-md", draft_path])
                if not any(f in extra_args_str for f in ("-ngld", "--spec-draft-ngl")):
                    cmd.extend(["-ngld", "99"])
                print(f"[llama-server] Speculative drafter: {draft_path} (fully offloaded)")
            else:
                print("[llama-server] Drafter found but -md supplied in Extra Args - using yours")

        # Add override tensor (the key MoE optimization!)
        if override_tensor and override_tensor.strip():
            # Support multiple patterns separated by semicolons.
            # Newer llama.cpp deprecates repeating -ot (only the last one applies),
            # so pass all patterns as a single comma-separated -ot value - the
            # comma-separated form is parsed by all llama.cpp versions.
            patterns = [p.strip() for p in override_tensor.split(";") if p.strip()]
            if patterns:
                cmd.extend(["-ot", ",".join(patterns)])

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

            # Terabyte-class MoE models spend a long time here: with --no-mmap the whole
            # file is read into RAM, and warmup then touches every weight. Kimi K2.6
            # Q2_K_XL (317GB) needs well over the 20 minutes this used to allow.
            max_wait = 5400  # 90 minutes
            start_time = time.time()
            server_ready = False
            last_status = None
            check_count = 0
            last_elapsed_report = 0

            while time.time() - start_time < max_wait:
                # Check if process died
                if self.server_process.poll() is not None:
                    return f"Error: llama-server exited with code {self.server_process.returncode}"

                check_count += 1
                elapsed = time.time() - start_time
                try:
                    resp = requests.get(health_url, timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data.get("status", "")
                        if status != last_status:
                            print(f"[llama-server] Health check: {resp.status_code}, status='{status}'")
                            last_status = status
                        if status == "ok":
                            server_ready = True
                            print(f"[llama-server] Server is ready! (took {elapsed / 60:.1f} min)")
                            break
                        elif status == "loading model":
                            progress(0.5, desc=f"Server loading model... ({elapsed / 60:.0f} min)")
                    elif resp.status_code != last_status:
                        print(f"[llama-server] Health check: {resp.status_code}")
                        last_status = resp.status_code
                except requests.exceptions.RequestException as e:
                    if check_count == 1:
                        print(f"[llama-server] Waiting for server to start...")
                    progress(0.5, desc=f"Waiting for llama-server... ({elapsed / 60:.0f} min)")

                # Reassure the console that we are still waiting, not wedged
                if elapsed - last_elapsed_report >= 60:
                    last_elapsed_report = elapsed
                    print(f"[llama-server] Still loading after {elapsed / 60:.0f} min "
                          f"(timeout at {max_wait / 60:.0f} min)")

                time.sleep(5)

            if not server_ready:
                self.server_process.terminate()
                self.server_process = None
                return (f"Error: Server failed to become ready within {max_wait / 60:.0f} min. "
                        "For very large MoE models, keep the experts off the GPU with Override "
                        "Tensor / --n-cpu-moe and consider leaving MMap enabled.")

            # Ask the server what modalities it actually accepts rather than guessing
            # from the model name. mtmd only reports video when llama.cpp was built
            # with MTMD_VIDEO *and* ffmpeg/ffprobe are on PATH *and* an mmproj is
            # loaded, so this is the only reliable gate for the input_video path.
            self.server_supports_video = False
            try:
                props = requests.get(f"{self.server_url}/props", timeout=10).json()
                modalities = props.get("modalities", {}) or {}
                self.server_supports_video = bool(modalities.get("video", False))
                print(f"[llama-server] Modalities: {modalities}")
            except Exception as e:
                # Older builds have no /props modalities block; fall back to frames.
                print(f"[llama-server] Could not read /props modalities ({e}); "
                      "assuming no native video support")

            self.current_model_path = model_path
            self.current_mmproj_path = mmproj_path
            self.use_server_backend = True
            self.is_text_only_model = is_deepseek or not (mmproj_path and os.path.exists(mmproj_path))
            # Set the type explicitly - unload_model() resets it to None, so carrying
            # over "self.model_type" here left every non-DeepSeek server load untyped
            if is_muse_glimmer:
                # generate_via_api keys the reasoning_strength kwarg off this
                self.model_type = "muse-glimmer"
            elif is_kimi:
                self.model_type = "kimi"
            elif is_deepseek:
                self.model_type = "deepseek"
            else:
                self.model_type = self.model_type or "generic"
            self.n_ctx = n_ctx  # Store context size for usage tracking

            vision_status = "with vision" if not self.is_text_only_model else "text-only"
            print(f"[llama-server] Flags set: use_server_backend={self.use_server_backend}, server_url={self.server_url}, vision={not self.is_text_only_model}, native_video={self.server_supports_video}")

            progress(1.0, desc="Server ready!")
            return f"Server started: {model_name} ({vision_status}, ctx={n_ctx}) @ {self.server_url}"

        except FileNotFoundError:
            return f"Error: llama-server not found at '{self.llama_server_path}'. Set the correct path."
        except Exception as e:
            if self.server_process:
                self.server_process.terminate()
                self.server_process = None
            return f"Error starting server: {str(e)}"

    def get_server_context_usage(self) -> Tuple[int, int]:
        """
        Query llama-server's /slots endpoint to get actual context usage.
        Returns (tokens_used, total_context).
        """
        if not self.server_url:
            return 0, self.n_ctx

        try:
            slots_url = f"{self.server_url}/slots"
            resp = requests.get(slots_url, timeout=2)
            if resp.status_code == 200:
                slots_data = resp.json()
                # slots_data is a list of slot objects
                if slots_data and isinstance(slots_data, list):
                    # Get the first slot (typically only one for single-user)
                    slot = slots_data[0]
                    n_ctx = slot.get("n_ctx", self.n_ctx)
                    # n_past is the number of tokens in context
                    n_past = slot.get("n_past", 0)
                    return n_past, n_ctx
        except Exception as e:
            print(f"[llama-server] Failed to query /slots: {e}")

        return 0, self.n_ctx

    def generate_via_api(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.0,
        seed: int = -1,
        video_max_frames: int = 81,
        every_other_frame: bool = False,
        stream: bool = False,
        thinking: bool = True,
        reasoning_level: str = "default",
    ):
        """Generate response via llama-server OpenAI-compatible API using requests only.
        Yields: (display_text, raw_text, speed_stats, context_info)
        """
        if not self.server_url:
            yield "Error: Server not running", "Error: Server not running", "0 tok/s", ""
            return

        api_url = f"{self.server_url}/v1/chat/completions"

        # Convert messages to OpenAI-compatible format (with multimodal support)
        api_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle multimodal content (lists with text and images)
            if isinstance(content, list):
                content_parts = []
                has_images = False

                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")

                        if item_type == "text":
                            text = item.get("text", "")
                            if text:
                                content_parts.append({"type": "text", "text": text})

                        elif item_type == "image" and "image" in item:
                            # Convert PIL image to base64 data URL for llama-server
                            img = item["image"]
                            if isinstance(img, Image.Image):
                                b64_url = image_to_base64(img)
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": b64_url}
                                })
                                has_images = True

                        elif item_type == "image_url":
                            # Already in correct format, pass through
                            content_parts.append(item)
                            has_images = True

                        elif item_type == "video" and "video" in item:
                            video_path = item["video"]
                            if isinstance(video_path, str) and os.path.exists(video_path):
                                try:
                                    if self.server_supports_video:
                                        # Hand the whole clip to mtmd. It shells out to
                                        # ffmpeg, samples at its own fps target and
                                        # interleaves "[MM:SSs]" timestamp text chunks
                                        # between frames - which is exactly what
                                        # Qwen3-VL's text time alignment expects, and
                                        # what the frame-dumping path below cannot give.
                                        # Frames also keep their aspect ratio here.
                                        #
                                        # Raw base64 only, no "data:video/mp4;base64,"
                                        # prefix: llama.cpp parses input_video with
                                        # accept_base64_uri=false (server-common.cpp),
                                        # so a data: URI is rejected. Same convention as
                                        # the input_audio branch below.
                                        # mtmd probes the buffer with ffprobe on a
                                        # non-seekable pipe, so a non-faststart MP4 is
                                        # accepted and then silently yields no frames.
                                        send_path, remuxed = ensure_video_pipe_safe(video_path)
                                        with open(send_path, "rb") as f:
                                            b64_video = base64.b64encode(f.read()).decode("utf-8")
                                        print(f"[llama-server] Sending video natively "
                                              f"({len(b64_video) // 1024} KB base64"
                                              f"{', remuxed +faststart' if remuxed else ''}): {send_path}")
                                        content_parts.append({
                                            "type": "input_video",
                                            "input_video": {"data": b64_video},
                                        })
                                        has_images = True
                                    else:
                                        # Fallback: no native video on this server, so
                                        # decode to stills client-side and send them as
                                        # separate images. Loses timestamps and squashes
                                        # every frame to a square.
                                        frames = extract_video_frames(video_path, max_frames=video_max_frames, every_other_frame=every_other_frame)
                                        print(f"[llama-server] Extracted {len(frames)} frames from video: {video_path}")
                                        for frame in frames:
                                            b64_url = image_to_base64(frame)
                                            content_parts.append({
                                                "type": "image_url",
                                                "image_url": {"url": b64_url}
                                            })
                                        content_parts.append({
                                            "type": "text",
                                            "text": f"[Video with {len(frames)} frames]"
                                        })
                                        has_images = True
                                except Exception as e:
                                    print(f"[llama-server] Video error: {e}")
                                    content_parts.append({
                                        "type": "text",
                                        "text": f"[Video error: {e}]"
                                    })

                        elif item_type == "audio" and "audio" in item:
                            audio_path = item["audio"]
                            if isinstance(audio_path, str) and os.path.exists(audio_path):
                                try:
                                    with open(audio_path, "rb") as f:
                                        b64_audio = base64.b64encode(f.read()).decode("utf-8")
                                    ext = audio_path.split('.')[-1].lower()
                                    if ext not in ['wav', 'mp3']:
                                        ext = 'wav'
                                    content_parts.append({
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": b64_audio,
                                            "format": ext
                                        }
                                    })
                                    has_images = True
                                except Exception as e:
                                    print(f"[llama-server] Audio processing error: {e}")
                                    content_parts.append({
                                        "type": "text",
                                        "text": f"[Audio error: {e}]"
                                    })

                    elif isinstance(item, str):
                        if item:
                            content_parts.append({"type": "text", "text": item})

                # If we have multimodal content, use the list format
                if content_parts:
                    if has_images:
                        # Use multimodal format
                        api_messages.append({"role": role, "content": content_parts})
                    else:
                        # Text only - extract and join
                        text_only = " ".join(
                            p.get("text", "") for p in content_parts if p.get("type") == "text"
                        )
                        if text_only:
                            api_messages.append({"role": role, "content": text_only})

            elif content:  # Simple string content
                if role == "assistant":
                    # Don't resend prior reasoning to the model
                    content = strip_reasoning_for_context(str(content))
                    if not content:
                        continue
                api_messages.append({"role": role, "content": str(content)})

        if not api_messages:
            yield "Error: No message content", "Error: No message content", "0 tok/s", ""
            return

        print(f"[llama-server] API request to {api_url}")
        print(f"[llama-server] {len(api_messages)} messages, max_tokens={max_new_tokens}, temp={temperature}, stream={stream}")
        for i, msg in enumerate(api_messages):
            content = msg['content']
            if isinstance(content, list):
                # Multimodal content - summarize
                text_parts = [p.get("text", "")[:50] for p in content if p.get("type") == "text"]
                # Count every media kind, not just image_url - a natively-sent video
                # is one "input_video" part and used to print as "[0 image(s)]",
                # which reads exactly like the video was dropped.
                kinds = {"image_url": "image", "input_video": "video", "input_audio": "audio"}
                counts: Dict[str, int] = {}
                for p in content:
                    kind = kinds.get(p.get("type"))
                    if kind:
                        counts[kind] = counts.get(kind, 0) + 1
                media = ", ".join(f"{n} {k}(s)" for k, n in counts.items()) or "no media"
                preview = f"[{media}] {' '.join(text_parts)}"[:80]
            else:
                preview = content[:80] + "..." if len(content) > 80 else content
            print(f"[llama-server]   [{i}] {msg['role']}: {preview}")

        payload = {
            "model": "local-model",
            "messages": api_messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.001,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stream": stream,
            # NOTE: top_k is deliberately absent, so a server-side default such as
            # Muse Glimmer's recommended "--top-k 64" (set via the recommended
            # profile's Extra Args) takes effect. Adding a top_k key here would
            # silently override it.
        }

        if seed is not None and int(seed) >= 0:
            payload["seed"] = int(seed)

        # Accumulate template kwargs rather than assigning the dict outright, so the
        # Kimi and Muse Glimmer paths cannot clobber one another.
        chat_template_kwargs: Dict[str, Any] = {}

        if not thinking:
            # Kimi K2.6 instant mode. Both spellings are sent because Kimi's own
            # template reads "thinking" while llama.cpp's server special-cases
            # "enable_thinking"; both must be JSON booleans (a string is rejected).
            # Templates that use neither just ignore the extra context keys.
            chat_template_kwargs.update({"thinking": False, "enable_thinking": False})
            print("[llama-server] Thinking disabled for this request (instant mode)")

        if self.model_type == "muse-glimmer":
            # Muse Glimmer's reasoning cannot be switched off, only dialled. Anything
            # outside the supported set (including the "default" sentinel) is omitted
            # so the model's own "high" default stands rather than being downgraded.
            strength = (reasoning_level or "").strip().lower()
            if strength in ("low", "medium", "high", "xhigh"):
                chat_template_kwargs["reasoning_strength"] = strength
                print(f"[llama-server] Muse Glimmer reasoning_strength={strength}")

        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        # Request usage info in streaming responses (OpenAI-compatible extension)
        if stream:
            payload["stream_options"] = {"include_usage": True}

        start_time = time.perf_counter()

        try:
            print(f"[llama-server] Sending POST request...")
            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=stream,
                timeout=3600,  # 1 hour timeout for long video processing
            )

            print(f"[llama-server] Response status: {response.status_code}")

            if response.status_code != 200:
                error_text = response.text[:500]
                print(f"[llama-server] Error response: {error_text}")
                yield f"Error {response.status_code}: {error_text}", f"Error {response.status_code}", "0 tok/s", ""
                return

            if stream:
                accumulated = ""
                token_count = 0
                prompt_tokens = 0
                ctx_info = ""  # Initialize context info for streaming
                in_reasoning = False  # Tracks whether we've opened a <think> tag we haven't closed yet
                finish_reason = None
                # Incremental <think> boundary tracking. The old code re-split the
                # entire accumulated buffer on every token, which is O(n^2) - by a
                # few thousand tokens that alone costs more than the generation.
                think_open = False   # a <think> (synthesised or literal) has been seen
                think_close = -1     # index in `accumulated` just past </think>, or -1
                scan_from = 0        # resume point for the </think> search
                last_yield = 0.0
                pending = False      # tokens arrived that the UI has not seen yet
                print(f"[llama-server] Reading streaming response...")

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        # Debug: show raw SSE data
                        if token_count == 0:
                            print(f"[llama-server] First SSE line: {line_str[:100]}")

                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                print(f"[llama-server] Received [DONE]")
                                break
                            try:
                                chunk = json.loads(data_str)
                                # Check for usage info (with stream_options.include_usage=true)
                                usage = chunk.get("usage")
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                    # Update context info with accurate values from usage
                                    total_tokens = usage.get("total_tokens", prompt_tokens + token_count)
                                    ctx_total_display = self.n_ctx if self.n_ctx > 0 else 32768
                                    ctx_pct = (total_tokens / ctx_total_display * 100) if ctx_total_display > 0 else 0
                                    ctx_info = f"{total_tokens:,} / {ctx_total_display:,} ({ctx_pct:.0f}%)"
                                    print(f"[llama-server] Usage info received: prompt={prompt_tokens}, total={total_tokens}")

                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    # GLM-4.7 sends thinking in reasoning_content field
                                    reasoning_content = delta.get("reasoning_content", "")
                                    finish_reason = choices[0].get("finish_reason") or finish_reason

                                    # Handle reasoning_content (GLM-4.7 thinking mode)
                                    if reasoning_content:
                                        # Wrap in <think> tags for compatibility with existing display logic.
                                        # Only open the tag once, on the first reasoning token - track this
                                        # with an explicit flag rather than inferring it from the string,
                                        # since arbitrary token text won't reliably end in ">".
                                        if not in_reasoning:
                                            accumulated += "<think>"
                                            in_reasoning = True
                                            think_open = True
                                            scan_from = len(accumulated)
                                        accumulated += reasoning_content
                                        token_count += 1

                                    # Handle regular content
                                    if content:
                                        # Close think tag if we were in reasoning mode
                                        if in_reasoning:
                                            accumulated += "</think>"
                                            in_reasoning = False
                                        accumulated += content
                                        token_count += 1
                                        if think_close < 0:
                                            # Models that emit a literal <think> inside content
                                            # (rather than reasoning_content) open the block here.
                                            if not think_open and accumulated[:32].lstrip().startswith("<think>"):
                                                think_open = True
                                            if think_open:
                                                hit = accumulated.find("</think>", scan_from)
                                                if hit >= 0:
                                                    think_close = hit + len("</think>")
                                                else:
                                                    # Only rescan the last few chars, in case the
                                                    # tag is split across two deltas.
                                                    scan_from = max(scan_from, len(accumulated) - 8)

                                    # Push to the UI at most every UI_STREAM_INTERVAL. Tokens
                                    # keep accumulating in between, so nothing is dropped - the
                                    # next frame just carries several tokens instead of one.
                                    if content or reasoning_content:
                                        pending = True
                                        now = time.perf_counter()
                                        if now - last_yield >= UI_STREAM_INTERVAL:
                                            last_yield = now
                                            pending = False
                                            elapsed = now - start_time
                                            tps = token_count / elapsed if elapsed > 0 else 0
                                            current_ctx = prompt_tokens + token_count
                                            ctx_total_display = self.n_ctx if self.n_ctx > 0 else 32768
                                            ctx_pct = (current_ctx / ctx_total_display * 100) if ctx_total_display > 0 else 0
                                            ctx_info = f"{current_ctx:,} / {ctx_total_display:,} ({ctx_pct:.0f}%)"
                                            # display_text = response with thinking stripped (for show_thinking=False)
                                            if think_close >= 0:
                                                display_text = accumulated[think_close:].lstrip()
                                            elif think_open or in_reasoning:
                                                # Still inside reasoning - show a placeholder instead of
                                                # raw <think> text (which Gradio renders as nothing)
                                                display_text = "*Thinking…*"
                                            else:
                                                display_text = accumulated
                                            yield display_text, accumulated, f"{tps:.1f} tok/s", ctx_info
                            except json.JSONDecodeError as e:
                                print(f"[llama-server] JSON decode error: {e} for: {data_str[:100]}")

                end_time = time.perf_counter()
                generation_time = end_time - start_time
                final_speed = token_count / generation_time if generation_time > 0 else 0

                # Calculate final context usage from accumulated values
                # Don't query /slots here - slot is already released and n_past will be 0
                total_ctx_used = prompt_tokens + token_count
                ctx_total = self.n_ctx if self.n_ctx > 0 else ctx_total
                ctx_pct = (total_ctx_used / ctx_total * 100) if ctx_total > 0 else 0
                final_ctx_info = f"{total_ctx_used:,} / {ctx_total:,} ({ctx_pct:.0f}%)"

                print(f"[llama-server] Done: {token_count} tokens in {generation_time:.2f}s ({final_speed:.1f} tok/s) | ctx: {total_ctx_used}/{ctx_total}")

                # Ensure think tag is closed if model only produced reasoning content
                if in_reasoning:
                    accumulated += "</think>"
                    in_reasoning = False

                if finish_reason == "length":
                    print(
                        f"[llama-server] TRUNCATED: hit max_tokens ({max_new_tokens}) before the model "
                        f"finished. Raise Max Tokens, or turn thinking down/off for this turn."
                    )

                if accumulated:
                    # Create final display_text with thinking stripped
                    if think_close < 0 and "</think>" in accumulated:
                        # Belt and braces: catch a </think> the incremental scan missed
                        think_close = accumulated.index("</think>") + len("</think>")
                    if think_close >= 0:
                        final_display = accumulated[think_close:].strip()
                    elif think_open or accumulated.lstrip().startswith("<think>"):
                        final_display = "*Thinking…* (no final response - hit max tokens during reasoning)"
                    else:
                        final_display = accumulated
                    if finish_reason == "length":
                        final_display += (
                            f"\n\n---\n*⚠ Truncated: hit the {max_new_tokens:,}-token limit.*"
                        )
                    if self.model_type == "muse-glimmer":
                        final_display = clean_muse_glimmer_markers(final_display)
                        accumulated = clean_muse_glimmer_markers(accumulated)
                    yield final_display, accumulated, f"{final_speed:.1f} tok/s | {generation_time:.1f}s", final_ctx_info
                else:
                    print(f"[llama-server] Warning: No content accumulated from stream")
                    yield "No response generated", "No response generated", f"0 tok/s | {generation_time:.1f}s", final_ctx_info
            else:
                # Non-streaming mode
                data = response.json()
                print(f"[llama-server] Got JSON response: {str(data)[:200]}")
                result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if self.model_type == "muse-glimmer":
                    result = clean_muse_glimmer_markers(result)
                end_time = time.perf_counter()
                elapsed = end_time - start_time

                # Get usage info from response (most reliable source)
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                if completion_tokens > 0:
                    speed = completion_tokens / elapsed if elapsed > 0 else 0
                else:
                    # Estimate from word count
                    completion_tokens = len(result.split())
                    total_tokens = prompt_tokens + completion_tokens
                    speed = completion_tokens / elapsed if elapsed > 0 else 0

                # Calculate context usage from usage info
                ctx_total = self.n_ctx if self.n_ctx > 0 else 32768
                ctx_pct = (total_tokens / ctx_total * 100) if ctx_total > 0 else 0
                ctx_info = f"{total_tokens:,} / {ctx_total:,} ({ctx_pct:.0f}%)"

                print(f"[llama-server] Done in {elapsed:.2f}s ({completion_tokens} tokens, {speed:.1f} tok/s) | ctx: {total_tokens}/{ctx_total}")
                yield result, result, f"{speed:.1f} tok/s | {elapsed:.1f}s", ctx_info

        except requests.exceptions.Timeout:
            print(f"[llama-server] Request timed out")
            yield "Error: Request timed out", "Error: Request timed out", "0 tok/s", ""
        except requests.exceptions.ConnectionError as e:
            print(f"[llama-server] Connection error: {e}")
            yield f"Error: Connection failed - {e}", f"Error: Connection failed", "0 tok/s", ""
        except Exception as e:
            print(f"[llama-server] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            yield f"Error: {str(e)}", f"Error: {str(e)}", "0 tok/s", ""

    def generate(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.0,
        seed: int = -1,
        video_max_frames: int = 8,
        every_other_frame: bool = False,
        stream: bool = False,
        reasoning_level: str = "default",
        thinking: bool = True,
    ):
        """
        Generate a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            images: Optional list of PIL Images
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            video_max_frames: Max frames for video processing
            every_other_frame: If True, use every other frame from videos
            stream: If True, yields partial responses as a generator
            reasoning_level: Muse Glimmer reasoning_strength (low, medium, high,
                xhigh) / GPT-OSS reasoning level. "default" leaves it to the model.
            thinking: If False, ask the server to skip reasoning (Kimi K2.6 instant mode).
                Only honoured by the llama-server backend.

        Returns:
            Generated response string, or generator if stream=True
        """
        # Debug: show current state
        print(f"[vlm.py] generate() called: use_server_backend={self.use_server_backend}, server_process={self.server_process is not None}, model={self.model is not None}")

        # Route to API if using server backend
        if self.use_server_backend and self.server_process is not None:
            print(f"[vlm.py] Using server backend at {self.server_url}")
            # Must use 'yield from' because generate() is a generator function (has yield statements)
            # Using 'return' would just terminate the generator without yielding anything
            yield from self.generate_via_api(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
                video_max_frames=video_max_frames,
                every_other_frame=every_other_frame,
                stream=stream,
                thinking=thinking,
                reasoning_level=reasoning_level,
            )
            return

        if self.model is None and not self.use_server_backend:
            error_msg = "Error: No model loaded. Please load a model first."
            yield error_msg, error_msg, "0 tok/s", ""
            return

        if self.model is None and self.use_server_backend:
            # Server mode but server_process is None - server may have crashed
            error_msg = f"Error: Server backend enabled but server not running. server_process={self.server_process}, url={self.server_url}"
            yield error_msg, error_msg, "0 tok/s", ""
            return

        try:
            # Build messages in llama.cpp format
            llama_messages = []

            # For text-only models (like GPT-OSS), we need to convert all content to plain text
            # The chat templates for these models expect string content, not list content
            use_text_only = self.is_text_only_model
            is_gpt_oss = self.model_type == "gpt-oss"

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

                    # For GPT-OSS models, the built-in Jinja template handles Harmony formatting
                    # Don't add manual Harmony headers - just pass clean system prompt
                    llama_messages.append({"role": "system", "content": str(content)})

                elif role == "assistant":
                    # For GPT-OSS models, clean harmony tags from previous assistant responses
                    if isinstance(content, str):
                        if is_gpt_oss:
                            content = clean_harmony_tags(content)
                        content = strip_reasoning_for_context(content)
                        llama_messages.append({"role": "assistant", "content": content})
                    elif isinstance(content, list):
                        # Extract text from list content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        text_content = " ".join(text_parts) if text_parts else ""
                        if is_gpt_oss:
                            text_content = clean_harmony_tags(text_content)
                        text_content = strip_reasoning_for_context(text_content)
                        if text_content:
                            llama_messages.append({"role": "assistant", "content": text_content})

                elif role == "user":
                    if isinstance(content, list):
                        if use_text_only:
                            # For text-only models, extract only text content
                            text_parts = []
                            has_images = False
                            has_video = False
                            has_audio = False
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))
                                    elif item.get("type") == "image":
                                        has_images = True
                                    elif item.get("type") == "video":
                                        has_video = True
                                    elif item.get("type") == "audio":
                                        has_audio = True
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
                            if has_audio:
                                final_text = f"[Note: Audio was provided but this model is text-only] {final_text}"

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
                                        # Frame extraction only. llama-cpp-python vendors
                                        # its own llama.cpp build with no input_video
                                        # endpoint, so native video decoding is
                                        # llama-server-only (see generate_via_api).
                                        video_path = item["video"]
                                        if isinstance(video_path, str) and os.path.exists(video_path):
                                            try:
                                                frames = extract_video_frames(video_path, max_frames=video_max_frames, every_other_frame=every_other_frame)
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
                                    elif item.get("type") == "audio" and "audio" in item:
                                        audio_path = item["audio"]
                                        if isinstance(audio_path, str) and os.path.exists(audio_path):
                                            try:
                                                with open(audio_path, "rb") as f:
                                                    b64_audio = base64.b64encode(f.read()).decode("utf-8")
                                                ext = audio_path.split('.')[-1].lower()
                                                if ext not in ['wav', 'mp3']:
                                                    ext = 'wav'
                                                parts.append({
                                                    "type": "input_audio",
                                                    "input_audio": {
                                                        "data": b64_audio,
                                                        "format": ext
                                                    }
                                                })
                                            except Exception as e:
                                                parts.append({
                                                    "type": "text",
                                                    "text": f"[Audio error: {e}]"
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
                print(f"[llama.cpp] GPT-OSS: Building request with {len(llama_messages)} messages, reasoning_effort={reasoning_level}")
                for i, msg in enumerate(llama_messages):
                    content_preview = str(msg.get('content', ''))[:100]
                    # Check for any channel tags that shouldn't be there
                    if '<|channel|>' in str(msg.get('content', '')):
                        print(f"[llama.cpp]   [{i}] {msg['role']}: WARNING - contains <|channel|> tags!")
                        print(f"[llama.cpp]   Full content: {msg.get('content', '')[:500]}")
                    else:
                        print(f"[llama.cpp]   [{i}] {msg['role']}: {content_preview}...")

            # Generate response
            start_time = time.perf_counter()

            # Build chat completion kwargs
            chat_kwargs = {
                "messages": llama_messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature if temperature > 0 else 0.001,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
            }

            if seed is not None and int(seed) >= 0:
                chat_kwargs["seed"] = int(seed)

            # Note: reasoning_effort for GPT-OSS is only supported via llama-server backend
            # The llama-cpp-python library doesn't support chat_template_kwargs
            # The template will use its default "medium" reasoning level
            if is_gpt_oss:
                print(f"[llama.cpp] GPT-OSS: Using template default reasoning (reasoning_level={reasoning_level} only works with llama-server)")

            if not thinking:
                print("[llama.cpp] Note: instant mode needs chat_template_kwargs, which llama-cpp-python")
                print("[llama.cpp]       does not support - using the template default (thinking on)")

            if stream:
                # Streaming mode - yield partial responses
                response_stream = self.model.create_chat_completion(
                    **chat_kwargs,
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

                        # Build context info for local llama-cpp-python (estimate)
                        # For local backend, we estimate based on completion tokens
                        if self.n_ctx > 0:
                            ctx_info = f"~{token_count:,} / {self.n_ctx:,}"
                        else:
                            ctx_info = f"~{token_count:,} tokens"
                        # Yield display_text, raw_text (with thinking formatted), speed, and context
                        yield display_text, raw_for_thinking, f"{tokens_per_sec:.1f} tok/s", ctx_info

                end_time = time.perf_counter()
                generation_time = end_time - start_time
                final_speed = token_count / generation_time if generation_time > 0 else 0

                # Build final context info
                if self.n_ctx > 0:
                    final_ctx_info = f"~{token_count:,} / {self.n_ctx:,}"
                else:
                    final_ctx_info = f"~{token_count:,} tokens"

                print(f"[llama.cpp] Streamed {token_count} tokens in {generation_time:.2f}s ({final_speed:.1f} tok/s)")
                # Yield final stats with total time and context info
                yield display_text, raw_for_thinking, f"{final_speed:.1f} tok/s | {generation_time:.1f}s", final_ctx_info

            else:
                # Non-streaming mode
                response = self.model.create_chat_completion(**chat_kwargs)

                end_time = time.perf_counter()
                generation_time = end_time - start_time

                # Extract response content
                result = response["choices"][0]["message"]["content"]

                # Get token count from response if available
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                if completion_tokens > 0:
                    speed = completion_tokens / generation_time if generation_time > 0 else 0
                else:
                    speed = 0

                # Build context info string
                if self.n_ctx > 0 and total_tokens > 0:
                    ctx_pct = (total_tokens / self.n_ctx) * 100
                    ctx_info = f"{total_tokens:,} / {self.n_ctx:,} ({ctx_pct:.0f}%)"
                elif total_tokens > 0:
                    ctx_info = f"{total_tokens:,} tokens"
                else:
                    ctx_info = ""

                print(f"[llama.cpp] Generated {completion_tokens} tokens in {generation_time:.2f}s ({speed:.1f} tok/s) | ctx: {total_tokens}/{self.n_ctx}")

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

                # Non-streaming mode - yield the final result with speed, time, and context info
                if speed > 0:
                    yield result, result, f"{speed:.1f} tok/s | {generation_time:.1f}s", ctx_info
                else:
                    yield result, result, f"{generation_time:.1f}s", ctx_info
                return

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error during generation: {str(e)}"
            yield error_msg, error_msg, "0 tok/s", ""
            return


# Global manager instance
vlm_manager: Optional[LlamaCppVLM] = None
# Global stop flag for generation
stop_generation: bool = False
# Snapshot of the last chat turn (model content + display history) for Regenerate
last_turn: Optional[Dict[str, Any]] = None

# Default settings file path
SETTINGS_FILE = "vlm_settings.json"
# System prompt presets file path
PROMPTS_FILE = "vlm_prompts.json"
# Per-model settings profiles file path
PROFILES_FILE = "vlm_model_profiles.json"

# Default settings values
DEFAULT_SETTINGS = {
    # Model Settings
    "model_name": None,  # Will use first available if None
    "n_gpu_layers": -1,
    "n_ctx": 32768,
    "backend_type": "llama-cpp-python",
    "tensor_split": "",
    "main_gpu": 0,
    "kv_cache_type": "f16",
    "flash_attn": False,
    "use_mmap": True,
    "use_mlock": False,
    "override_tensor": r"\.ffn_.*_exps\.weight=CPU",
    "extra_args": "",
    "server_port": 8080,
    "llama_server_path": "llama.cpp/build/bin/llama-server",
    # Generation Settings
    "system_prompt": "You are a helpful AI assistant that can understand and describe images and videos in detail.",
    "max_tokens": 8096,
    "temperature": 0.7,
    "top_p": 0.95,
    "repeat_penalty": 1.0,
    "seed": -1,
    "video_max_frames": 8,
    "every_other_frame": False,
    "show_thinking": True,
    # Send previous turns' reasoning back to the model. On is the model's native
    # format and keeps the chain of thought coherent across turns; turn it off to
    # reclaim the thousands of tokens per turn that reasoning costs when n_ctx is tight.
    "keep_reasoning": True,
    # "default" = send nothing and let the model's own default stand. Naming a level
    # here would silently downgrade Muse Glimmer from its native "high".
    "reasoning_level": "default",  # Muse Glimmer strength / GPT-OSS effort
    "thinking_mode": True,  # Kimi K2.6: off = instant mode (llama-server backend only)
    # Batch Caption Settings
    "batch_system_prompt": "You are an image captioning assistant. Provide detailed, accurate descriptions suitable for training image generation models.",
    "batch_prompt": "Describe this image in detail, including the subject, style, composition, colors, lighting, and any notable features.",
}


def save_settings(
    # Model Settings
    model_name: str,
    n_gpu_layers: int,
    n_ctx: int,
    backend_type: str,
    tensor_split: str,
    main_gpu: int,
    kv_cache_type: str,
    flash_attn: bool,
    use_mmap: bool,
    use_mlock: bool,
    override_tensor: str,
    extra_args: str,
    server_port: int,
    llama_server_path: str,
    # Generation Settings
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    seed: int,
    video_max_frames: int,
    every_other_frame: bool,
    show_thinking: bool,
    reasoning_level: str,
    thinking_mode: bool,
    keep_reasoning: bool,
    # Batch Caption Settings
    batch_system_prompt: str,
    batch_prompt: str,
) -> str:
    """Save all settings to a JSON file."""
    settings = {
        # Model Settings
        "model_name": model_name,
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "backend_type": backend_type,
        "tensor_split": tensor_split,
        "main_gpu": main_gpu,
        "kv_cache_type": kv_cache_type,
        "flash_attn": flash_attn,
        "use_mmap": use_mmap,
        "use_mlock": use_mlock,
        "override_tensor": override_tensor,
        "extra_args": extra_args,
        "server_port": server_port,
        "llama_server_path": llama_server_path,
        # Generation Settings
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "seed": seed,
        "video_max_frames": video_max_frames,
        "every_other_frame": every_other_frame,
        "show_thinking": show_thinking,
        "reasoning_level": reasoning_level,
        "thinking_mode": thinking_mode,
        "keep_reasoning": keep_reasoning,
        # Batch Caption Settings
        "batch_system_prompt": batch_system_prompt,
        "batch_prompt": batch_prompt,
    }

    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"[vlm.py] Settings saved to {SETTINGS_FILE}")
        return f"Settings saved to {SETTINGS_FILE}"
    except Exception as e:
        print(f"[vlm.py] Error saving settings: {e}")
        return f"Error saving settings: {e}"


def load_settings() -> Dict[str, Any]:
    """Load settings from JSON file. Returns defaults if file doesn't exist."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            # Merge with defaults to handle missing keys
            merged = {**DEFAULT_SETTINGS, **settings}
            print(f"[vlm.py] Settings loaded from {SETTINGS_FILE}")
            return merged
        else:
            print(f"[vlm.py] No settings file found, using defaults")
            return DEFAULT_SETTINGS.copy()
    except Exception as e:
        print(f"[vlm.py] Error loading settings: {e}, using defaults")
        return DEFAULT_SETTINGS.copy()


def load_prompt_presets() -> Dict[str, str]:
    """Load system prompt presets ({name: prompt}) from JSON file."""
    try:
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            if isinstance(presets, dict):
                return presets
    except Exception as e:
        print(f"[vlm.py] Error loading prompt presets: {e}")
    return {}


def write_prompt_presets(presets: Dict[str, str]) -> None:
    """Write system prompt presets to JSON file."""
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)


def save_prompt_preset_handler(name: str, prompt_text: str):
    """Save the current system prompt under a preset name."""
    name = (name or "").strip()
    if not name:
        return gr.update(), gr.update(), "Enter a preset name before saving"
    try:
        presets = load_prompt_presets()
        presets[name] = prompt_text
        write_prompt_presets(presets)
        choices = sorted(presets)
        print(f"[vlm.py] Saved prompt preset '{name}'")
        return (
            gr.update(choices=choices, value=name),
            gr.update(choices=choices),
            f"Saved preset '{name}'",
        )
    except Exception as e:
        return gr.update(), gr.update(), f"Error saving preset: {e}"


def delete_prompt_preset_handler(name: str):
    """Delete the selected system prompt preset."""
    if not name:
        return gr.update(), gr.update(), "No preset selected"
    try:
        presets = load_prompt_presets()
        if name in presets:
            del presets[name]
            write_prompt_presets(presets)
        choices = sorted(presets)
        return (
            gr.update(choices=choices, value=None),
            gr.update(choices=choices, value=None),
            f"Deleted preset '{name}'",
        )
    except Exception as e:
        return gr.update(), gr.update(), f"Error deleting preset: {e}"


def apply_prompt_preset(name: str):
    """Fill the system prompt textbox from the selected preset."""
    presets = load_prompt_presets()
    if name and name in presets:
        return presets[name], name
    return gr.update(), gr.update()


def apply_batch_prompt_preset(name: str):
    """Fill the batch system prompt textbox from the selected preset."""
    presets = load_prompt_presets()
    if name and name in presets:
        return presets[name]
    return gr.update()


# Per-model settings saved/restored when the model selection changes
PROFILE_KEYS = [
    "n_gpu_layers", "n_ctx", "backend_type", "tensor_split", "main_gpu",
    "kv_cache_type", "flash_attn", "use_mmap", "use_mlock",
    "override_tensor", "extra_args", "server_port",
]


# Starting points for models that need non-default load settings, matched on a
# substring of the model name. Only used when the model has no saved profile yet -
# the first successful load replaces it with whatever actually worked.
RECOMMENDED_PROFILES = {
    "kimi": {
        "n_gpu_layers": 99,
        "n_ctx": 98304,  # unsloth's recommendation; K2.6 supports up to 262144
        "backend_type": "llama-server",
        "kv_cache_type": "f16",
        "flash_attn": True,
        "use_mmap": True,
        "use_mlock": False,
        "override_tensor": r"\.ffn_.*_exps\.weight=CPU",
        "extra_args": "",
    },
    "muse-glimmer": {
        "n_gpu_layers": 99,
        "n_ctx": 32768,  # model supports 131072; KV is cheap here (SWA on 39 of 52
                         # layers, 2 KV heads, head_dim 128 -> ~0.5GB f16 at 32k)
        "backend_type": "llama-server",
        "kv_cache_type": "f16",
        "flash_attn": True,
        "use_mmap": True,
        "use_mlock": False,
        # DENSE model - the default MoE expert-offload pattern matches no tensors
        # here and only produces a confusing -ot on the command line.
        "override_tensor": "",
        # Recommended sampling is temp 1.0 / top_p 0.95 / top_k 64. There is no
        # top_k control in this UI, but the API payload deliberately omits top_k,
        # so this server-side default is what takes effect.
        "extra_args": "--top-k 64",
    },
}


def get_recommended_profile(model_name: str) -> Optional[Dict[str, Any]]:
    """Built-in load settings for a known model family, or None."""
    lowered = (model_name or "").lower()
    for key, profile in RECOMMENDED_PROFILES.items():
        if key in lowered:
            return profile
    return None


def load_model_profiles() -> Dict[str, Dict[str, Any]]:
    """Load per-model settings profiles from JSON file."""
    try:
        if os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            if isinstance(profiles, dict):
                return profiles
    except Exception as e:
        print(f"[vlm.py] Error loading model profiles: {e}")
    return {}


def save_model_profile(model_name: str, profile: Dict[str, Any]) -> None:
    """Save the settings used to load a model, keyed by model name."""
    try:
        profiles = load_model_profiles()
        profiles[model_name] = profile
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        print(f"[vlm.py] Saved settings profile for '{model_name}'")
    except Exception as e:
        print(f"[vlm.py] Error saving model profile: {e}")


def apply_model_profile(model_name: str):
    """Restore saved settings for the selected model, if a profile exists.

    Returns updates for the PROFILE_KEYS components plus the server options
    row visibility.
    """
    profiles = load_model_profiles()
    profile = profiles.get(model_name)
    source = "Restored saved"
    if not profile:
        profile = get_recommended_profile(model_name)
        source = "Applied recommended"
    if not profile:
        return [gr.update()] * (len(PROFILE_KEYS) + 1)

    updates = [
        gr.update(value=profile[key]) if key in profile else gr.update()
        for key in PROFILE_KEYS
    ]
    updates.append(gr.update(visible=(profile.get("backend_type") == "llama-server")))
    print(f"[vlm.py] {source} settings for '{model_name}'")
    return updates


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
    use_mmap: bool,
    use_mlock: bool,
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
        use_mmap,
        use_mlock,
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
    use_mmap: bool,
    use_mlock: bool,
    override_tensor: str,
    extra_args: str,
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
        use_mmap=use_mmap,
        use_mlock=use_mlock,
        override_tensor=override_tensor,
        server_port=server_port,
        extra_args=extra_args,
        progress=progress,
    )


def unload_model_handler():
    """Handle model unloading."""
    if vlm_manager is None:
        return "Manager not initialized"
    status = vlm_manager.unload_model()
    vram = get_vram_info()
    if vram:
        status = f"{status} | VRAM: {vram}"
    return status


def status_handler():
    """Handle status request."""
    if vlm_manager is None:
        return "Manager not initialized"
    status = vlm_manager.get_status()
    vram = get_vram_info()
    if vram:
        status = f"{status} | VRAM: {vram}"
    return status


def extract_text_history(
    history: List[Dict[str, Any]],
    keep_reasoning: bool = True,
) -> List[Dict[str, Any]]:
    """Extract text-only messages from display history for the model.

    keep_reasoning=False drops each previous assistant turn's <think> block before
    resending it. Reasoning is thousands of tokens per turn, so on a small n_ctx a
    few turns fill the window and the final answer gets truncated; on a large
    window keeping it (the default) preserves the model's native transcript.
    """
    messages = []
    if history:
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if role and content:
                    # Extract text content for the model
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        # Extract text from multimodal content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        if not text_parts:
                            continue
                        text = " ".join(text_parts)
                    else:
                        continue
                    if role == "assistant" and not keep_reasoning:
                        text = strip_reasoning_for_context(text)
                        if not text:
                            continue
                    messages.append({"role": role, "content": text})
    return messages


def stream_chat_response(
    messages: List[Dict[str, Any]],
    new_history: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    seed: int,
    video_max_frames: int,
    every_other_frame: bool,
    show_thinking: bool,
    reasoning_level: str,
    thinking_mode: bool = True,
):
    """Stream a generation into the last (assistant) entry of new_history.

    Yields (history, stats, ctx_info) tuples.
    """
    global stop_generation
    stop_generation = False

    stats = ""
    ctx_info = ""
    for display_text, raw_text, stats, ctx_info in vlm_manager.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        seed=seed,
        video_max_frames=video_max_frames,
        every_other_frame=every_other_frame,
        stream=True,
        reasoning_level=reasoning_level,
        thinking=thinking_mode,
    ):
        # Check stop flag
        if stop_generation:
            new_history[-1]["content"] += "\n\n[Generation stopped]"
            stop_generation = False
            yield new_history, stats, ctx_info
            return

        # Show thinking if enabled, otherwise show cleaned text
        if show_thinking:
            # Convert <think> tags to markdown - Gradio strips literal <think>
            # as unknown HTML, blanking the whole message
            new_history[-1]["content"] = format_think_tags_for_display(raw_text)
        else:
            new_history[-1]["content"] = display_text
        yield new_history, stats, ctx_info


# ===== Chat media gallery =====
# One multi-upload dropzone feeds a card grid, replacing the fixed image/video/audio
# slots. The model still takes at most one video and one audio clip per turn.
VLM_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VLM_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
VLM_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
VLM_MEDIA_LIMITS = {"image": 10, "video": 1, "audio": 1}


def vlm_media_kind(path: str) -> Optional[str]:
    """image / video / audio for a dropped file, or None when we cannot send it."""
    ext = os.path.splitext(path)[1].lower()
    if ext in VLM_IMAGE_EXTENSIONS:
        return "image"
    if ext in VLM_AUDIO_EXTENSIONS:
        return "audio"
    if ext in VLM_VIDEO_EXTENSIONS:
        return "video"
    return None


def vlm_media_counts(files) -> Dict[str, int]:
    counts = {"image": 0, "video": 0, "audio": 0}
    for path in (files or []):
        kind = vlm_media_kind(path)
        if kind:
            counts[kind] += 1
    return counts


def vlm_media_preview_html(files) -> str:
    """Render the accumulated media list as a card grid (index badge, kind badge, inline
    preview, per-card remove button wired through the hidden textbox+button)."""
    files = files or []
    if not files:
        return (
            '<div class="vlm-media-empty">No media attached — drop images, a video or '
            'an audio clip into the box above.</div>'
        )
    cards = []
    for i, path in enumerate(files):
        kind = vlm_media_kind(path) or "image"
        url = "/gradio_api/file=" + url_quote(path)
        name = html.escape(os.path.basename(path))
        if kind == "image":
            media = f'<img src="{url}" loading="lazy" alt="{name}">'
        elif kind == "video":
            media = f'<video src="{url}" controls preload="metadata"></video>'
        else:
            media = f'<audio src="{url}" controls preload="metadata"></audio>'
        cards.append(
            f'<div class="vlm-media-card">'
            f'<div class="vlm-media-head">'
            f'<span class="vlm-media-index">{i + 1}</span>'
            f'<span class="vlm-media-kind vlm-media-kind-{kind}">{kind}</span>'
            f'<button type="button" class="vlm-media-remove" title="Remove this file" '
            f'onclick="vlmRemoveMedia({i + 1})">&#10005;</button>'
            f'</div>'
            f'<div class="vlm-media-body">{media}</div>'
            f'<div class="vlm-media-name" title="{name}">{name}</div>'
            f'</div>'
        )
    counts = vlm_media_counts(files)
    summary = (
        f'<div class="vlm-media-summary">'
        f'{counts["image"]}/{VLM_MEDIA_LIMITS["image"]} images &middot; '
        f'{counts["video"]}/{VLM_MEDIA_LIMITS["video"]} video &middot; '
        f'{counts["audio"]}/{VLM_MEDIA_LIMITS["audio"]} audio</div>'
    )
    return summary + '<div class="vlm-media-grid">' + "".join(cards) + "</div>"


def vlm_split_media(media_paths) -> Tuple[List[Image.Image], Optional[str], Optional[str]]:
    """Parse the gallery into the model's content slots: PIL images in drop order,
    plus at most one video and one audio path. Unreadable files are dropped with a warning."""
    images: List[Image.Image] = []
    video = None
    audio = None
    for path in (media_paths or []):
        kind = vlm_media_kind(path)
        if kind == "image":
            try:
                with Image.open(path) as opened:
                    images.append(opened.convert("RGB"))
            except Exception as e:
                gr.Warning(f"Could not read image {os.path.basename(path)}: {e}")
        elif kind == "video" and video is None:
            video = path
        elif kind == "audio" and audio is None:
            audio = path
    return images, video, audio


def chat_handler(
    message: str,
    history: List[Dict[str, Any]],
    system_prompt: str,
    media_paths: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float = 1.0,
    seed: int = -1,
    video_max_frames: int = 8,
    every_other_frame: bool = False,
    show_thinking: bool = False,
    reasoning_level: str = "default",
    thinking_mode: bool = True,
    keep_reasoning: bool = True,
):
    """Handle chat messages from UI with streaming support."""
    # Check if any model is loaded (either local or server mode)
    model_ready = (
        vlm_manager is not None and
        (vlm_manager.model is not None or vlm_manager.use_server_backend)
    )
    if not model_ready:
        error_history = list(history) if history else []
        error_history.append({"role": "user", "content": message})
        error_history.append({"role": "assistant", "content": "Error: No model loaded. Please load a model first."})
        yield error_history, "", "", ""
        return

    # Build messages list for the model
    messages = []

    # Add system prompt if provided
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # Add chat history (handle None case)
    messages.extend(extract_text_history(history, keep_reasoning=keep_reasoning))

    # Build current message content for model
    model_content = []

    # Add all provided images
    images, video, audio = vlm_split_media(media_paths)
    for img in images:
        model_content.append({"type": "image", "image": img})

    if video is not None:
        model_content.append({"type": "video", "video": video})

    if audio is not None:
        model_content.append({"type": "audio", "audio": audio})

    if message.strip():
        model_content.append({"type": "text", "text": message})
    elif not model_content:
        model_content.append({"type": "text", "text": "Describe this image."})

    messages.append({"role": "user", "content": model_content})

    # Build initial display content for chatbot
    new_history = list(history) if history else []

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

    elif audio is not None:
        display_text = message if message else "Describe this audio"
        new_history.append({"role": "user", "content": display_text})
        # Use dict with "path" key for file content in chatbot
        new_history.append({"role": "user", "content": {"path": audio}})

    else:
        new_history.append({"role": "user", "content": message})

    # Add empty assistant message that we'll stream into
    new_history.append({"role": "assistant", "content": ""})

    # Snapshot this turn so Regenerate can replay it (with current settings)
    global last_turn
    base_len = len(history) if history else 0
    last_turn = {
        "model_content": model_content,
        "history_before": list(history) if history else [],
        "display_entries": new_history[base_len:-1],
    }

    # Stream the response
    for streamed_history, stats, ctx_info in stream_chat_response(
        messages, new_history, max_tokens, temperature, top_p, repeat_penalty,
        seed, video_max_frames, every_other_frame, show_thinking, reasoning_level,
        thinking_mode,
    ):
        yield streamed_history, "", stats, ctx_info


def regenerate_handler(
    history: List[Dict[str, Any]],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    seed: int,
    video_max_frames: int = 8,
    every_other_frame: bool = False,
    show_thinking: bool = False,
    reasoning_level: str = "default",
    thinking_mode: bool = True,
    keep_reasoning: bool = True,
):
    """Re-run the last user turn, discarding the previous assistant reply.

    Uses the snapshot saved by chat_handler (which keeps the original images/
    video/audio), but applies the current system prompt and sampling settings.
    """
    model_ready = (
        vlm_manager is not None and
        (vlm_manager.model is not None or vlm_manager.use_server_backend)
    )
    if not model_ready or last_turn is None:
        yield history, "", ""
        return

    # Rebuild model messages with the current system prompt
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(extract_text_history(last_turn["history_before"], keep_reasoning=keep_reasoning))
    messages.append({"role": "user", "content": last_turn["model_content"]})

    # Rebuild display history without the previous assistant reply
    new_history = list(last_turn["history_before"]) + list(last_turn["display_entries"])
    new_history.append({"role": "assistant", "content": ""})

    yield from stream_chat_response(
        messages, new_history, max_tokens, temperature, top_p, repeat_penalty,
        seed, video_max_frames, every_other_frame, show_thinking, reasoning_level,
        thinking_mode,
    )


def edit_last_handler(history: List[Dict[str, Any]]):
    """Remove the last exchange from the chat and return the user text for editing.

    Attached media from that turn is discarded - re-attach before resending.
    """
    global last_turn
    last_turn = None

    history = list(history) if history else []
    if not history:
        return history, ""

    # Drop the trailing assistant reply
    while history and history[-1].get("role") == "assistant":
        history.pop()

    # Drop the trailing user entries (text + media thumbnails), keeping the text
    text = ""
    while history and history[-1].get("role") == "user":
        content = history.pop().get("content")
        if isinstance(content, str) and not text:
            text = content

    return history, text


def clear_chat_handler():
    """Clear chat history."""
    global last_turn
    last_turn = None
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
    repeat_penalty: float = 1.0,
    seed: int = -1,
    video_max_frames: int = 8,
    every_other_frame: bool = False,
    progress=gr.Progress(),
):
    """Process a folder of images and videos and generate captions."""
    model_ready = (
        vlm_manager is not None and
        (vlm_manager.model is not None or vlm_manager.use_server_backend)
    )
    if not model_ready:
        return "Error: No model loaded. Please load a model first."

    if not folder_path or not os.path.isdir(folder_path):
        return f"Error: Invalid folder path: {folder_path}"

    # Supported extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'}

    # Find all media files in folder
    media_files = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            media_files.append((f, 'image'))
        elif ext in video_extensions:
            media_files.append((f, 'video'))

    if not media_files:
        return f"No images or videos found in {folder_path}"

    results = []
    total = len(media_files)

    for i, (filename, media_type) in enumerate(media_files):
        progress((i / total), desc=f"Processing {filename}...")

        file_path = os.path.join(folder_path, filename)
        try:
            # Build messages
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})

            if media_type == 'image':
                # Load image
                img = Image.open(file_path).convert("RGB")
                content = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]
            else:
                # Video - use video type which will be processed by generate()
                content = [
                    {"type": "video", "video": file_path},
                    {"type": "text", "text": prompt},
                ]

            messages.append({"role": "user", "content": content})

            # Generate caption (generate() is a generator, consume it to get the final result)
            caption = ""
            for display_text, raw_text, stats, ctx_info in vlm_manager.generate(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                seed=seed,
                video_max_frames=video_max_frames,
                every_other_frame=every_other_frame,
                stream=False,
            ):
                caption = display_text

            # Check if caption is an error message (don't save errors as captions)
            if caption.startswith("Error"):
                print(f"[batch_caption] Generation failed for {filename}: {caption}")
                results.append(f"[ERROR] {filename}: {caption[:100]}")
            else:
                # Save caption to .txt file
                base_name = os.path.splitext(filename)[0]
                txt_path = os.path.join(folder_path, f"{base_name}.txt")
                print(f"[batch_caption] Saving to: {txt_path}")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                print(f"[batch_caption] Saved successfully, caption length: {len(caption)}")
                results.append(f"[OK] {filename} -> {base_name}.txt")

            # Brief delay between files to allow server memory cleanup (helps with vision encoder fragmentation)
            if media_type == 'video' and i < total - 1:
                time.sleep(1)

        except Exception as e:
            results.append(f"[ERROR] {filename}: {str(e)}")

    progress(1.0, desc="Complete!")
    return f"Processed {total} files:\n\n" + "\n".join(results)


def create_ui():
    """Create the Gradio interface."""
    # Load saved settings
    saved_settings = load_settings()
    preset_choices = sorted(load_prompt_presets())

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
    /* Chat media gallery: one dropzone plus a card grid of what is attached */
    .vlm-hidden { display: none !important; }
    .vlm-media-summary {
        font-size: 0.85em;
        opacity: 0.8;
        margin: 4px 0 6px 2px;
    }
    .vlm-media-empty {
        font-size: 0.85em;
        opacity: 0.6;
        margin: 4px 0 6px 2px;
    }
    .vlm-media-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 10px;
    }
    .vlm-media-card {
        border: 1px solid var(--border-color-primary, #444);
        border-radius: 8px;
        padding: 6px;
        background: var(--background-fill-secondary, rgba(128,128,128,0.05));
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 0;
    }
    .vlm-media-head {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .vlm-media-index {
        font-weight: bold;
        background: #0060df;
        color: white;
        border-radius: 4px;
        padding: 0 6px;
        font-size: 0.85em;
    }
    .vlm-media-kind {
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-radius: 4px;
        padding: 0 5px;
        color: white;
    }
    .vlm-media-kind-image { background: #27ae60; }
    .vlm-media-kind-video { background: #8e44ad; }
    .vlm-media-kind-audio { background: #e67e22; }
    .vlm-media-remove {
        margin-left: auto;
        border: none;
        background: transparent;
        color: var(--body-text-color, inherit);
        opacity: 0.6;
        cursor: pointer;
        font-size: 0.95em;
        line-height: 1;
        padding: 2px 4px;
    }
    .vlm-media-remove:hover { opacity: 1; color: #e74c3c; }
    .vlm-media-body img, .vlm-media-body video {
        width: 100%;
        max-height: 140px;
        object-fit: contain;
        border-radius: 4px;
        display: block;
    }
    .vlm-media-body audio { width: 100%; }
    .vlm-media-name {
        font-size: 0.75em;
        opacity: 0.75;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
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

                # Media gallery - one multi-upload dropzone feeding a card grid
                gr.Markdown(
                    "Drop **multiple files at once** below (or click to browse). New drops are "
                    "**added** to the set; up to **10 images / 1 video / 1 audio clip**. "
                    "Images are sent in the order shown — remove one with its ✕."
                )
                chat_media_state = gr.State(value=[])
                chat_media_files = gr.File(
                    label="Drop images / video / audio here (or click to browse)",
                    file_count="multiple",
                    type="filepath",
                    height=110,
                    elem_id="vlm_media_dropzone",
                )
                chat_media_preview = gr.HTML(value=vlm_media_preview_html([]))
                with gr.Row():
                    chat_media_clear_btn = gr.Button("Clear all media", size="sm")
                # Hidden plumbing for the per-card ✕ buttons in the HTML preview
                # (kept visible=True so they exist in the DOM; hidden via CSS).
                chat_media_remove_idx = gr.Textbox(
                    value="", elem_id="vlm_media_remove_idx",
                    elem_classes=["vlm-hidden"],
                )
                chat_media_remove_btn = gr.Button(
                    "remove", elem_id="vlm_media_remove_btn",
                    elem_classes=["vlm-hidden"],
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
                    regen_btn = gr.Button("Regenerate", scale=1)
                    edit_last_btn = gr.Button("Edit Last", scale=1)
                    stop_btn = gr.Button("Stop", variant="stop", scale=1, elem_classes=["red-btn"])
                    stats_display = gr.Textbox(
                        label="Speed",
                        value="",
                        interactive=False,
                        scale=2,
                    )
                    context_display = gr.Textbox(
                        label="Context",
                        value="",
                        interactive=False,
                        scale=2,
                    )

            # Batch Caption Tab
            with gr.TabItem("Batch Caption"):
                gr.Markdown("Generate captions for all images and videos in a folder. Outputs .txt files with matching names.")

                batch_folder = gr.Textbox(
                    label="Folder Path",
                    placeholder="Enter the full path to folder containing images/videos...",
                    lines=1,
                )

                batch_preset_dropdown = gr.Dropdown(
                    label="Load System Prompt Preset",
                    choices=preset_choices,
                    value=None,
                    info="Presets are managed in Generation Settings below",
                )

                batch_system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="System instructions for captioning...",
                    lines=3,
                    value=saved_settings.get("batch_system_prompt", DEFAULT_SETTINGS["batch_system_prompt"]),
                )

                batch_prompt = gr.Textbox(
                    label="Caption Prompt",
                    placeholder="Describe this image in detail.",
                    lines=2,
                    value=saved_settings.get("batch_prompt", DEFAULT_SETTINGS["batch_prompt"]),
                )

                batch_start_btn = gr.Button("Start Batch Captioning", variant="primary", elem_classes=["green-btn"])

                batch_output = gr.Textbox(
                    label="Output",
                    lines=15,
                    interactive=False,
                )

        # Settings section below chat - in accordions
        # Determine initial model selection from saved settings
        saved_model = saved_settings.get("model_name")
        if saved_model and saved_model in initial_models:
            initial_model_value = saved_model
        else:
            initial_model_value = initial_models[0] if initial_models else None

        # Determine if server options should be visible based on saved backend
        saved_backend = saved_settings.get("backend_type", DEFAULT_SETTINGS["backend_type"])
        server_options_visible = saved_backend == "llama-server"

        with gr.Accordion("Model Settings", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Select Model",
                            choices=initial_models,
                            value=initial_model_value,
                            interactive=True,
                            scale=4,
                        )
                        refresh_models_btn = gr.Button("Refresh", scale=1, min_width=80)

                with gr.Column(scale=1):
                    n_gpu_layers = gr.Slider(
                        minimum=-1,
                        maximum=100,
                        value=saved_settings.get("n_gpu_layers", DEFAULT_SETTINGS["n_gpu_layers"]),
                        step=1,
                        label="GPU Layers (-1 = all)",
                        info="Layers to offload to GPU",
                    )

                with gr.Column(scale=1):
                    n_ctx = gr.Slider(
                        minimum=512,
                        # 262144 is Qwen3.5/3.6/3.8's native window (and Kimi K2.6's
                        # maximum). The old 200000 ceiling could not reach either.
                        maximum=262144,
                        value=saved_settings.get("n_ctx", DEFAULT_SETTINGS["n_ctx"]),
                        step=512,
                        label="Context Length",
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    backend_type = gr.Dropdown(
                        label="Backend",
                        choices=["llama-cpp-python", "llama-server"],
                        value=saved_settings.get("backend_type", DEFAULT_SETTINGS["backend_type"]),
                        info="llama-server enables MoE --override-tensor",
                    )
                with gr.Column(scale=2):
                    tensor_split = gr.Textbox(
                        label="Tensor Split (Multi-GPU)",
                        placeholder="e.g., 2,1 for 48GB+24GB GPUs (ratio-based distribution)",
                        value=saved_settings.get("tensor_split", DEFAULT_SETTINGS["tensor_split"]),
                        info="Comma-separated ratios for distributing layers across GPUs",
                    )
                with gr.Column(scale=1):
                    main_gpu = gr.Slider(
                        minimum=0,
                        maximum=7,
                        value=saved_settings.get("main_gpu", DEFAULT_SETTINGS["main_gpu"]),
                        step=1,
                        label="Main GPU",
                        info="GPU for small tensors/scratch buffer",
                    )
                with gr.Column(scale=1):
                    kv_cache_type = gr.Dropdown(
                        label="KV Cache Type",
                        choices=["f16", "q8_0", "q4_0"],
                        value=saved_settings.get("kv_cache_type", DEFAULT_SETTINGS["kv_cache_type"]),
                        info="Quantize KV cache to save VRAM",
                    )
                with gr.Column(scale=1):
                    flash_attn = gr.Checkbox(
                        label="Flash Attention",
                        value=saved_settings.get("flash_attn", DEFAULT_SETTINGS["flash_attn"]),
                        info="Faster attention (requires layers on GPU)",
                    )
                with gr.Column(scale=1):
                    use_mmap = gr.Checkbox(
                        label="Use MMap",
                        value=saved_settings.get("use_mmap", DEFAULT_SETTINGS["use_mmap"]),
                        info="Uncheck to load fully into RAM (requires enough RAM)",
                    )
                with gr.Column(scale=1):
                    use_mlock = gr.Checkbox(
                        label="Lock in RAM (mlock)",
                        value=saved_settings.get("use_mlock", DEFAULT_SETTINGS["use_mlock"]),
                        info="Lock model in RAM after loading - prevents SSD access after warmup",
                    )

            # Server-mode specific options
            with gr.Row(visible=server_options_visible) as server_options_row:
                with gr.Column(scale=3):
                    override_tensor = gr.Textbox(
                        label="Override Tensor (-ot)",
                        placeholder=r"\.ffn_.*_exps\.weight=CPU",
                        value=saved_settings.get("override_tensor", DEFAULT_SETTINGS["override_tensor"]),
                        info="MoE optimization: offload expert FFN to CPU. Use ; for multiple patterns.",
                    )
                with gr.Column(scale=2):
                    extra_args = gr.Textbox(
                        label="Extra Args",
                        placeholder="--cpu-moe or --n-cpu-moe 82",
                        value=saved_settings.get("extra_args", DEFAULT_SETTINGS["extra_args"]),
                        info="Additional llama-server args (e.g., --cpu-moe, --numa distribute)",
                    )
                with gr.Column(scale=1):
                    server_port = gr.Number(
                        label="Server Port",
                        value=saved_settings.get("server_port", DEFAULT_SETTINGS["server_port"]),
                        precision=0,
                        info="Port for llama-server",
                    )
                with gr.Column(scale=2):
                    llama_server_path = gr.Textbox(
                        label="llama-server Path",
                        placeholder="llama.cpp/build/bin/llama-server",
                        value=saved_settings.get("llama_server_path", DEFAULT_SETTINGS["llama_server_path"]),
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
                prompt_preset_dropdown = gr.Dropdown(
                    label="System Prompt Preset",
                    choices=preset_choices,
                    value=None,
                    scale=2,
                )
                preset_name = gr.Textbox(
                    label="Preset Name",
                    placeholder="Name to save the current prompt as...",
                    scale=2,
                )
                save_preset_btn = gr.Button("Save Preset", scale=1)
                delete_preset_btn = gr.Button("Delete Preset", scale=1, elem_classes=["red-btn"])

            with gr.Row():
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter a system prompt to guide the model's behavior...",
                    lines=2,
                    value=saved_settings.get("system_prompt", DEFAULT_SETTINGS["system_prompt"]),
                    scale=3,
                )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=262048,
                    value=saved_settings.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]),
                    step=64,
                    label="Max New Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=saved_settings.get("temperature", DEFAULT_SETTINGS["temperature"]),
                    step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=saved_settings.get("top_p", DEFAULT_SETTINGS["top_p"]),
                    step=0.05,
                    label="Top P",
                )
                video_max_frames = gr.Slider(
                    minimum=1,
                    maximum=201,
                    value=saved_settings.get("video_max_frames", DEFAULT_SETTINGS["video_max_frames"]),
                    step=1,
                    label="Max Video Frames",
                    info="Frame-extraction fallback only - ignored when the server decodes video natively",
                )
                every_other_frame = gr.Checkbox(
                    label="Every Other Frame",
                    value=saved_settings.get("every_other_frame", DEFAULT_SETTINGS["every_other_frame"]),
                    info="Fallback only - ignored when the server decodes video natively",
                )

            with gr.Row():
                repeat_penalty = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=saved_settings.get("repeat_penalty", DEFAULT_SETTINGS["repeat_penalty"]),
                    step=0.01,
                    label="Repeat Penalty",
                    info="1.0 = disabled",
                )
                seed = gr.Number(
                    label="Seed",
                    value=saved_settings.get("seed", DEFAULT_SETTINGS["seed"]),
                    precision=0,
                    info="-1 = random, >=0 for reproducible output",
                )
                show_thinking = gr.Checkbox(
                    label="Show Thinking",
                    value=saved_settings.get("show_thinking", DEFAULT_SETTINGS["show_thinking"]),
                )
                keep_reasoning = gr.Checkbox(
                    label="Send Reasoning Back",
                    value=saved_settings.get("keep_reasoning", DEFAULT_SETTINGS["keep_reasoning"]),
                    info="Include previous turns' thinking in the prompt. Off reclaims thousands of tokens per turn when n_ctx is tight",
                )
                reasoning_level = gr.Dropdown(
                    label="Reasoning Level",
                    choices=["default", "low", "medium", "high", "xhigh"],
                    value=saved_settings.get("reasoning_level", DEFAULT_SETTINGS["reasoning_level"]),
                    info="Muse Glimmer reasoning strength (xhigh included) / GPT-OSS effort. default = model's own",
                )
                thinking_mode = gr.Checkbox(
                    label="Thinking Mode",
                    value=saved_settings.get("thinking_mode", DEFAULT_SETTINGS["thinking_mode"]),
                    info="Kimi K2.6: off = instant mode (also drop temperature to 0.6). llama-server only. No effect on Muse Glimmer - use Reasoning Level",
                )
                save_settings_btn = gr.Button("Save Settings", variant="secondary")
                save_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    scale=2,
                    show_label=False,
                )

        # Event handlers
        refresh_models_btn.click(
            fn=refresh_models_handler,
            outputs=[model_dropdown],
        )

        def load_model_dispatcher(
            backend, model_name, n_gpu_layers, n_ctx, tensor_split, flash_attn,
            main_gpu, kv_cache_type, use_mmap_val, use_mlock_val, override_tensor, extra_args_val, server_port, llama_server_path,
            progress=gr.Progress()
        ):
            """Route to appropriate load handler based on backend selection."""
            if backend == "llama-server":
                status = load_model_server_handler(
                    model_name, n_gpu_layers, n_ctx, tensor_split, flash_attn,
                    main_gpu, kv_cache_type, use_mmap_val, use_mlock_val, override_tensor, extra_args_val, int(server_port),
                    llama_server_path, progress
                )
            else:
                status = load_model_handler(
                    model_name, n_gpu_layers, n_ctx, tensor_split, flash_attn,
                    main_gpu, kv_cache_type, use_mmap_val, use_mlock_val, progress
                )

            if isinstance(status, str) and not status.startswith("Error"):
                # Remember the settings that successfully loaded this model
                save_model_profile(model_name, {
                    "n_gpu_layers": n_gpu_layers,
                    "n_ctx": n_ctx,
                    "backend_type": backend,
                    "tensor_split": tensor_split,
                    "main_gpu": main_gpu,
                    "kv_cache_type": kv_cache_type,
                    "flash_attn": flash_attn,
                    "use_mmap": use_mmap_val,
                    "use_mlock": use_mlock_val,
                    "override_tensor": override_tensor,
                    "extra_args": extra_args_val,
                    "server_port": int(server_port),
                })
                vram = get_vram_info()
                if vram:
                    status = f"{status} | VRAM: {vram}"

            return status

        load_model_btn.click(
            fn=load_model_dispatcher,
            inputs=[
                backend_type, model_dropdown, n_gpu_layers, n_ctx, tensor_split,
                flash_attn, main_gpu, kv_cache_type, use_mmap, use_mlock, override_tensor, extra_args, server_port,
                llama_server_path
            ],
            outputs=[status_display],
        )

        unload_model_btn.click(
            fn=unload_model_handler,
            outputs=[status_display],
        )

        # The turn being sent is captured into state before the composer is
        # cleared, so the streaming step below never has to write back to the
        # message box or the dropzone - leaving both free to type into and drop
        # onto while the model is still generating.
        pending_msg = gr.State("")
        pending_media = gr.State([])

        def start_send(msg, media):
            """Take the pending turn, then empty the composer: message box,
            media state, dropzone and preview all reset in one shot."""
            return msg, list(media or []), "", [], None, vlm_media_preview_html([])

        def send_message(msg, history, sys_prompt, media, max_tok, temp, top_p_val, rep_pen, seed_val, vid_frames, every_other, thinking, reasoning, think_mode, keep_reason):
            media = list(media or [])

            if not msg.strip() and not media:
                yield history, "", ""
                return

            # Stream responses from chat_handler generator
            for new_history, _, stats, ctx_info in chat_handler(
                msg, history, sys_prompt, media,
                max_tok, temp, top_p_val, rep_pen, seed_val, vid_frames, every_other, thinking, reasoning,
                think_mode, keep_reason
            ):
                yield new_history, stats, ctx_info

        chat_media_outputs = [chat_media_state, chat_media_files, chat_media_preview]

        start_send_outputs = [pending_msg, pending_media, msg_input, *chat_media_outputs]

        send_inputs = [
            pending_msg, chatbot, system_prompt,
            pending_media,
            max_tokens, temperature, top_p, repeat_penalty, seed, video_max_frames, every_other_frame, show_thinking, reasoning_level,
            thinking_mode, keep_reasoning
        ]
        send_outputs = [chatbot, stats_display, context_display]

        for send_trigger in (send_btn.click, msg_input.submit):
            send_trigger(
                fn=start_send,
                inputs=[msg_input, chat_media_state],
                outputs=start_send_outputs,
            ).then(
                fn=send_message,
                inputs=send_inputs,
                outputs=send_outputs,
            )

        def add_chat_media(state_files, uploaded):
            """Append newly dropped files to the accumulated set, then clear the dropzone
            so it stays an always-available drop target.

            Anything the model cannot take - an unknown extension, or a second video or
            audio clip - is rejected here rather than silently ignored at send time.
            """
            files = list(state_files or [])
            counts = vlm_media_counts(files)
            unsupported, over_limit = [], []

            for f in (uploaded or []):
                path = f if isinstance(f, str) else getattr(f, "name", str(f))
                if not path or path in files:
                    continue
                kind = vlm_media_kind(path)
                if kind is None:
                    unsupported.append(os.path.basename(path))
                elif counts[kind] >= VLM_MEDIA_LIMITS[kind]:
                    over_limit.append(f"{os.path.basename(path)} ({kind})")
                else:
                    counts[kind] += 1
                    files.append(path)

            if unsupported:
                gr.Warning("Not an image, video or audio file: " + ", ".join(unsupported))
            if over_limit:
                gr.Warning(
                    "Already at the per-turn limit (%s), skipped: %s" % (
                        ", ".join(f"{v} {k}" for k, v in VLM_MEDIA_LIMITS.items()),
                        ", ".join(over_limit),
                    )
                )
            return files, None, vlm_media_preview_html(files)

        chat_media_files.upload(
            fn=add_chat_media,
            inputs=[chat_media_state, chat_media_files],
            outputs=chat_media_outputs,
        )

        def remove_chat_media(state_files, idx_text):
            files = list(state_files or [])
            try:
                idx = int(str(idx_text).strip())
            except ValueError:
                return files, None, vlm_media_preview_html(files)
            if 1 <= idx <= len(files):
                files.pop(idx - 1)
            return files, None, vlm_media_preview_html(files)

        chat_media_remove_btn.click(
            fn=remove_chat_media,
            inputs=[chat_media_state, chat_media_remove_idx],
            outputs=chat_media_outputs,
        )

        chat_media_clear_btn.click(
            fn=lambda: ([], None, vlm_media_preview_html([])),
            inputs=None,
            outputs=chat_media_outputs,
        )

        demo.load(None, None, None, js=r"""
            () => {
                // A file dropped outside an upload zone would otherwise navigate the
                // browser to the file, wiping out the page. Swallow stray drops at the
                // window level; Gradio's own dropzones handle their drops first.
                window.addEventListener('dragover', (e) => { e.preventDefault(); }, false);
                window.addEventListener('drop', (e) => { e.preventDefault(); }, false);

                // Per-card remove buttons in the media preview: write the 1-based index
                // into the hidden textbox, then click the hidden button.
                window.vlmRemoveMedia = (idx) => {
                    const box = document.querySelector('#vlm_media_remove_idx textarea, #vlm_media_remove_idx input');
                    if (!box) return;
                    box.value = String(idx);
                    box.dispatchEvent(new Event('input', { bubbles: true }));
                    setTimeout(() => {
                        const btn = document.querySelector('#vlm_media_remove_btn button, button#vlm_media_remove_btn, #vlm_media_remove_btn');
                        if (btn) btn.click();
                    }, 60);
                };
            }
        """)

        regen_btn.click(
            fn=regenerate_handler,
            inputs=[
                chatbot, system_prompt, max_tokens, temperature, top_p,
                repeat_penalty, seed, video_max_frames, every_other_frame,
                show_thinking, reasoning_level, thinking_mode, keep_reasoning
            ],
            outputs=[chatbot, stats_display, context_display],
        )

        edit_last_btn.click(
            fn=edit_last_handler,
            inputs=[chatbot],
            outputs=[chatbot, msg_input],
        )

        clear_btn.click(
            fn=clear_chat_handler,
            outputs=[chatbot],
        )

        # System prompt preset handlers
        prompt_preset_dropdown.change(
            fn=apply_prompt_preset,
            inputs=[prompt_preset_dropdown],
            outputs=[system_prompt, preset_name],
        )

        save_preset_btn.click(
            fn=save_prompt_preset_handler,
            inputs=[preset_name, system_prompt],
            outputs=[prompt_preset_dropdown, batch_preset_dropdown, save_status],
        )

        delete_preset_btn.click(
            fn=delete_prompt_preset_handler,
            inputs=[prompt_preset_dropdown],
            outputs=[prompt_preset_dropdown, batch_preset_dropdown, save_status],
        )

        batch_preset_dropdown.change(
            fn=apply_batch_prompt_preset,
            inputs=[batch_preset_dropdown],
            outputs=[batch_system_prompt],
        )

        # Restore per-model settings profile when the model selection changes
        model_dropdown.change(
            fn=apply_model_profile,
            inputs=[model_dropdown],
            outputs=[
                n_gpu_layers, n_ctx, backend_type, tensor_split, main_gpu,
                kv_cache_type, flash_attn, use_mmap, use_mlock,
                override_tensor, extra_args, server_port, server_options_row
            ],
        )

        stop_btn.click(
            fn=stop_generation_handler,
        )

        batch_start_btn.click(
            fn=batch_caption_handler,
            inputs=[
                batch_folder, batch_prompt, batch_system_prompt,
                max_tokens, temperature, repeat_penalty, seed, video_max_frames, every_other_frame
            ],
            outputs=[batch_output],
        )

        # Save settings handler
        save_settings_btn.click(
            fn=save_settings,
            inputs=[
                # Model Settings
                model_dropdown, n_gpu_layers, n_ctx, backend_type,
                tensor_split, main_gpu, kv_cache_type, flash_attn,
                use_mmap, use_mlock, override_tensor, extra_args,
                server_port, llama_server_path,
                # Generation Settings
                system_prompt, max_tokens, temperature, top_p, repeat_penalty,
                seed, video_max_frames, every_other_frame, show_thinking, reasoning_level,
                thinking_mode, keep_reasoning,
                # Batch Caption Settings
                batch_system_prompt, batch_prompt
            ],
            outputs=[save_status],
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
            draft = " [+draft]" if m.get("draft_path") else ""
            print(f"  - {m['name']}{vision}{draft}")
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
