#!/usr/bin/env python3
"""
Convert Qwen3.5-397B-A17B HF bf16 safetensors → GGUF Q4_K_M.

Reads bf16 weights directly from safetensors, quantizes with CUDA acceleration,
and writes Q4_K_M GGUF. No intermediate files. No dependency on llama-quantize
or convert_hf_to_gguf.py.

Usage:
    python convert_hf_to_q4km.py /path/to/Qwen3.5-397B-A17B --output Qwen3.5-397B-A17B-Q4_K_M.gguf
"""

import argparse
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf
from gguf import GGUFWriter, GGMLQuantizationType
import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_float, POINTER

# ---------------------------------------------------------------------------
# Load ggml library for native quantization
# ---------------------------------------------------------------------------

def load_ggml_lib():
    """Load libggml for native quantization functions.

    Set GGML_LIB_PATH environment variable to specify the library path.
    Example: export GGML_LIB_PATH=/path/to/llama.cpp/build/bin
    """
    import os
    import glob

    search_patterns = []

    # Check environment variable first
    env_path = os.environ.get('GGML_LIB_PATH')
    if env_path:
        search_patterns.append(os.path.join(env_path, "libggml*.so*"))

    # Try common relative paths
    search_patterns.extend([
        "llama.cpp/build/bin/libggml*.so*",
        "llama.cpp/build/ggml/src/libggml*.so*",
        "llama.cpp/build/src/libggml*.so*",
        "llama.cpp/build/lib/libggml*.so*",
        "llama.cpp/build/libggml*.so*",
        "/usr/local/lib/libggml*.so*",
        "/usr/lib/libggml*.so*",
    ])

    all_paths = []
    for pattern in search_patterns:
        all_paths.extend(glob.glob(pattern))

    # Prefer libggml-base.so as it contains the quant functions
    all_paths.sort(key=lambda x: (0 if 'base' in x else 1, x))

    for path in all_paths:
        try:
            lib = ctypes.CDLL(path)
            # Check if it has the quantization function
            if hasattr(lib, 'ggml_quantize_q4_K'):
                print(f"  Loaded ggml library: {path}")
                return lib
        except (OSError, AttributeError):
            continue

    return None

GGML_LIB = None

def init_ggml():
    """Initialize ggml library and set up function signatures."""
    global GGML_LIB

    GGML_LIB = load_ggml_lib()
    if GGML_LIB is None:
        print("  [WARN] Could not load libggml.so, falling back to Python quantization")
        return False

    try:
        # size_t ggml_quantize_q4_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix)
        GGML_LIB.ggml_quantize_q4_K.argtypes = [
            POINTER(c_float),  # src
            c_void_p,          # dst
            ctypes.c_int64,    # nrows
            ctypes.c_int64,    # n_per_row
            c_void_p,          # imatrix (can be NULL)
        ]
        GGML_LIB.ggml_quantize_q4_K.restype = c_size_t

        GGML_LIB.ggml_quantize_q6_K.argtypes = [
            POINTER(c_float),
            c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            c_void_p,
        ]
        GGML_LIB.ggml_quantize_q6_K.restype = c_size_t

        return True
    except AttributeError as e:
        print(f"  [WARN] ggml library missing quantization functions: {e}")
        GGML_LIB = None
        return False

def ggml_quantize_q4_k(data: np.ndarray) -> np.ndarray:
    """Quantize float32 data to Q4_K using ggml library."""
    assert data.dtype == np.float32

    # Ensure contiguous
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    # Flatten to 2D: [nrows, n_per_row]
    original_shape = data.shape
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        data = data.reshape(-1, data.shape[-1])

    nrows, n_per_row = data.shape

    # Q4_K: 144 bytes per 256 elements
    assert n_per_row % QK_K == 0, f"n_per_row ({n_per_row}) must be multiple of {QK_K}"
    n_blocks = n_per_row // QK_K
    output_bytes = nrows * n_blocks * Q4_K_BLOCK_SIZE

    output = np.zeros(output_bytes, dtype=np.uint8)

    # Call ggml
    src_ptr = data.ctypes.data_as(POINTER(c_float))
    dst_ptr = output.ctypes.data_as(c_void_p)

    GGML_LIB.ggml_quantize_q4_K(src_ptr, dst_ptr, nrows, n_per_row, None)

    return output

def ggml_quantize_q6_k(data: np.ndarray) -> np.ndarray:
    """Quantize float32 data to Q6_K using ggml library."""
    assert data.dtype == np.float32

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    original_shape = data.shape
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        data = data.reshape(-1, data.shape[-1])

    nrows, n_per_row = data.shape

    # Q6_K: 210 bytes per 256 elements
    assert n_per_row % QK_K == 0
    n_blocks = n_per_row // QK_K
    output_bytes = nrows * n_blocks * Q6_K_BLOCK_SIZE

    output = np.zeros(output_bytes, dtype=np.uint8)

    src_ptr = data.ctypes.data_as(POINTER(c_float))
    dst_ptr = output.ctypes.data_as(c_void_p)

    GGML_LIB.ggml_quantize_q6_K(src_ptr, dst_ptr, nrows, n_per_row, None)

    return output

# ---------------------------------------------------------------------------
# Native Q4_K / Q6_K quantization (gguf library doesn't implement K-quants)
# ---------------------------------------------------------------------------

QK_K = 256  # Super-block size for K-quants
Q4_K_BLOCK_SIZE = 144  # bytes per Q4_K block
Q6_K_BLOCK_SIZE = 210  # bytes per Q6_K block


def quantize_q4_k_rows(data: np.ndarray) -> np.ndarray:
    """Quantize float32 data to Q4_K format, row by row.

    Input: [..., n_cols] float32 array where n_cols must be multiple of 256
    Output: [..., n_cols//256 * 144] uint8 array

    Q4_K block structure (144 bytes per 256 elements):
    - d: float16 (2 bytes) - super-block scale
    - dmin: float16 (2 bytes) - super-block minimum
    - scales: uint8[12] - packed 6-bit scales/mins for 8 sub-blocks
    - qs: uint8[128] - 4-bit quantized values (2 per byte)
    """
    assert data.dtype == np.float32
    original_shape = data.shape

    # Handle 1D case
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        # Flatten all but last dimension
        data = data.reshape(-1, data.shape[-1])

    n_rows, n_cols = data.shape
    assert n_cols % QK_K == 0, f"Columns ({n_cols}) must be multiple of {QK_K}"

    n_blocks_per_row = n_cols // QK_K
    bytes_per_row = n_blocks_per_row * Q4_K_BLOCK_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(data).to(device)

    # Reshape to [n_rows, n_blocks_per_row, QK_K]
    tensor = tensor.view(n_rows, n_blocks_per_row, QK_K)

    # Reshape to [n_rows, n_blocks_per_row, 8, 32] for sub-blocks
    sub_blocks = tensor.view(n_rows, n_blocks_per_row, 8, 32)

    # Find min/max per sub-block (vectorized)
    sb_mins = sub_blocks.min(dim=3).values  # [n_rows, n_blocks, 8]
    sb_maxs = sub_blocks.max(dim=3).values

    # Compute scales per sub-block
    scales = (sb_maxs - sb_mins) / 15.0
    scales = torch.where(sb_maxs != sb_mins, scales, torch.zeros_like(scales))
    mins = sb_mins

    # Super-block scales
    max_scale = scales.abs().max(dim=2).values  # [n_rows, n_blocks]
    max_min = mins.abs().max(dim=2).values

    d = max_scale / 63.0
    dmin = max_min / 63.0

    # Avoid division by zero
    inv_d = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    inv_dmin = torch.where(dmin > 0, 1.0 / dmin, torch.zeros_like(dmin))

    # Quantize scales/mins to 6-bit
    q_scales = (scales * inv_d.unsqueeze(2)).round().clamp(0, 63).to(torch.uint8)
    q_mins = ((-mins) * inv_dmin.unsqueeze(2)).round().clamp(0, 63).to(torch.uint8)

    # Quantize values to 4-bit
    eff_scales = (d.unsqueeze(2) * q_scales.float()).unsqueeze(3)
    eff_mins = (dmin.unsqueeze(2) * q_mins.float()).unsqueeze(3)
    inv_eff_scales = torch.where(eff_scales > 0, 1.0 / eff_scales, torch.zeros_like(eff_scales))
    qs = ((sub_blocks + eff_mins) * inv_eff_scales).round().clamp(0, 15).to(torch.uint8)

    # Move to CPU for packing
    d_cpu = d.half().cpu().numpy()
    dmin_cpu = dmin.half().cpu().numpy()
    q_scales_cpu = q_scales.cpu().numpy()
    q_mins_cpu = q_mins.cpu().numpy()
    qs_cpu = qs.cpu().numpy().reshape(n_rows, n_blocks_per_row, 256)

    # Pack into output [n_rows, bytes_per_row]
    output = np.zeros((n_rows, bytes_per_row), dtype=np.uint8)

    for blk in range(n_blocks_per_row):
        offset = blk * Q4_K_BLOCK_SIZE

        # d and dmin (bytes 0-4)
        output[:, offset:offset+2] = np.ascontiguousarray(d_cpu[:, blk]).view(np.uint8).reshape(n_rows, 2)
        output[:, offset+2:offset+4] = np.ascontiguousarray(dmin_cpu[:, blk]).view(np.uint8).reshape(n_rows, 2)

        # Pack scales (6-bit) into 12 bytes (bytes 4-16)
        for i in range(4):
            output[:, offset+4+i] = (q_scales_cpu[:, blk, i] & 0xF) | ((q_scales_cpu[:, blk, i+4] & 0xF) << 4)
        for i in range(4):
            output[:, offset+8+i] = (q_mins_cpu[:, blk, i] & 0xF) | ((q_mins_cpu[:, blk, i+4] & 0xF) << 4)
        for i in range(4):
            output[:, offset+12+i] = ((q_scales_cpu[:, blk, i] >> 4) & 0x3) | \
                                     (((q_scales_cpu[:, blk, i+4] >> 4) & 0x3) << 2) | \
                                     (((q_mins_cpu[:, blk, i] >> 4) & 0x3) << 4) | \
                                     (((q_mins_cpu[:, blk, i+4] >> 4) & 0x3) << 6)

        # Pack quantized values (4-bit) into 128 bytes (bytes 16-144)
        for i in range(128):
            output[:, offset+16+i] = (qs_cpu[:, blk, i*2] & 0xF) | ((qs_cpu[:, blk, i*2+1] & 0xF) << 4)

    if len(original_shape) == 1:
        return output.ravel()
    return output


def quantize_q6_k_rows(data: np.ndarray) -> np.ndarray:
    """Quantize float32 data to Q6_K format, row by row.

    Input: [..., n_cols] float32 array where n_cols must be multiple of 256
    Output: [..., n_cols//256 * 210] uint8 array

    Q6_K block structure (210 bytes per 256 elements):
    - ql: uint8[128] - lower 4 bits of 6-bit quants
    - qh: uint8[64] - upper 2 bits of 6-bit quants
    - scales: int8[16] - scales for 16 sub-blocks of 16 elements
    - d: float16 (2 bytes) - super-block scale
    """
    assert data.dtype == np.float32
    original_shape = data.shape

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        # Flatten all but last dimension
        data = data.reshape(-1, data.shape[-1])

    n_rows, n_cols = data.shape
    assert n_cols % QK_K == 0, f"Columns ({n_cols}) must be multiple of {QK_K}"

    n_blocks_per_row = n_cols // QK_K
    bytes_per_row = n_blocks_per_row * Q6_K_BLOCK_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(data).to(device)

    # Reshape to [n_rows, n_blocks_per_row, 16, 16]
    sub_blocks = tensor.view(n_rows, n_blocks_per_row, 16, 16)

    # Find max abs per sub-block (symmetric quantization)
    max_abs = sub_blocks.abs().max(dim=3).values  # [n_rows, n_blocks, 16]
    scales = max_abs / 31.0

    # Super-block scale
    max_scale = scales.abs().max(dim=2).values  # [n_rows, n_blocks]
    d = max_scale / 127.0

    inv_d = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))

    # Quantize scales to 8-bit signed
    q_scales = (scales * inv_d.unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)

    # Quantize values
    eff_scales = (d.unsqueeze(2) * q_scales.float()).unsqueeze(3)
    inv_eff_scales = torch.where(eff_scales > 0, 1.0 / eff_scales, torch.zeros_like(eff_scales))
    qs = (sub_blocks * inv_eff_scales).round().clamp(-32, 31).to(torch.int8) + 32
    qs = qs.to(torch.uint8)

    # Move to CPU
    d_cpu = d.half().cpu().numpy()
    q_scales_cpu = q_scales.cpu().numpy()
    qs_cpu = qs.cpu().numpy().reshape(n_rows, n_blocks_per_row, 256)

    # Pack into output
    output = np.zeros((n_rows, bytes_per_row), dtype=np.uint8)

    for blk in range(n_blocks_per_row):
        offset = blk * Q6_K_BLOCK_SIZE

        # ql: lower 4 bits (128 bytes)
        for i in range(128):
            output[:, offset+i] = (qs_cpu[:, blk, i*2] & 0xF) | ((qs_cpu[:, blk, i*2+1] & 0xF) << 4)

        # qh: upper 2 bits (64 bytes)
        for i in range(256):
            qh_byte = i // 4
            qh_shift = (i % 4) * 2
            output[:, offset+128+qh_byte] |= ((qs_cpu[:, blk, i] >> 4) & 0x3) << qh_shift

        # scales (16 bytes)
        output[:, offset+192:offset+208] = np.ascontiguousarray(q_scales_cpu[:, blk]).view(np.uint8).reshape(n_rows, 16)

        # d as float16 (2 bytes)
        output[:, offset+208:offset+210] = np.ascontiguousarray(d_cpu[:, blk]).view(np.uint8).reshape(n_rows, 2)

    if len(original_shape) == 1:
        return output.ravel()
    return output


def pad_to_quantize(data: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Pad tensor so last dimension is multiple of QK_K (256).

    Returns (padded_data, original_shape).
    """
    original_shape = data.shape

    if data.ndim == 1:
        n_cols = data.shape[0]
        pad_cols = (QK_K - (n_cols % QK_K)) % QK_K
        if pad_cols > 0:
            data = np.pad(data, (0, pad_cols), mode='constant')
        return data, original_shape

    # 2D or higher: pad last dimension
    n_cols = data.shape[-1]
    pad_cols = (QK_K - (n_cols % QK_K)) % QK_K
    if pad_cols > 0:
        pad_width = [(0, 0)] * (data.ndim - 1) + [(0, pad_cols)]
        data = np.pad(data, pad_width, mode='constant')

    return data, original_shape


# Maximum rows to process at once (tune based on GPU memory)
# For 32GB GPU with 512 experts, use larger chunks for better GPU utilization
# 128K rows of 4096 elements = 2GB per chunk, fits well in 32GB VRAM
MAX_ROWS_PER_CHUNK = 131072


def quantize_tensor_chunked(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    """Quantize a large tensor in chunks to avoid OOM.

    Processes MAX_ROWS_PER_CHUNK rows at a time.
    """
    # Pad to block boundary
    padded_data, original_shape = pad_to_quantize(data)

    # Flatten to 2D for chunked processing
    if padded_data.ndim == 1:
        padded_data = padded_data.reshape(1, -1)
    elif padded_data.ndim > 2:
        padded_data = padded_data.reshape(-1, padded_data.shape[-1])

    n_rows, n_cols = padded_data.shape
    n_blocks_per_row = n_cols // QK_K

    if qtype == GGMLQuantizationType.Q4_K:
        bytes_per_row = n_blocks_per_row * Q4_K_BLOCK_SIZE
        quant_func = quantize_q4_k_rows
    elif qtype == GGMLQuantizationType.Q6_K:
        bytes_per_row = n_blocks_per_row * Q6_K_BLOCK_SIZE
        quant_func = quantize_q6_k_rows
    else:
        raise ValueError(f"Unsupported quantization type for chunked: {qtype}")

    # Process in chunks
    output_chunks = []
    for start_row in range(0, n_rows, MAX_ROWS_PER_CHUNK):
        end_row = min(start_row + MAX_ROWS_PER_CHUNK, n_rows)
        chunk = padded_data[start_row:end_row]

        # Quantize chunk
        quantized_chunk = quant_func(chunk)
        output_chunks.append(quantized_chunk)

        # Clear CUDA cache between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all chunks
    return np.concatenate(output_chunks, axis=0)


def quantize_tensor(data: np.ndarray | torch.Tensor, qtype: GGMLQuantizationType) -> np.ndarray:
    """Quantize a tensor using the specified quantization type.

    Uses native ggml library when available (fastest).
    Falls back to CUDA-accelerated Python implementation.
    Handles padding to block boundaries automatically.
    """
    # Convert torch tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy()

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    # Pad to block boundary
    padded_data, original_shape = pad_to_quantize(data)

    # Use native ggml library if available (much faster)
    if GGML_LIB is not None:
        if qtype == GGMLQuantizationType.Q4_K:
            return ggml_quantize_q4_k(padded_data)
        elif qtype == GGMLQuantizationType.Q6_K:
            return ggml_quantize_q6_k(padded_data)

    # Fall back to Python implementation
    if qtype == GGMLQuantizationType.Q4_K:
        return quantize_q4_k_rows(padded_data)
    elif qtype == GGMLQuantizationType.Q6_K:
        return quantize_q6_k_rows(padded_data)
    else:
        return gguf.quants.quantize(data, qtype)

# ---------------------------------------------------------------------------
# HF → GGUF tensor name mapping for qwen35moe
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    """Map a HuggingFace tensor name to its GGUF equivalent.

    Returns None if the tensor should be skipped (e.g. MTP layers).
    """
    # Strip common prefixes
    name = hf_name

    # --- Top-level ---
    if name == "lm_head.weight":
        return "output.weight"
    if name == "model.language_model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.language_model.norm.weight":
        return "output_norm.weight"

    # --- Vision encoder ---
    m = re.match(r"model\.visual\.patch_embed\.proj\.(weight|bias)", name)
    if m:
        return f"v.patch_embd.{m.group(1)}"

    if name == "model.visual.pos_embed.weight":
        return "v.position_embd.weight"

    m = re.match(r"model\.visual\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)", name)
    if m:
        return f"v.blk.{m.group(1)}.attn_qkv.{m.group(2)}"

    m = re.match(r"model\.visual\.blocks\.(\d+)\.attn\.proj\.(weight|bias)", name)
    if m:
        return f"v.blk.{m.group(1)}.attn_out.{m.group(2)}"

    m = re.match(r"model\.visual\.blocks\.(\d+)\.norm1\.(weight|bias)", name)
    if m:
        return f"v.blk.{m.group(1)}.ln1.{m.group(2)}"

    m = re.match(r"model\.visual\.blocks\.(\d+)\.norm2\.(weight|bias)", name)
    if m:
        return f"v.blk.{m.group(1)}.ln2.{m.group(2)}"

    m = re.match(r"model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.(weight|bias)", name)
    if m:
        return f"v.blk.{m.group(1)}.ffn_up.{m.group(2)}"

    m = re.match(r"model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.(weight|bias)", name)
    if m:
        return f"v.blk.{m.group(1)}.ffn_down.{m.group(2)}"

    m = re.match(r"model\.visual\.merger\.norm\.(weight|bias)", name)
    if m:
        return f"v.merger.ln.{m.group(1)}"

    m = re.match(r"model\.visual\.merger\.linear_fc1\.(weight|bias)", name)
    if m:
        return f"v.merger.fc1.{m.group(1)}"

    m = re.match(r"model\.visual\.merger\.linear_fc2\.(weight|bias)", name)
    if m:
        return f"v.merger.fc2.{m.group(1)}"

    # --- Language model layers ---
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", name)
    if not m:
        print(f"  [WARN] Unmapped tensor: {hf_name}")
        return None

    layer = m.group(1)
    rest = m.group(2)

    # Norms
    if rest == "input_layernorm.weight":
        return f"blk.{layer}.attn_norm.weight"
    if rest == "post_attention_layernorm.weight":
        return f"blk.{layer}.ffn_norm.weight"

    # Full attention (self_attn)
    if rest == "self_attn.q_proj.weight":
        return f"blk.{layer}.attn_q.weight"
    if rest == "self_attn.k_proj.weight":
        return f"blk.{layer}.attn_k.weight"
    if rest == "self_attn.v_proj.weight":
        return f"blk.{layer}.attn_v.weight"
    if rest == "self_attn.o_proj.weight":
        return f"blk.{layer}.attn_output.weight"
    if rest == "self_attn.q_norm.weight":
        return f"blk.{layer}.attn_q_norm.weight"
    if rest == "self_attn.k_norm.weight":
        return f"blk.{layer}.attn_k_norm.weight"

    # Linear attention (Gated Delta Network / SSM-like)
    if rest == "linear_attn.in_proj_qkv.weight":
        return f"blk.{layer}.ssm_in.weight"
    if rest == "linear_attn.in_proj_a.weight":
        return f"blk.{layer}.ssm_a.weight"
    if rest == "linear_attn.in_proj_b.weight":
        return f"blk.{layer}.ssm_b.weight"
    if rest == "linear_attn.in_proj_z.weight":
        return f"blk.{layer}.ssm_z.weight"
    if rest == "linear_attn.conv1d.weight":
        return f"blk.{layer}.ssm_conv1d.weight"
    if rest == "linear_attn.dt_bias":
        return f"blk.{layer}.ssm_dt.bias"
    if rest == "linear_attn.A_log":
        return f"blk.{layer}.ssm_a_log"
    if rest == "linear_attn.norm.weight":
        return f"blk.{layer}.ssm_norm.weight"
    if rest == "linear_attn.out_proj.weight":
        return f"blk.{layer}.ssm_out.weight"

    # MoE router
    if rest == "mlp.gate.weight":
        return f"blk.{layer}.ffn_gate_inp.weight"

    # Shared expert gate
    if rest == "mlp.shared_expert_gate.weight":
        return f"blk.{layer}.ffn_gate_inp_shexp.weight"

    # Shared expert projections
    if rest == "mlp.shared_expert.gate_proj.weight":
        return f"blk.{layer}.ffn_gate_shexp.weight"
    if rest == "mlp.shared_expert.up_proj.weight":
        return f"blk.{layer}.ffn_up_shexp.weight"
    if rest == "mlp.shared_expert.down_proj.weight":
        return f"blk.{layer}.ffn_down_shexp.weight"

    # Individual expert weights - these get merged per-layer later
    m2 = re.match(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", rest)
    if m2:
        # Return a special name; caller handles merging
        return f"blk.{layer}.expert.{m2.group(1)}.{m2.group(2)}.weight"

    print(f"  [WARN] Unmapped tensor: {hf_name}")
    return None


# ---------------------------------------------------------------------------
# Quantization type selection (Q4_K_M strategy)
# ---------------------------------------------------------------------------

def choose_quant_type(
    gguf_name: str,
    shape: tuple,
    num_layers: int,
) -> GGMLQuantizationType:
    """Pick the quantization type for a tensor following Q4_K_M conventions."""

    # Vision tensors → F16
    if gguf_name.startswith("v."):
        return GGMLQuantizationType.F16

    # 1-D tensors (norms, biases) → F32
    if len(shape) == 1:
        return GGMLQuantizationType.F32

    # Biases → F32
    if gguf_name.endswith(".bias"):
        return GGMLQuantizationType.F32

    # Embedding and output head → Q6_K
    if gguf_name in ("token_embd.weight", "output.weight"):
        return GGMLQuantizationType.Q6_K

    # Output norm → F32 (1-D, already caught above, but be explicit)
    if gguf_name == "output_norm.weight":
        return GGMLQuantizationType.F32

    # ALL SSM/linear attention tensors → F32
    # These have small dimensions (64, 128, etc.) that don't fit K-quant 256-element blocks
    # Includes: ssm_in, ssm_a, ssm_b, ssm_z, ssm_conv1d, ssm_dt, ssm_a_log, ssm_norm, ssm_out
    if "ssm_" in gguf_name:
        return GGMLQuantizationType.F32

    # Extract block number for first/last layer heuristic
    m = re.match(r"blk\.(\d+)\.", gguf_name)
    if m:
        blk = int(m.group(1))
        is_edge = (blk == 0) or (blk == num_layers - 1)
    else:
        is_edge = False

    # Attention norms → F32 (1-D, caught above)
    # Q/K norms → F32 (1-D, caught above)

    # Router weights → F32 (small, critical for routing)
    if "ffn_gate_inp" in gguf_name and "shexp" not in gguf_name:
        return GGMLQuantizationType.F32

    # Shared expert gate → F32 (1-D, already caught)

    # Expert weights always use Q4_K (too large for Q6_K with 512 experts)
    if "ffn_gate_exps" in gguf_name or "ffn_up_exps" in gguf_name or "ffn_down_exps" in gguf_name:
        return GGMLQuantizationType.Q4_K

    # Tensors with dimensions not divisible by 256 → F32 (can't use K-quants)
    if len(shape) >= 2 and shape[-1] % QK_K != 0:
        return GGMLQuantizationType.F32

    # First and last block non-expert weights → Q6_K
    if is_edge:
        return GGMLQuantizationType.Q6_K

    # Everything else (attention projections, shared expert weights) → Q4_K
    return GGMLQuantizationType.Q4_K


# ---------------------------------------------------------------------------
# CUDA-accelerated bf16 → f32 conversion
# ---------------------------------------------------------------------------

def bf16_tensor_to_f32_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a bf16 torch tensor to f32 numpy via CUDA if available."""
    if torch.cuda.is_available():
        # Direct GPU conversion - much faster than CPU
        tensor = tensor.to(device='cuda', non_blocking=True).float().cpu()
    else:
        tensor = tensor.float()
    return tensor.numpy()


def bf16_tensor_to_f16_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to f16 numpy via CUDA."""
    if torch.cuda.is_available():
        tensor = tensor.to(device='cuda', non_blocking=True).half().cpu()
    else:
        tensor = tensor.half()
    return tensor.numpy()


def bf16_tensor_to_f32_gpu(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a bf16 torch tensor to f32 on GPU (stays on GPU)."""
    if torch.cuda.is_available():
        return tensor.to(device='cuda', non_blocking=True).float()
    else:
        return tensor.float()


# ---------------------------------------------------------------------------
# Tokenizer writing
# ---------------------------------------------------------------------------

def write_tokenizer(writer: GGUFWriter, model_dir: Path):
    """Write tokenizer data to GGUF from tokenizer.json."""
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    if not tokenizer_path.exists():
        print("[WARN] tokenizer.json not found, skipping tokenizer")
        return

    with open(tokenizer_path, "r") as f:
        tokenizer = json.load(f)

    with open(tokenizer_config_path, "r") as f:
        tokenizer_config = json.load(f)

    # BPE model
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")

    model_data = tokenizer.get("model", {})
    vocab = model_data.get("vocab", {})

    # Build token list and scores
    n_vocab = max(vocab.values()) + 1 if vocab else 0
    tokens = [""] * n_vocab
    scores = [0.0] * n_vocab
    toktypes = [0] * n_vocab  # 0 = normal

    for token_str, token_id in vocab.items():
        if token_id < n_vocab:
            tokens[token_id] = token_str.encode("utf-8", errors="replace")
            scores[token_id] = -float(token_id)  # BPE: lower id = higher priority

    # Added tokens (special tokens)
    added_tokens = tokenizer.get("added_tokens", [])
    for at in added_tokens:
        tid = at["id"]
        if tid < len(tokens):
            tokens[tid] = at["content"].encode("utf-8", errors="replace")
            if at.get("special", False):
                toktypes[tid] = 3  # control token

    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    # Merges
    merges = model_data.get("merges", [])
    if merges:
        # Merges can be strings ("a b") or lists (["a", "b"])
        encoded = []
        for m in merges:
            if isinstance(m, list):
                encoded.append(" ".join(m).encode("utf-8"))
            else:
                encoded.append(m.encode("utf-8"))
        writer.add_token_merges(encoded)

    # Special token IDs
    eos_token = tokenizer_config.get("eos_token")
    pad_token = tokenizer_config.get("pad_token")

    # Find token IDs by content
    token_to_id = {at["content"]: at["id"] for at in added_tokens}

    if eos_token and eos_token in token_to_id:
        writer.add_eos_token_id(token_to_id[eos_token])
    if pad_token and pad_token in token_to_id:
        writer.add_pad_token_id(token_to_id[pad_token])

    # Chat template
    chat_template_path = model_dir / "chat_template.jinja"
    if chat_template_path.exists():
        with open(chat_template_path, "r") as f:
            writer.add_chat_template(f.read())

    print(f"  Tokenizer: {n_vocab} tokens, {len(merges)} merges")


# ---------------------------------------------------------------------------
# Metadata writing
# ---------------------------------------------------------------------------

def write_metadata(writer: GGUFWriter, config: dict):
    """Write model metadata to GGUF header."""
    text_cfg = config.get("text_config", {})
    vision_cfg = config.get("vision_config", {})

    writer.add_name("Qwen3.5-397B-A17B")
    writer.add_architecture()
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_K_M)

    # Text model params
    writer.add_context_length(text_cfg.get("max_position_embeddings", 262144))
    writer.add_embedding_length(text_cfg.get("hidden_size", 4096))
    writer.add_block_count(text_cfg.get("num_hidden_layers", 60))
    writer.add_feed_forward_length(text_cfg.get("moe_intermediate_size", 1024))
    writer.add_head_count(text_cfg.get("num_attention_heads", 32))
    writer.add_head_count_kv(text_cfg.get("num_key_value_heads", 2))
    writer.add_key_length(text_cfg.get("head_dim", 256))
    writer.add_value_length(text_cfg.get("head_dim", 256))
    writer.add_layer_norm_rms_eps(text_cfg.get("rms_norm_eps", 1e-6))
    writer.add_vocab_size(text_cfg.get("vocab_size", 248320))

    # MoE params
    writer.add_expert_count(text_cfg.get("num_experts", 512))
    writer.add_expert_used_count(text_cfg.get("num_experts_per_tok", 10))
    writer.add_expert_shared_count(1)  # 1 shared expert
    writer.add_expert_shared_feed_forward_length(
        text_cfg.get("shared_expert_intermediate_size", 1024)
    )

    # RoPE
    rope_params = text_cfg.get("rope_parameters", {})
    writer.add_rope_freq_base(rope_params.get("rope_theta", 10000000))
    rope_sections = rope_params.get("mrope_section")
    if rope_sections:
        writer.add_rope_dimension_sections(rope_sections)
    partial_rotary = rope_params.get("partial_rotary_factor", 0.25)
    head_dim = text_cfg.get("head_dim", 256)
    writer.add_rope_dimension_count(int(head_dim * partial_rotary))

    # SSM / linear attention params
    writer.add_ssm_conv_kernel(text_cfg.get("linear_conv_kernel_dim", 4))
    writer.add_ssm_state_size(text_cfg.get("linear_key_head_dim", 128))

    # Layer types
    layer_types = text_cfg.get("layer_types", [])
    if layer_types:
        # Encode as array: 0 = linear_attention, 1 = full_attention
        lt_encoded = [0 if lt == "linear_attention" else 1 for lt in layer_types]
        writer.add_array("qwen35moe.layer_types", lt_encoded)

    # Vision params
    writer.add_vision_image_size(vision_cfg.get("num_position_embeddings", 2304))
    writer.add_vision_patch_size(vision_cfg.get("patch_size", 16))
    writer.add_vision_embedding_length(vision_cfg.get("hidden_size", 1152))
    writer.add_vision_block_count(vision_cfg.get("depth", 27))
    writer.add_vision_head_count(vision_cfg.get("num_heads", 16))
    writer.add_vision_feed_forward_length(vision_cfg.get("intermediate_size", 4304))
    writer.add_vision_projection_dim(vision_cfg.get("out_hidden_size", 4096))
    writer.add_vision_spatial_merge_size(vision_cfg.get("spatial_merge_size", 2))
    writer.add_clip_has_vision_encoder(True)

    # Special token IDs
    if config.get("image_token_id"):
        writer.add_uint32("tokenizer.ggml.image_token_id", config["image_token_id"])
    if config.get("video_token_id"):
        writer.add_uint32("tokenizer.ggml.video_token_id", config["video_token_id"])
    if config.get("vision_start_token_id"):
        writer.add_uint32("tokenizer.ggml.vision_start_token_id", config["vision_start_token_id"])
    if config.get("vision_end_token_id"):
        writer.add_uint32("tokenizer.ggml.vision_end_token_id", config["vision_end_token_id"])

    eos = text_cfg.get("eos_token_id")
    if eos is not None:
        writer.add_eos_token_id(eos)

    print("  Metadata written")


# ---------------------------------------------------------------------------
# Expert tensor merging
# ---------------------------------------------------------------------------

def merge_and_quantize_experts(
    expert_tensors: dict[int, torch.Tensor],
    num_experts: int,
    qtype: GGMLQuantizationType,
) -> tuple[np.ndarray, tuple]:
    """Merge and quantize expert tensors.

    Returns (quantized_data, shape) tuple.
    Uses native ggml quantization when available for maximum speed.
    """
    # Get shape from first expert
    first_expert = expert_tensors[0]
    expert_shape = first_expert.shape
    merged_shape = (num_experts,) + tuple(expert_shape)

    # Stack all experts into one tensor (on CPU to avoid GPU OOM)
    tensors = []
    for i in range(num_experts):
        t = expert_tensors[i].float().cpu().numpy()
        tensors.append(t)
    merged = np.stack(tensors, axis=0)  # [num_experts, ...]
    del tensors

    if qtype == GGMLQuantizationType.F32:
        return merged, merged_shape

    # Flatten and pad for quantization
    n_cols = merged.shape[-1]
    pad_cols = (QK_K - (n_cols % QK_K)) % QK_K

    merged_flat = merged.reshape(-1, n_cols)
    if pad_cols > 0:
        padding = np.zeros((merged_flat.shape[0], pad_cols), dtype=np.float32)
        merged_flat = np.concatenate([merged_flat, padding], axis=1)

    # Ensure contiguous
    merged_flat = np.ascontiguousarray(merged_flat)

    # Quantize using native ggml or fallback
    if GGML_LIB is not None:
        if qtype == GGMLQuantizationType.Q4_K:
            quantized = ggml_quantize_q4_k(merged_flat)
        elif qtype == GGMLQuantizationType.Q6_K:
            quantized = ggml_quantize_q6_k(merged_flat)
        else:
            quantized = quantize_tensor(merged_flat, qtype)
    else:
        quantized = quantize_tensor(merged_flat, qtype)

    return quantized, merged_shape


def quantize_tensor_gpu(tensor: torch.Tensor, qtype: GGMLQuantizationType) -> np.ndarray:
    """Quantize a GPU tensor directly. Tensor should be float32 on GPU.

    Handles padding and quantization on GPU in chunks to avoid OOM.
    Returns numpy array.
    """
    assert tensor.device.type == 'cuda', "Tensor must be on GPU"
    assert tensor.dtype == torch.float32, "Tensor must be float32"

    shape = tensor.shape

    # For K-quants, need to pad last dimension to multiple of 256
    if qtype in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q6_K):
        n_cols = shape[-1]
        pad_cols = (QK_K - (n_cols % QK_K)) % QK_K
        if pad_cols > 0:
            # Pad on GPU
            pad_shape = list(shape)
            pad_shape[-1] = pad_cols
            padding = torch.zeros(pad_shape, dtype=torch.float32, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=-1)

    # Flatten to 2D for quantization: [batch, cols]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    n_rows, n_cols = tensor.shape
    n_blocks_per_row = n_cols // QK_K

    if qtype == GGMLQuantizationType.Q4_K:
        bytes_per_row = n_blocks_per_row * Q4_K_BLOCK_SIZE
    elif qtype == GGMLQuantizationType.Q6_K:
        bytes_per_row = n_blocks_per_row * Q6_K_BLOCK_SIZE
    else:
        # Fall back to CPU for other types
        return gguf.quants.quantize(tensor.cpu().numpy(), qtype)

    # Process in chunks to avoid OOM
    # Each row needs ~4x memory during quantization (intermediates)
    # With 32GB VRAM, process ~64 experts (rows) at a time for 1024-dim tensors
    max_rows_per_chunk = 64

    if n_rows <= max_rows_per_chunk:
        # Small enough to process at once
        if qtype == GGMLQuantizationType.Q4_K:
            return _quantize_q4_k_gpu(tensor, n_rows, n_cols, n_blocks_per_row, bytes_per_row)
        else:
            return _quantize_q6_k_gpu(tensor, n_rows, n_cols, n_blocks_per_row, bytes_per_row)

    # Process in chunks
    output_chunks = []
    for start_row in range(0, n_rows, max_rows_per_chunk):
        end_row = min(start_row + max_rows_per_chunk, n_rows)
        chunk = tensor[start_row:end_row]
        chunk_rows = end_row - start_row

        if qtype == GGMLQuantizationType.Q4_K:
            chunk_out = _quantize_q4_k_gpu(chunk, chunk_rows, n_cols, n_blocks_per_row, bytes_per_row)
        else:
            chunk_out = _quantize_q6_k_gpu(chunk, chunk_rows, n_cols, n_blocks_per_row, bytes_per_row)

        output_chunks.append(chunk_out)

        # Free intermediate GPU memory
        del chunk
        torch.cuda.empty_cache()

    return np.concatenate(output_chunks, axis=0)


def _quantize_q4_k_gpu(tensor: torch.Tensor, n_rows: int, n_cols: int,
                        n_blocks_per_row: int, bytes_per_row: int) -> np.ndarray:
    """Q4_K quantization entirely on GPU with vectorized packing."""
    # Reshape to [n_rows, n_blocks_per_row, QK_K]
    tensor = tensor.view(n_rows, n_blocks_per_row, QK_K)

    # Reshape to [n_rows, n_blocks_per_row, 8, 32] for sub-blocks
    sub_blocks = tensor.view(n_rows, n_blocks_per_row, 8, 32)

    # Find min/max per sub-block (vectorized on GPU)
    sb_mins = sub_blocks.min(dim=3).values
    sb_maxs = sub_blocks.max(dim=3).values

    # Compute scales per sub-block
    scales = (sb_maxs - sb_mins) / 15.0
    scales = torch.where(sb_maxs != sb_mins, scales, torch.zeros_like(scales))
    mins = sb_mins

    # Super-block scales
    max_scale = scales.abs().max(dim=2).values
    max_min = mins.abs().max(dim=2).values

    d = max_scale / 63.0
    dmin = max_min / 63.0

    # Avoid division by zero
    inv_d = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    inv_dmin = torch.where(dmin > 0, 1.0 / dmin, torch.zeros_like(dmin))

    # Quantize scales/mins to 6-bit
    q_scales = (scales * inv_d.unsqueeze(2)).round().clamp(0, 63).to(torch.uint8)
    q_mins = ((-mins) * inv_dmin.unsqueeze(2)).round().clamp(0, 63).to(torch.uint8)

    # Quantize values to 4-bit
    eff_scales = (d.unsqueeze(2) * q_scales.float()).unsqueeze(3)
    eff_mins = (dmin.unsqueeze(2) * q_mins.float()).unsqueeze(3)
    inv_eff_scales = torch.where(eff_scales > 0, 1.0 / eff_scales, torch.zeros_like(eff_scales))
    qs = ((sub_blocks + eff_mins) * inv_eff_scales).round().clamp(0, 15).to(torch.uint8)

    # Move to CPU for packing
    d_cpu = d.half().cpu().numpy()  # [n_rows, n_blocks]
    dmin_cpu = dmin.half().cpu().numpy()
    q_scales_cpu = q_scales.cpu().numpy()  # [n_rows, n_blocks, 8]
    q_mins_cpu = q_mins.cpu().numpy()
    qs_cpu = qs.cpu().numpy().reshape(n_rows, n_blocks_per_row, 256)

    # Vectorized packing - no Python loops!
    output = np.zeros((n_rows, bytes_per_row), dtype=np.uint8)

    # Pack all blocks at once using stride tricks
    for blk in range(n_blocks_per_row):
        off = blk * Q4_K_BLOCK_SIZE

        # d and dmin (2 bytes each)
        output[:, off:off+2] = d_cpu[:, blk].view(np.uint8).reshape(n_rows, 2)
        output[:, off+2:off+4] = dmin_cpu[:, blk].view(np.uint8).reshape(n_rows, 2)

        # Scales packing (vectorized across all 4 indices)
        sc = q_scales_cpu[:, blk]  # [n_rows, 8]
        mn = q_mins_cpu[:, blk]
        output[:, off+4:off+8] = (sc[:, :4] & 0xF) | ((sc[:, 4:] & 0xF) << 4)
        output[:, off+8:off+12] = (mn[:, :4] & 0xF) | ((mn[:, 4:] & 0xF) << 4)
        output[:, off+12:off+16] = (
            ((sc[:, :4] >> 4) & 0x3) |
            (((sc[:, 4:] >> 4) & 0x3) << 2) |
            (((mn[:, :4] >> 4) & 0x3) << 4) |
            (((mn[:, 4:] >> 4) & 0x3) << 6)
        )

        # Quantized values packing (vectorized)
        q = qs_cpu[:, blk]  # [n_rows, 256]
        output[:, off+16:off+144] = (q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)

    return output


def _quantize_q6_k_gpu(tensor: torch.Tensor, n_rows: int, n_cols: int,
                        n_blocks_per_row: int, bytes_per_row: int) -> np.ndarray:
    """Q6_K quantization entirely on GPU with vectorized packing."""
    # Reshape to [n_rows, n_blocks_per_row, 16, 16]
    sub_blocks = tensor.view(n_rows, n_blocks_per_row, 16, 16)

    # Find max abs per sub-block (symmetric quantization)
    max_abs = sub_blocks.abs().max(dim=3).values
    scales = max_abs / 31.0

    # Super-block scale
    max_scale = scales.abs().max(dim=2).values
    d = max_scale / 127.0

    inv_d = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))

    # Quantize scales to 8-bit signed
    q_scales = (scales * inv_d.unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)

    # Quantize values
    eff_scales = (d.unsqueeze(2) * q_scales.float()).unsqueeze(3)
    inv_eff_scales = torch.where(eff_scales > 0, 1.0 / eff_scales, torch.zeros_like(eff_scales))
    qs = (sub_blocks * inv_eff_scales).round().clamp(-32, 31).to(torch.int8) + 32
    qs = qs.to(torch.uint8)

    # Move to CPU for packing
    d_cpu = d.half().cpu().numpy()  # [n_rows, n_blocks]
    q_scales_cpu = q_scales.cpu().numpy()  # [n_rows, n_blocks, 16]
    qs_cpu = qs.cpu().numpy().reshape(n_rows, n_blocks_per_row, 256)

    # Vectorized packing
    output = np.zeros((n_rows, bytes_per_row), dtype=np.uint8)

    for blk in range(n_blocks_per_row):
        off = blk * Q6_K_BLOCK_SIZE
        q = qs_cpu[:, blk]  # [n_rows, 256]

        # ql: lower 4 bits packed (128 bytes) - vectorized
        output[:, off:off+128] = (q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)

        # qh: upper 2 bits packed (64 bytes) - vectorized
        qh = (q >> 4) & 0x3  # [n_rows, 256]
        # Pack 4 values into each byte
        output[:, off+128:off+192] = (
            qh[:, 0::4] |
            (qh[:, 1::4] << 2) |
            (qh[:, 2::4] << 4) |
            (qh[:, 3::4] << 6)
        )

        # scales (16 bytes)
        output[:, off+192:off+208] = q_scales_cpu[:, blk].view(np.uint8).reshape(n_rows, 16)

        # d as float16 (2 bytes)
        output[:, off+208:off+210] = d_cpu[:, blk].view(np.uint8).reshape(n_rows, 2)

    return output


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def get_safetensor_files(model_dir: Path) -> list[Path]:
    """Get ordered list of safetensor files."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return [model_dir / f for f in files]

    # Single file
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def convert(model_dir: Path, output_path: Path):
    """Main conversion: HF bf16 safetensors → GGUF Q4_K_M."""
    print(f"Model dir: {model_dir}")
    print(f"Output:    {output_path}")

    # Initialize native ggml library for fast quantization
    print("\nInitializing...")
    use_native = init_ggml()
    if use_native:
        print("  Using native ggml quantization (C++ optimized)")
    else:
        print("  Using Python quantization (slower)")

    # Enable TF32 for faster GPU computation on Ampere+
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    text_cfg = config.get("text_config", {})
    num_layers = text_cfg.get("num_hidden_layers", 60)
    num_experts = text_cfg.get("num_experts", 512)
    layer_types = text_cfg.get("layer_types", [])

    print(f"Architecture: qwen3_5_moe")
    print(f"  Layers: {num_layers}, Experts: {num_experts}")
    print(f"  Layer types: {sum(1 for lt in layer_types if lt == 'full_attention')} full_attention, "
          f"{sum(1 for lt in layer_types if lt == 'linear_attention')} linear_attention")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA: not available (CPU mode)")

    # Create GGUF writer
    writer = GGUFWriter(str(output_path), arch="qwen35moe")

    # Write metadata
    print("\nWriting metadata...")
    write_metadata(writer, config)

    # Write tokenizer
    print("Writing tokenizer...")
    write_tokenizer(writer, model_dir)

    # Build weight_map: tensor_name → safetensors file
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Group tensors by layer for expert merging
    # Collect all tensor names and organize by processing order
    all_tensor_names = sorted(weight_map.keys())

    # Separate into categories
    non_layer_tensors = []  # top-level + vision
    layer_tensors = {}  # layer_idx → list of tensor names

    for name in all_tensor_names:
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
        if m:
            layer_idx = int(m.group(1))
            layer_tensors.setdefault(layer_idx, []).append(name)
        else:
            non_layer_tensors.append(name)

    # Process non-layer tensors first (embeddings, vision, norms)
    print(f"\nProcessing {len(non_layer_tensors)} non-layer tensors...")
    safetensor_cache = {}

    # Use CUDA for safetensor loading if available
    st_device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_safetensor_handle(filepath: str):
        if filepath not in safetensor_cache:
            full_path = str(model_dir / filepath)
            safetensor_cache[filepath] = safe_open(full_path, framework="pt", device=st_device)
        return safetensor_cache[filepath]

    tensor_count = 0
    for hf_name in non_layer_tensors:
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            continue

        st_file = weight_map[hf_name]
        handle = get_safetensor_handle(st_file)
        tensor = handle.get_tensor(hf_name)
        shape = tuple(tensor.shape)

        # Handle 5D vision patch embedding tensor (for temporal video support)
        # [out_ch, h, w, temporal, in_ch] or similar → merge to 4D
        if len(shape) == 5:
            old_shape = shape
            # Merge dims 1 and 2: [d0, d1, d2, d3, d4] -> [d0, d1*d2, d3, d4]
            shape = (shape[0], shape[1] * shape[2], shape[3], shape[4])
            print(f"  [5D→4D] {gguf_name}: {old_shape} → {shape}")
            # Reshape the tensor data
            tensor = tensor.reshape(shape)

        qtype = choose_quant_type(gguf_name, shape, num_layers)

        if qtype == GGMLQuantizationType.F32:
            data = bf16_tensor_to_f32_numpy(tensor)
            writer.add_tensor(gguf_name, data, raw_shape=shape)
        elif qtype == GGMLQuantizationType.F16:
            data = bf16_tensor_to_f16_numpy(tensor)
            writer.add_tensor(gguf_name, data, raw_shape=shape)
        else:
            data = bf16_tensor_to_f32_numpy(tensor)
            data = quantize_tensor(data, qtype)
            # Don't pass raw_shape with raw_dtype - library derives element shape from byte shape
            writer.add_tensor(gguf_name, data, raw_dtype=qtype)
        tensor_count += 1
        print(f"  [{tensor_count}] {gguf_name} {shape} → {qtype.name}")

    # Close all handles before processing layers
    safetensor_cache.clear()

    # Process layers
    print(f"\nProcessing {num_layers} layers...")

    for layer_idx in range(num_layers):
        if layer_idx not in layer_tensors:
            print(f"  [WARN] No tensors for layer {layer_idx}")
            continue

        t_start = time.time()
        names = layer_tensors[layer_idx]
        is_linear = (layer_idx < len(layer_types) and layer_types[layer_idx] == "linear_attention")

        # Separate expert tensors from non-expert tensors
        expert_groups = {}  # proj_type → {expert_idx: tensor}
        regular_tensors = []

        for hf_name in names:
            m = re.match(
                r"model\.language_model\.layers\.\d+\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
                hf_name,
            )
            if m:
                expert_idx = int(m.group(1))
                proj_type = m.group(2)
                expert_groups.setdefault(proj_type, {})[expert_idx] = hf_name
            else:
                regular_tensors.append(hf_name)

        # Process regular (non-expert) tensors
        for hf_name in regular_tensors:
            gguf_name = map_tensor_name(hf_name)
            if gguf_name is None:
                continue

            st_file = weight_map[hf_name]
            handle = get_safetensor_handle(st_file)
            tensor = handle.get_tensor(hf_name)
            shape = tuple(tensor.shape)

            # Handle 5D tensors in layer processing too (shouldn't happen but just in case)
            if len(shape) == 5:
                old_shape = shape
                shape = (shape[0], shape[1] * shape[2], shape[3], shape[4])
                tensor = tensor.reshape(shape)
                print(f"    [5D→4D] {gguf_name}: {old_shape} → {shape}")

            qtype = choose_quant_type(gguf_name, shape, num_layers)

            if qtype == GGMLQuantizationType.F32:
                # Stay on GPU for conversion, move to CPU only at the end
                data = bf16_tensor_to_f32_numpy(tensor)
                writer.add_tensor(gguf_name, data, raw_shape=shape)
            elif qtype == GGMLQuantizationType.F16:
                data = bf16_tensor_to_f16_numpy(tensor)
                writer.add_tensor(gguf_name, data, raw_shape=shape)
            else:
                # For quantization, convert on GPU then quantize
                data = bf16_tensor_to_f32_numpy(tensor)
                data = quantize_tensor(data, qtype)
                writer.add_tensor(gguf_name, data, raw_dtype=qtype)
            tensor_count += 1

        # Process expert tensors (merge all experts per projection type)
        # This is the bulk of the model - optimize for GPU
        proj_to_gguf = {
            "gate_proj": f"blk.{layer_idx}.ffn_gate_exps.weight",
            "up_proj": f"blk.{layer_idx}.ffn_up_exps.weight",
            "down_proj": f"blk.{layer_idx}.ffn_down_exps.weight",
        }

        for proj_type in ["gate_proj", "up_proj", "down_proj"]:
            if proj_type not in expert_groups:
                continue

            gguf_name = proj_to_gguf[proj_type]
            expert_dict = expert_groups[proj_type]

            # Load all expert tensors for this projection (stays on source device initially)
            expert_tensors_dict = {}
            for expert_idx, hf_name in expert_dict.items():
                st_file = weight_map[hf_name]
                handle = get_safetensor_handle(st_file)
                expert_tensors_dict[expert_idx] = handle.get_tensor(hf_name)

            # Determine shape for quant type selection
            first_shape = expert_tensors_dict[0].shape
            merged_shape = (num_experts,) + tuple(first_shape)
            qtype = choose_quant_type(gguf_name, merged_shape, num_layers)

            # Merge and quantize in chunks to avoid OOM
            data, merged_shape = merge_and_quantize_experts(expert_tensors_dict, num_experts, qtype)

            if qtype == GGMLQuantizationType.F32:
                writer.add_tensor(gguf_name, data, raw_shape=merged_shape)
            else:
                writer.add_tensor(gguf_name, data, raw_dtype=qtype)
            tensor_count += 1

            # Free memory
            del expert_tensors_dict, data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Close safetensor handles to free memory between layers
        safetensor_cache.clear()

        elapsed = time.time() - t_start
        layer_type = "linear" if is_linear else "full"
        n_expert_tensors = sum(len(v) for v in expert_groups.values())
        print(
            f"  Layer {layer_idx:2d}/{num_layers} ({layer_type:6s}): "
            f"{len(regular_tensors)} regular + {n_expert_tensors} expert tensors "
            f"→ {tensor_count} total [{elapsed:.1f}s]"
        )

    # Finalize
    print(f"\nWriting GGUF file ({tensor_count} tensors)...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = output_path.stat().st_size
    print(f"Done! Output: {output_path} ({file_size / (1024**3):.1f} GB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5-397B-A17B HF bf16 safetensors → GGUF Q4_K_M"
    )
    parser.add_argument("model_dir", type=Path, help="Path to HF model directory")
    parser.add_argument(
        "--output", "-o", type=Path,
        default=None,
        help="Output GGUF file path (default: <model_dir_name>-Q4_K_M.gguf)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"{args.model_dir.name}-Q4_K_M.gguf")

    convert(args.model_dir, args.output)


if __name__ == "__main__":
    main()
