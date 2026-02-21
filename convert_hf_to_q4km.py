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

# ---------------------------------------------------------------------------
# Native Q4_K / Q6_K quantization (gguf library doesn't implement K-quants)
# ---------------------------------------------------------------------------

QK_K = 256  # Super-block size for K-quants


def quantize_q4_k_cuda(data: np.ndarray) -> np.ndarray:
    """CUDA-accelerated Q4_K quantization using PyTorch."""
    assert data.dtype == np.float32

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(data).to(device)

    # Reshape to super-blocks of 256 elements
    n_elements = tensor.numel()
    n_blocks = (n_elements + QK_K - 1) // QK_K
    padded_size = n_blocks * QK_K

    if padded_size > n_elements:
        tensor = torch.nn.functional.pad(tensor.view(-1), (0, padded_size - n_elements))
    else:
        tensor = tensor.view(-1)

    tensor = tensor.view(n_blocks, QK_K)

    # Divide into 8 sub-blocks of 32 elements each
    sub_blocks = tensor.view(n_blocks, 8, 32)

    # Find min/max per sub-block (vectorized)
    sb_mins = sub_blocks.min(dim=2).values  # [n_blocks, 8]
    sb_maxs = sub_blocks.max(dim=2).values  # [n_blocks, 8]

    # Compute scales per sub-block
    scales = (sb_maxs - sb_mins) / 15.0  # [n_blocks, 8]
    scales = torch.where(sb_maxs != sb_mins, scales, torch.zeros_like(scales))
    mins = sb_mins

    # Super-block scales
    max_scale = scales.abs().max(dim=1).values  # [n_blocks]
    max_min = mins.abs().max(dim=1).values  # [n_blocks]

    d = max_scale / 63.0  # [n_blocks]
    dmin = max_min / 63.0  # [n_blocks]

    # Avoid division by zero
    inv_d = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    inv_dmin = torch.where(dmin > 0, 1.0 / dmin, torch.zeros_like(dmin))

    # Quantize scales/mins to 6-bit
    q_scales = (scales * inv_d.unsqueeze(1)).round().clamp(0, 63).to(torch.uint8)
    q_mins = ((-mins) * inv_dmin.unsqueeze(1)).round().clamp(0, 63).to(torch.uint8)

    # Move back to CPU for packing
    d_cpu = d.half().cpu().numpy()
    dmin_cpu = dmin.half().cpu().numpy()
    q_scales_cpu = q_scales.cpu().numpy()
    q_mins_cpu = q_mins.cpu().numpy()

    # Quantize values to 4-bit
    # Reconstruct effective scales and mins
    eff_scales = (d.unsqueeze(1) * q_scales.float()).unsqueeze(2)  # [n_blocks, 8, 1]
    eff_mins = (dmin.unsqueeze(1) * q_mins.float()).unsqueeze(2)  # [n_blocks, 8, 1]

    # Quantize: q = round((x + min) / scale)
    inv_eff_scales = torch.where(eff_scales > 0, 1.0 / eff_scales, torch.zeros_like(eff_scales))
    qs = ((sub_blocks + eff_mins) * inv_eff_scales).round().clamp(0, 15).to(torch.uint8)
    qs_cpu = qs.cpu().numpy()  # [n_blocks, 8, 32]

    # Pack into output (144 bytes per block)
    output = np.zeros((n_blocks, 144), dtype=np.uint8)

    # Store d and dmin as float16 (bytes 0-4)
    output[:, 0:2] = d_cpu.view(np.uint8).reshape(n_blocks, 2)
    output[:, 2:4] = dmin_cpu.view(np.uint8).reshape(n_blocks, 2)

    # Pack scales (6-bit) into 12 bytes (bytes 4-16)
    for i in range(4):
        output[:, 4 + i] = (q_scales_cpu[:, i] & 0xF) | ((q_scales_cpu[:, i + 4] & 0xF) << 4)
    for i in range(4):
        output[:, 8 + i] = (q_mins_cpu[:, i] & 0xF) | ((q_mins_cpu[:, i + 4] & 0xF) << 4)
    for i in range(4):
        output[:, 12 + i] = ((q_scales_cpu[:, i] >> 4) & 0x3) | \
                           (((q_scales_cpu[:, i + 4] >> 4) & 0x3) << 2) | \
                           (((q_mins_cpu[:, i] >> 4) & 0x3) << 4) | \
                           (((q_mins_cpu[:, i + 4] >> 4) & 0x3) << 6)

    # Pack quantized values (4-bit) into 128 bytes (bytes 16-144)
    qs_flat = qs_cpu.reshape(n_blocks, 256)
    for i in range(128):
        output[:, 16 + i] = (qs_flat[:, i * 2] & 0xF) | ((qs_flat[:, i * 2 + 1] & 0xF) << 4)

    return output.ravel()


def quantize_q6_k_cuda(data: np.ndarray) -> np.ndarray:
    """CUDA-accelerated Q6_K quantization using PyTorch."""
    assert data.dtype == np.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(data).to(device)

    n_elements = tensor.numel()
    n_blocks = (n_elements + QK_K - 1) // QK_K
    padded_size = n_blocks * QK_K

    if padded_size > n_elements:
        tensor = torch.nn.functional.pad(tensor.view(-1), (0, padded_size - n_elements))
    else:
        tensor = tensor.view(-1)

    tensor = tensor.view(n_blocks, QK_K)

    # Divide into 16 sub-blocks of 16 elements
    sub_blocks = tensor.view(n_blocks, 16, 16)

    # Find max abs per sub-block (symmetric quantization)
    max_abs = sub_blocks.abs().max(dim=2).values  # [n_blocks, 16]
    scales = max_abs / 31.0  # 6-bit symmetric = -32 to 31

    # Super-block scale
    max_scale = scales.abs().max(dim=1).values  # [n_blocks]
    d = max_scale / 127.0  # 8-bit scales

    inv_d = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))

    # Quantize scales to 8-bit signed
    q_scales = (scales * inv_d.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

    # Quantize values
    eff_scales = (d.unsqueeze(1) * q_scales.float()).unsqueeze(2)  # [n_blocks, 16, 1]
    inv_eff_scales = torch.where(eff_scales > 0, 1.0 / eff_scales, torch.zeros_like(eff_scales))
    qs = (sub_blocks * inv_eff_scales).round().clamp(-32, 31).to(torch.int8) + 32  # shift to 0-63
    qs = qs.to(torch.uint8)

    # Move to CPU
    d_cpu = d.half().cpu().numpy()
    q_scales_cpu = q_scales.cpu().numpy()
    qs_cpu = qs.cpu().numpy().reshape(n_blocks, 256)

    # Pack into output (210 bytes per block)
    output = np.zeros((n_blocks, 210), dtype=np.uint8)

    # ql: lower 4 bits (128 bytes, positions 0-128)
    for i in range(128):
        output[:, i] = (qs_cpu[:, i * 2] & 0xF) | ((qs_cpu[:, i * 2 + 1] & 0xF) << 4)

    # qh: upper 2 bits (64 bytes, positions 128-192)
    for i in range(256):
        qh_byte = i // 4
        qh_shift = (i % 4) * 2
        output[:, 128 + qh_byte] |= ((qs_cpu[:, i] >> 4) & 0x3) << qh_shift

    # scales (16 bytes, positions 192-208)
    output[:, 192:208] = q_scales_cpu.view(np.uint8).reshape(n_blocks, 16)

    # d as float16 (2 bytes, positions 208-210)
    output[:, 208:210] = d_cpu.view(np.uint8).reshape(n_blocks, 2)

    return output.ravel()


def quantize_q4_k(data: np.ndarray) -> np.ndarray:
    """Quantize float32 data to Q4_K format.

    Q4_K block structure (144 bytes per 256 elements):
    - d: float16 (2 bytes) - super-block scale
    - dmin: float16 (2 bytes) - super-block minimum
    - scales: uint8[12] - packed 6-bit scales/mins for 8 sub-blocks
    - qs: uint8[128] - 4-bit quantized values (2 per byte)
    """
    assert data.dtype == np.float32

    # Reshape to super-blocks of 256 elements
    n_elements = data.size
    n_blocks = (n_elements + QK_K - 1) // QK_K

    # Pad to multiple of QK_K
    padded_size = n_blocks * QK_K
    if padded_size > n_elements:
        data = np.pad(data.ravel(), (0, padded_size - n_elements), mode='constant')
    else:
        data = data.ravel()

    data = data.reshape(n_blocks, QK_K)

    # Output: 144 bytes per block (2 + 2 + 12 + 128)
    output = np.zeros((n_blocks, 144), dtype=np.uint8)

    for block_idx in range(n_blocks):
        block = data[block_idx]

        # Divide into 8 sub-blocks of 32 elements
        sub_blocks = block.reshape(8, 32)

        # Find scales and mins for each sub-block
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)

        for i in range(8):
            sb = sub_blocks[i]
            sb_min = sb.min()
            sb_max = sb.max()

            mins[i] = sb_min
            if sb_max != sb_min:
                scales[i] = (sb_max - sb_min) / 15.0  # 4-bit = 0-15
            else:
                scales[i] = 0.0

        # Find super-block scale for scales
        max_scale = np.abs(scales).max()
        max_min = np.abs(mins).max()

        if max_scale > 0:
            d = max_scale / 63.0  # 6-bit scales
            inv_d = 1.0 / d
        else:
            d = 0.0
            inv_d = 0.0

        if max_min > 0:
            dmin = max_min / 63.0  # 6-bit mins
            inv_dmin = 1.0 / dmin
        else:
            dmin = 0.0
            inv_dmin = 0.0

        # Quantize sub-block scales/mins to 6-bit
        q_scales = np.zeros(8, dtype=np.uint8)
        q_mins = np.zeros(8, dtype=np.uint8)

        for i in range(8):
            if d > 0:
                q_scales[i] = min(63, int(np.round(scales[i] * inv_d)))
            if dmin > 0:
                q_mins[i] = min(63, int(np.round(-mins[i] * inv_dmin)))

        # Store d and dmin as float16
        d_f16 = np.float16(d)
        dmin_f16 = np.float16(dmin)
        output[block_idx, 0:2] = np.frombuffer(d_f16.tobytes(), dtype=np.uint8)
        output[block_idx, 2:4] = np.frombuffer(dmin_f16.tobytes(), dtype=np.uint8)

        # Pack scales (6-bit) into 12 bytes
        # Layout: scales[0-3] low 4 bits, scales[4-7] low 4 bits,
        #         mins[0-3] low 4 bits, mins[4-7] low 4 bits,
        #         scales[0-3] high 2 bits + mins[0-3] high 2 bits,
        #         scales[4-7] high 2 bits + mins[4-7] high 2 bits
        scales_bytes = np.zeros(12, dtype=np.uint8)

        for i in range(4):
            scales_bytes[i] = (q_scales[i] & 0xF) | ((q_scales[i + 4] & 0xF) << 4)
        for i in range(4):
            scales_bytes[4 + i] = (q_mins[i] & 0xF) | ((q_mins[i + 4] & 0xF) << 4)
        for i in range(4):
            scales_bytes[8 + i] = ((q_scales[i] >> 4) & 0x3) | (((q_scales[i + 4] >> 4) & 0x3) << 2) | \
                                  (((q_mins[i] >> 4) & 0x3) << 4) | (((q_mins[i + 4] >> 4) & 0x3) << 6)

        output[block_idx, 4:16] = scales_bytes

        # Quantize values to 4-bit
        qs = np.zeros(128, dtype=np.uint8)

        for i in range(8):
            sb = sub_blocks[i]
            sc = d * q_scales[i]
            mn = dmin * q_mins[i]

            for j in range(32):
                val = sb[j]
                if sc > 0:
                    q = int(np.round((val + mn) / sc))
                    q = max(0, min(15, q))
                else:
                    q = 0

                # Pack two 4-bit values per byte
                byte_idx = (i * 32 + j) // 2
                if j % 2 == 0:
                    qs[byte_idx] = q
                else:
                    qs[byte_idx] |= (q << 4)

        output[block_idx, 16:144] = qs

    return output.ravel()


def quantize_q6_k(data: np.ndarray) -> np.ndarray:
    """Quantize float32 data to Q6_K format.

    Q6_K block structure (210 bytes per 256 elements):
    - ql: uint8[128] - lower 4 bits of 6-bit quants
    - qh: uint8[64] - upper 2 bits of 6-bit quants
    - scales: int8[16] - scales for 16 sub-blocks of 16 elements
    - d: float16 (2 bytes) - super-block scale
    """
    assert data.dtype == np.float32

    # Reshape to super-blocks of 256 elements
    n_elements = data.size
    n_blocks = (n_elements + QK_K - 1) // QK_K

    # Pad to multiple of QK_K
    padded_size = n_blocks * QK_K
    if padded_size > n_elements:
        data = np.pad(data.ravel(), (0, padded_size - n_elements), mode='constant')
    else:
        data = data.ravel()

    data = data.reshape(n_blocks, QK_K)

    # Output: 210 bytes per block (128 + 64 + 16 + 2)
    output = np.zeros((n_blocks, 210), dtype=np.uint8)

    for block_idx in range(n_blocks):
        block = data[block_idx]

        # Divide into 16 sub-blocks of 16 elements
        sub_blocks = block.reshape(16, 16)

        # Find scales for each sub-block (symmetric quantization)
        scales = np.zeros(16, dtype=np.float32)

        for i in range(16):
            sb = sub_blocks[i]
            max_abs = np.abs(sb).max()
            if max_abs > 0:
                scales[i] = max_abs / 31.0  # 6-bit symmetric = -32 to 31

        # Find super-block scale
        max_scale = np.abs(scales).max()

        if max_scale > 0:
            d = max_scale / 127.0  # 8-bit scales
            inv_d = 1.0 / d
        else:
            d = 0.0
            inv_d = 0.0

        # Quantize sub-block scales to 8-bit signed
        q_scales = np.zeros(16, dtype=np.int8)
        for i in range(16):
            if d > 0:
                q_scales[i] = max(-128, min(127, int(np.round(scales[i] * inv_d))))

        # Quantize values to 6-bit symmetric (-32 to 31)
        ql = np.zeros(128, dtype=np.uint8)  # lower 4 bits
        qh = np.zeros(64, dtype=np.uint8)   # upper 2 bits

        for i in range(16):
            sb = sub_blocks[i]
            sc = d * q_scales[i]

            for j in range(16):
                val = sb[j]
                if sc > 0:
                    q = int(np.round(val / sc))
                    q = max(-32, min(31, q)) + 32  # shift to 0-63
                else:
                    q = 32  # zero

                # Element index in block
                elem_idx = i * 16 + j

                # Store lower 4 bits
                byte_idx = elem_idx // 2
                if elem_idx % 2 == 0:
                    ql[byte_idx] = q & 0xF
                else:
                    ql[byte_idx] |= (q & 0xF) << 4

                # Store upper 2 bits
                qh_byte_idx = elem_idx // 4
                qh_bit_offset = (elem_idx % 4) * 2
                qh[qh_byte_idx] |= ((q >> 4) & 0x3) << qh_bit_offset

        # Pack output
        output[block_idx, 0:128] = ql
        output[block_idx, 128:192] = qh
        output[block_idx, 192:208] = q_scales.view(np.uint8)
        d_f16 = np.float16(d)
        output[block_idx, 208:210] = np.frombuffer(d_f16.tobytes(), dtype=np.uint8)

    return output.ravel()


def quantize_tensor(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    """Quantize a tensor using the specified quantization type.

    Uses CUDA-accelerated implementation for K-quants when available.
    """
    if qtype == GGMLQuantizationType.Q4_K:
        if torch.cuda.is_available():
            return quantize_q4_k_cuda(data)
        return quantize_q4_k(data)
    elif qtype == GGMLQuantizationType.Q6_K:
        if torch.cuda.is_available():
            return quantize_q6_k_cuda(data)
        return quantize_q6_k(data)
    else:
        # Fall back to gguf library for basic quants
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

    # SSM A_log → F32 (small, critical for recurrence)
    if "ssm_a_log" in gguf_name:
        return GGMLQuantizationType.F32

    # SSM dt_bias → F32
    if "ssm_dt.bias" in gguf_name:
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

    # First and last block weights → Q6_K
    if is_edge:
        return GGMLQuantizationType.Q6_K

    # Everything else (attention projections, expert weights, shared expert weights) → Q4_K
    return GGMLQuantizationType.Q4_K


# ---------------------------------------------------------------------------
# CUDA-accelerated bf16 → f32 conversion
# ---------------------------------------------------------------------------

def bf16_tensor_to_f32_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a bf16 torch tensor to f32 numpy via CUDA if available."""
    if tensor.dtype == torch.bfloat16:
        if torch.cuda.is_available():
            tensor = tensor.cuda().float().cpu()
        else:
            tensor = tensor.float()
    elif tensor.dtype == torch.float16:
        if torch.cuda.is_available():
            tensor = tensor.cuda().float().cpu()
        else:
            tensor = tensor.float()
    else:
        tensor = tensor.float()
    return tensor.numpy()


def bf16_tensor_to_f16_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to f16 numpy via CUDA."""
    if torch.cuda.is_available():
        tensor = tensor.cuda().half().cpu()
    else:
        tensor = tensor.half()
    return tensor.numpy()


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

def merge_experts(
    expert_tensors: dict[int, torch.Tensor],
    num_experts: int,
) -> np.ndarray:
    """Merge individual expert tensors into a single stacked tensor [n_experts, ...].

    Uses CUDA for the stacking operation.
    """
    tensors = []
    for i in range(num_experts):
        if i not in expert_tensors:
            raise ValueError(f"Missing expert {i}")
        tensors.append(expert_tensors[i])

    if torch.cuda.is_available():
        stacked = torch.stack([t.cuda() for t in tensors]).float().cpu()
    else:
        stacked = torch.stack(tensors).float()
    return stacked.numpy()


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

    def get_safetensor_handle(filepath: str):
        if filepath not in safetensor_cache:
            full_path = str(model_dir / filepath)
            safetensor_cache[filepath] = safe_open(full_path, framework="pt", device="cpu")
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

        qtype = choose_quant_type(gguf_name, shape, num_layers)

        if qtype == GGMLQuantizationType.F32:
            data = bf16_tensor_to_f32_numpy(tensor)
        elif qtype == GGMLQuantizationType.F16:
            data = bf16_tensor_to_f16_numpy(tensor)
        else:
            data = bf16_tensor_to_f32_numpy(tensor)
            data = quantize_tensor(data, qtype)

        writer.add_tensor(gguf_name, data, raw_shape=shape, raw_dtype=qtype)
        tensor_count += 1
        print(f"  [{tensor_count}] {gguf_name} {shape} → {qtype.name}")

    # Close all handles before processing layers
    safetensor_cache.clear()

    # Process layers one at a time for memory efficiency
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

            qtype = choose_quant_type(gguf_name, shape, num_layers)

            if qtype == GGMLQuantizationType.F32:
                data = bf16_tensor_to_f32_numpy(tensor)
            elif qtype == GGMLQuantizationType.F16:
                data = bf16_tensor_to_f16_numpy(tensor)
            else:
                data = bf16_tensor_to_f32_numpy(tensor)
                data = quantize_tensor(data, qtype)

            writer.add_tensor(gguf_name, data, raw_shape=shape, raw_dtype=qtype)
            tensor_count += 1

        # Process expert tensors (merge all experts per projection type)
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

            # Load all expert tensors for this projection
            expert_tensors_dict = {}
            for expert_idx, hf_name in expert_dict.items():
                st_file = weight_map[hf_name]
                handle = get_safetensor_handle(st_file)
                expert_tensors_dict[expert_idx] = handle.get_tensor(hf_name)

            # Merge: stack into [n_experts, out_dim, in_dim]
            merged = merge_experts(expert_tensors_dict, num_experts)
            merged_shape = merged.shape

            qtype = choose_quant_type(gguf_name, merged_shape, num_layers)

            if qtype == GGMLQuantizationType.F32:
                data = merged
            else:
                data = quantize_tensor(merged, qtype)

            writer.add_tensor(gguf_name, data, raw_shape=merged_shape, raw_dtype=qtype)
            tensor_count += 1

            # Free memory
            del expert_tensors_dict, merged
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
