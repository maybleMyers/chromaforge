#!/usr/bin/env python3
"""
SDNQ Pre-Quantization Script for VLM Models

This script pre-quantizes large VLM models using SDNQ, creating smaller
quantized versions that load quickly without on-the-fly quantization.

Uses a streaming approach to handle models that don't fit in memory.

Usage:
    python sdnq_quantize.py --model models/LLM/Qwen3.5-397B-A17B --output models/LLM/Qwen3.5-397B-A17B-SDNQ-int4 --dtype int4
"""

import os
import sys
import argparse
import json
import gc
import shutil
import re
from glob import glob
from typing import Optional, List, Dict, Any
from tqdm import tqdm

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_gpu_info():
    """Get GPU memory info."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {free_mem / 1024**3:.1f} GB free / {total_mem / 1024**3:.1f} GB total")
    else:
        print("No CUDA GPUs available")


def should_skip_quantization(param_name: str, skip_patterns: List[str]) -> bool:
    """Check if a parameter should skip quantization."""
    # Split param name into parts for matching
    parts = param_name.split(".")

    for pattern in skip_patterns:
        # Direct substring match
        if pattern in param_name:
            return True
        # Part match (e.g., "visual" matches "model.visual.encoder")
        if pattern in parts:
            return True

    return False


def quantize_tensor(
    tensor: torch.Tensor,
    weights_dtype: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Quantize a single tensor using SDNQ.

    Returns dict with quantized weight, scale, and optionally zero_point.
    """
    from sdnq.quantizer import sdnq_quantize_layer_weight
    from sdnq.common import dtype_dict

    # Move to quantization device
    original_dtype = tensor.dtype
    tensor = tensor.to(device=device, dtype=torch.float32)

    # Quantize (assuming linear layer - most common case)
    weight, scale, zero_point, svd_up, svd_down, dequantizer = sdnq_quantize_layer_weight(
        tensor,
        layer_class_name="Linear",
        weights_dtype=weights_dtype,
        torch_dtype=torch.bfloat16,
        group_size=0,  # Auto
        use_quantized_matmul=False,  # Will be set at load time
        use_stochastic_rounding=False,
        dequantize_fp32=False,
    )

    result = {
        "weight": weight.cpu(),
        "scale": scale.cpu(),
    }

    if zero_point is not None:
        result["zero_point"] = zero_point.cpu()

    return result


def quantize_model_streaming(
    model_path: str,
    output_path: str,
    weights_dtype: str = "int8",
    max_shard_size_gb: float = 5.0,
):
    """
    Quantize a model using streaming - process one tensor at a time.

    This approach:
    1. Reads tensors from source safetensors files one at a time
    2. Quantizes each tensor on GPU
    3. Accumulates into output shards
    4. Saves shards when they reach max_shard_size_gb

    This never needs to hold the full model in memory.
    """
    from sdnq import SDNQConfig
    from sdnq.common import dtype_dict, linear_types

    print(f"\n{'='*60}")
    print("SDNQ Streaming Quantization")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Weights dtype: {weights_dtype}")
    print(f"Max shard size: {max_shard_size_gb} GB")
    print(f"{'='*60}\n")

    get_gpu_info()

    # Skip patterns - modules that shouldn't be quantized
    skip_patterns = [
        'visual',  # Vision encoder
        'embed_tokens', 'wte',  # Token embeddings
        'lm_head',  # Output head
        'layernorm', 'ln_f', 'ln_1', 'ln_2',  # LayerNorms
        'input_layernorm', 'post_attention_layernorm',
        'norm', 'final_layernorm',
        'rotary_emb',  # Rotary embeddings
        'gate',  # MoE gate (keep in high precision)
    ]

    # Find source safetensor files
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    print(f"Found {len(safetensor_files)} safetensor files")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Copy non-weight files (config, tokenizer, etc.)
    print("\nCopying config and tokenizer files...")
    for filename in os.listdir(model_path):
        if filename.endswith('.safetensors'):
            continue
        if filename == 'offload_cache':
            continue
        src = os.path.join(model_path, filename)
        dst = os.path.join(output_path, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # Set up quantization device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Quantization device: {device}")

    # Track output shards
    current_shard = {}
    current_shard_size = 0
    shard_index = 1
    max_shard_bytes = int(max_shard_size_gb * 1024**3)

    # Track weight map for index
    weight_map = {}

    # Statistics
    stats = {
        "total_params": 0,
        "quantized_params": 0,
        "skipped_params": 0,
        "original_size": 0,
        "quantized_size": 0,
    }

    def save_current_shard():
        nonlocal current_shard, current_shard_size, shard_index
        if not current_shard:
            return

        shard_name = f"model-{shard_index:05d}-of-XXXXX.safetensors"
        shard_path = os.path.join(output_path, shard_name)

        print(f"  Saving shard {shard_index} ({len(current_shard)} tensors, {current_shard_size / 1024**3:.2f} GB)")
        save_file(current_shard, shard_path)

        for key in current_shard.keys():
            weight_map[key] = shard_name

        current_shard = {}
        current_shard_size = 0
        shard_index += 1

    def add_to_shard(name: str, tensor: torch.Tensor):
        nonlocal current_shard_size
        tensor_size = tensor.numel() * tensor.element_size()

        # Save current shard if adding this tensor would exceed limit
        if current_shard_size + tensor_size > max_shard_bytes and current_shard:
            save_current_shard()

        current_shard[name] = tensor
        current_shard_size += tensor_size
        stats["quantized_size"] += tensor_size

    # Process each source file
    print("\nQuantizing weights...")
    for sf_path in safetensor_files:
        sf_name = os.path.basename(sf_path)
        print(f"\nProcessing {sf_name}...")

        with safe_open(sf_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

            for key in tqdm(keys, desc=f"  {sf_name}"):
                tensor = f.get_tensor(key)
                stats["total_params"] += 1
                stats["original_size"] += tensor.numel() * tensor.element_size()

                # Check if we should quantize this tensor
                should_skip = (
                    should_skip_quantization(key, skip_patterns) or
                    tensor.ndim != 2 or  # Only quantize 2D tensors (Linear weights)
                    tensor.dtype not in [torch.float16, torch.bfloat16, torch.float32] or
                    min(tensor.shape) < 32  # Skip small tensors
                )

                if should_skip:
                    # Keep original tensor
                    add_to_shard(key, tensor)
                    stats["skipped_params"] += 1
                else:
                    # Quantize this tensor
                    try:
                        quantized = quantize_tensor(tensor, weights_dtype, device)

                        # Add quantized tensors with SDNQ naming convention
                        base_name = key
                        add_to_shard(base_name, quantized["weight"])

                        # Scale uses same base name with .scale suffix
                        scale_name = base_name.replace(".weight", ".scale")
                        if scale_name == base_name:
                            scale_name = base_name + ".scale"
                        add_to_shard(scale_name, quantized["scale"])

                        if "zero_point" in quantized:
                            zp_name = base_name.replace(".weight", ".zero_point")
                            if zp_name == base_name:
                                zp_name = base_name + ".zero_point"
                            add_to_shard(zp_name, quantized["zero_point"])

                        stats["quantized_params"] += 1

                    except Exception as e:
                        # If quantization fails, keep original
                        print(f"    Warning: Failed to quantize {key}: {e}")
                        add_to_shard(key, tensor)
                        stats["skipped_params"] += 1

                # Clear CUDA cache periodically
                if stats["total_params"] % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gc.collect()

    # Save final shard
    save_current_shard()

    # Fix shard names (replace XXXXX with actual count)
    total_shards = shard_index - 1
    final_weight_map = {}

    for old_name in sorted(glob(os.path.join(output_path, "model-*-of-XXXXX.safetensors"))):
        new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
        os.rename(old_name, new_name)

        old_basename = os.path.basename(old_name)
        new_basename = os.path.basename(new_name)

        for key, shard in weight_map.items():
            if shard == old_basename:
                final_weight_map[key] = new_basename

    # Save model index
    index = {
        "metadata": {
            "total_size": stats["quantized_size"],
        },
        "weight_map": final_weight_map,
    }

    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Save quantization config
    quant_config = {
        "quant_method": "sdnq",
        "weights_dtype": weights_dtype,
        "group_size": 0,
        "use_quantized_matmul": False,
        "modules_to_not_convert": skip_patterns,
    }

    with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Update config.json to include quantization_config
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["quantization_config"] = quant_config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Print statistics
    print(f"\n{'='*60}")
    print("Quantization Complete!")
    print(f"{'='*60}")
    print(f"Total parameters: {stats['total_params']}")
    print(f"Quantized: {stats['quantized_params']}")
    print(f"Skipped: {stats['skipped_params']}")
    print(f"Original size: {stats['original_size'] / 1024**3:.2f} GB")
    print(f"Quantized size: {stats['quantized_size'] / 1024**3:.2f} GB")
    print(f"Compression ratio: {stats['original_size'] / stats['quantized_size']:.2f}x")
    print(f"Output shards: {total_shards}")
    print(f"\nOutput saved to: {output_path}")
    print(f"\nTo load: Use 'bfloat16' precision in vlm_diffusers.py")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-quantize VLM models using SDNQ (streaming mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # INT4 quantization (smallest, ~4x compression)
    python sdnq_quantize.py --model models/LLM/Qwen3.5-397B-A17B --output models/LLM/Qwen3.5-397B-A17B-SDNQ-int4 --dtype int4

    # INT8 quantization (balanced quality/size)
    python sdnq_quantize.py --model models/LLM/Qwen3.5-397B-A17B --output models/LLM/Qwen3.5-397B-A17B-SDNQ-int8 --dtype int8

Supported dtypes:
    int8, int7, int6, int5, int4, int3, int2
    uint8, uint7, uint6, uint5, uint4, uint3, uint2
    float8_e4m3fn, float7_e3m3fn, float6_e3m2fn
        """,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the original model",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save the quantized model",
    )

    parser.add_argument(
        "--dtype", "-d",
        type=str,
        default="int8",
        help="Quantization dtype (default: int8)",
    )

    parser.add_argument(
        "--shard-size",
        type=float,
        default=5.0,
        help="Max shard size in GB (default: 5.0)",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        sys.exit(1)

    if os.path.exists(args.output):
        print(f"Warning: Output path already exists: {args.output}")
        response = input("Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(args.output)

    quantize_model_streaming(
        model_path=args.model,
        output_path=args.output,
        weights_dtype=args.dtype,
        max_shard_size_gb=args.shard_size,
    )


if __name__ == "__main__":
    main()
