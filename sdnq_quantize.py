#!/usr/bin/env python3
"""
SDNQ Pre-Quantization Script for VLM Models

This script pre-quantizes large VLM models using SDNQ, creating smaller
quantized versions that load quickly without on-the-fly quantization.

Usage:
    python sdnq_quantize.py --model models/LLM/Qwen3.5-397B-A17B --output models/LLM/Qwen3.5-397B-A17B-SDNQ-int4 --dtype int4

    # With custom settings
    python sdnq_quantize.py --model /path/to/model --output /path/to/output --dtype int8 --use-quantized-matmul
"""

import os
import sys
import argparse
import json
import gc
from typing import Optional

import torch


def get_gpu_info():
    """Get GPU memory info."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {free_mem / 1024**3:.1f} GB free / {total_mem / 1024**3:.1f} GB total")
    else:
        print("No CUDA GPUs available")


def quantize_model(
    model_path: str,
    output_path: str,
    weights_dtype: str = "int8",
    use_quantized_matmul: bool = False,
    max_memory_gpu: Optional[int] = None,
    max_memory_cpu: int = 256,
    trust_remote_code: bool = True,
):
    """
    Quantize a model using SDNQ and save it.

    Args:
        model_path: Path to the original model
        output_path: Path to save the quantized model
        weights_dtype: SDNQ weight dtype (int8, int4, float8_e4m3fn, etc.)
        use_quantized_matmul: Enable quantized matmul kernels (requires triton)
        max_memory_gpu: Max GPU memory in GB (None = auto)
        max_memory_cpu: Max CPU memory in GB for offloading
        trust_remote_code: Trust remote code for custom models
    """
    # Import here to avoid slow startup for --help
    from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
    from sdnq import SDNQConfig
    from sdnq.loader import save_sdnq_model
    from sdnq.common import use_torch_compile as sdnq_triton_available

    print(f"\n{'='*60}")
    print("SDNQ Pre-Quantization Script")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Weights dtype: {weights_dtype}")
    print(f"Quantized matmul: {use_quantized_matmul}")
    print(f"Triton available: {sdnq_triton_available}")
    print(f"{'='*60}\n")

    get_gpu_info()

    # Skip modules that shouldn't be quantized
    modules_to_not_convert = [
        'visual',  # Vision encoder - critical for VLMs
        'embed_tokens', 'wte',  # Token embeddings
        'lm_head',  # Output head
        'norm', 'layernorm', 'ln_f',  # LayerNorms
        'input_layernorm', 'post_attention_layernorm',
    ]

    # Create SDNQ config
    # Key: quantize on GPU, return to CPU for memory efficiency
    sdnq_config = SDNQConfig(
        weights_dtype=weights_dtype,
        group_size=0,  # Auto
        use_quantized_matmul=use_quantized_matmul and sdnq_triton_available,
        quantization_device="cuda:0" if torch.cuda.is_available() else None,
        return_device="cpu",  # Return quantized weights to CPU to save VRAM
        modules_to_not_convert=modules_to_not_convert,
    )

    # Set up memory constraints
    max_memory = {}
    if torch.cuda.is_available():
        if max_memory_gpu is not None:
            max_memory[0] = f"{max_memory_gpu}GiB"
        else:
            # Use 80% of available VRAM
            free_mem, _ = torch.cuda.mem_get_info(0)
            gpu_mem_gb = int((free_mem / 1024**3) * 0.8)
            max_memory[0] = f"{gpu_mem_gb}GiB"
            print(f"Auto-detected GPU memory: {gpu_mem_gb} GB")

    max_memory["cpu"] = f"{max_memory_cpu}GiB"
    print(f"Memory config: {max_memory}")

    # Create offload folder
    offload_folder = os.path.join(os.path.dirname(model_path), "offload_cache")
    os.makedirs(offload_folder, exist_ok=True)
    print(f"Offload folder: {offload_folder}")

    # Load processor first (small)
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # Load and quantize model
    print("\nLoading and quantizing model (this may take a while)...")
    print("Progress will be shown for weight loading...")

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_folder,
        quantization_config=sdnq_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print("\nModel loaded and quantized successfully!")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Save quantized model
    print(f"\nSaving quantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Save processor
    processor.save_pretrained(output_path)
    print("Saved processor")

    # Save model using SDNQ's save function
    save_sdnq_model(model, output_path, max_shard_size="5GB", sdnq_config=sdnq_config)
    print("Saved quantized model")

    # Save additional metadata
    metadata = {
        "original_model": model_path,
        "sdnq_weights_dtype": weights_dtype,
        "sdnq_use_quantized_matmul": use_quantized_matmul,
        "modules_to_not_convert": modules_to_not_convert,
    }
    with open(os.path.join(output_path, "sdnq_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Quantization complete!")
    print(f"{'='*60}")

    # Show size comparison
    def get_dir_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total

    orig_size = get_dir_size(model_path)
    quant_size = get_dir_size(output_path)

    print(f"Original model size: {orig_size / 1024**3:.2f} GB")
    print(f"Quantized model size: {quant_size / 1024**3:.2f} GB")
    print(f"Compression ratio: {orig_size / quant_size:.2f}x")
    print(f"\nTo use: Select '{os.path.basename(output_path)}' in vlm_diffusers.py with 'bfloat16' precision")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-quantize VLM models using SDNQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # INT4 quantization (smallest, ~4x compression)
    python sdnq_quantize.py --model models/LLM/Qwen3.5-397B-A17B --output models/LLM/Qwen3.5-397B-A17B-SDNQ-int4 --dtype int4

    # INT8 quantization (balanced quality/size)
    python sdnq_quantize.py --model models/LLM/Qwen3.5-397B-A17B --output models/LLM/Qwen3.5-397B-A17B-SDNQ-int8 --dtype int8

    # FP8 quantization (best quality, ~2x compression)
    python sdnq_quantize.py --model models/LLM/Qwen3.5-397B-A17B --output models/LLM/Qwen3.5-397B-A17B-SDNQ-fp8 --dtype float8_e4m3fn

Supported dtypes:
    int8, int7, int6, int5, int4, int3, int2
    uint8, uint7, uint6, uint5, uint4, uint3, uint2
    float8_e4m3fn, float7_e3m3fn, float6_e3m2fn, float5_e2m2fn, float4_e2m1fn
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
        help="Quantization dtype (default: int8). Options: int8, int4, float8_e4m3fn, etc.",
    )

    parser.add_argument(
        "--use-quantized-matmul",
        action="store_true",
        help="Enable quantized matmul kernels (requires triton, faster inference)",
    )

    parser.add_argument(
        "--max-gpu-memory",
        type=int,
        default=None,
        help="Max GPU memory in GB (default: auto-detect)",
    )

    parser.add_argument(
        "--max-cpu-memory",
        type=int,
        default=256,
        help="Max CPU memory in GB for offloading (default: 256)",
    )

    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Don't trust remote code (may break custom models)",
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

    quantize_model(
        model_path=args.model,
        output_path=args.output,
        weights_dtype=args.dtype,
        use_quantized_matmul=args.use_quantized_matmul,
        max_memory_gpu=args.max_gpu_memory,
        max_memory_cpu=args.max_cpu_memory,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()
