#!/usr/bin/env python3
"""
Convert HunyuanImage-3.0 model to FP8 format.

This script loads the model, converts eligible Linear layers to FP8 with per-tensor scaling,
and saves the result as a new checkpoint. Vision components are kept in bf16.

Usage:
    python convert_to_fp8.py --input models/LLM/HunyuanImage-3.0-Instruct --output models/LLM/HunyuanImage-3.0-Instruct-FP8
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm


# Modules to skip during FP8 conversion (keep in original precision)
# These are critical for quality and numerical stability
SKIP_PATTERNS = [
    # VAE - pixel reconstruction quality
    'vae',
    # Vision Encoder - semantic understanding (SigLIP2)
    'vision_model', 'vision_aligner',
    # Image Generation Head - latent output
    'final_layer', 'patch_embed',
    # Timestep/Guidance Embeddings - diffusion timing
    'time_embed', 'time_embed_2', 'timestep_emb', 'guidance_emb', 'timestep_r_emb',
    # MoE Router Gates - numerical stability
    'wg',
    # Token Embeddings
    'wte', 'lm_head',
    # All LayerNorms - must stay high precision for stability
    'ln_f', 'layernorm', 'input_layernorm', 'post_attention_layernorm',
    'key_layernorm', 'query_layernorm',
]


def should_convert_layer(name: str, skip_patterns: list[str]) -> bool:
    """Check if a layer should be converted to FP8."""
    name_lower = name.lower()
    return not any(pattern in name_lower for pattern in skip_patterns)


def convert_linear_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a weight tensor to FP8 with per-tensor scaling.

    Returns:
        tuple of (fp8_weight, scale)
    """
    # Compute scale factor (FP8 E4M3 max value is ~448)
    weight_float = weight.float()
    abs_max = weight_float.abs().max()

    if abs_max == 0:
        abs_max = torch.tensor(1.0, dtype=torch.float32)

    scale = (abs_max / 448.0).float()

    # Convert to FP8
    fp8_weight = (weight_float / scale).to(torch.float8_e4m3fn)

    return fp8_weight, scale


def check_dynamic_range(weight: torch.Tensor, threshold: float = 1e6) -> bool:
    """Check if weight has acceptable dynamic range for FP8."""
    weight_float = weight.float()
    abs_max = weight_float.abs().max()
    non_zero = weight_float[weight_float != 0]

    if non_zero.numel() == 0:
        return True

    abs_min = non_zero.abs().min()

    if abs_max > 0 and abs_min > 0:
        dynamic_range = abs_max / abs_min
        return dynamic_range <= threshold

    return True


def convert_model(
    input_path: str,
    output_path: str,
    skip_patterns: Optional[list[str]] = None,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
):
    """
    Convert a model to FP8 format.

    Args:
        input_path: Path to input model directory
        output_path: Path to output model directory
        skip_patterns: Layer name patterns to skip (keep in original precision)
        dtype: Data type for non-FP8 layers
        verbose: Print detailed conversion info
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS

    # Normalize patterns to lowercase
    skip_patterns = [p.lower() for p in skip_patterns]

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"[FP8 Convert] Input: {input_path}")
    print(f"[FP8 Convert] Output: {output_path}")
    print(f"[FP8 Convert] Base dtype: {dtype}")
    print(f"[FP8 Convert] Skip patterns: {skip_patterns}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files (config, tokenizer, etc.)
    print("\n[FP8 Convert] Copying config and tokenizer files...")
    for file in input_path.iterdir():
        if file.is_file() and not file.name.endswith('.safetensors') and not file.name.endswith('.bin'):
            if file.name != 'model.safetensors.index.json':  # We'll regenerate this
                shutil.copy2(file, output_path / file.name)
                print(f"  Copied: {file.name}")

    # Copy subdirectories (like __pycache__, .cache)
    for subdir in input_path.iterdir():
        if subdir.is_dir() and subdir.name not in ['__pycache__', '.cache']:
            if (output_path / subdir.name).exists():
                shutil.rmtree(output_path / subdir.name)
            shutil.copytree(subdir, output_path / subdir.name)
            print(f"  Copied dir: {subdir.name}")

    # Load model
    print("\n[FP8 Convert] Loading model (this may take a while)...")
    print("  Note: Loading in bf16 without device_map to ensure all weights are accessible")

    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,  # Load to CPU, no offloading
        low_cpu_mem_usage=True,
    )

    print(f"[FP8 Convert] Model loaded successfully")

    # Collect all linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    print(f"[FP8 Convert] Found {len(linear_layers)} linear layers")

    # Convert layers
    converted_count = 0
    skipped_pattern = 0
    skipped_range = 0
    total_original_bytes = 0
    total_fp8_bytes = 0

    # Track which parameters were converted (for state_dict modification)
    converted_params = {}  # name -> (fp8_weight, scale)

    print("\n[FP8 Convert] Converting layers...")
    for name, module in tqdm(linear_layers, desc="Converting"):
        weight = module.weight.data

        # Check skip patterns
        if not should_convert_layer(name, skip_patterns):
            skipped_pattern += 1
            if verbose:
                print(f"  Skip (pattern): {name}")
            continue

        # Check dynamic range
        if not check_dynamic_range(weight):
            skipped_range += 1
            if verbose:
                print(f"  Skip (range): {name}")
            continue

        # Track original size
        original_bytes = weight.numel() * weight.element_size()
        total_original_bytes += original_bytes

        # Convert to FP8
        fp8_weight, scale = convert_linear_to_fp8(weight)

        # Track FP8 size
        fp8_bytes = fp8_weight.numel() * fp8_weight.element_size() + 4  # +4 for scale
        total_fp8_bytes += fp8_bytes

        # Store converted weight and scale
        converted_params[name + '.weight'] = fp8_weight
        converted_params[name + '.scale_weight'] = scale.view(1)

        converted_count += 1

    print(f"\n[FP8 Convert] Conversion complete:")
    print(f"  Converted: {converted_count} layers")
    print(f"  Skipped (pattern): {skipped_pattern} layers")
    print(f"  Skipped (range): {skipped_range} layers")

    if total_original_bytes > 0:
        original_gb = total_original_bytes / (1024**3)
        fp8_gb = total_fp8_bytes / (1024**3)
        reduction = (1 - fp8_gb / original_gb) * 100
        print(f"  Memory: {original_gb:.2f}GB -> {fp8_gb:.2f}GB ({reduction:.1f}% reduction)")

    # Get state dict and modify it
    print("\n[FP8 Convert] Building state dict...")
    state_dict = model.state_dict()

    # Replace converted weights and add scales
    for param_name, tensor in converted_params.items():
        if param_name.endswith('.scale_weight'):
            # Add new scale parameter
            state_dict[param_name] = tensor
        else:
            # Replace original weight with FP8 version
            state_dict[param_name] = tensor

    # Save model in shards
    print("\n[FP8 Convert] Saving model...")
    save_sharded_model(state_dict, output_path, max_shard_size_gb=4.0)

    # Update config to mark as FP8
    config_path = output_path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['fp8_converted'] = True
        config['fp8_skip_patterns'] = skip_patterns
        config['fp8_converted_layers'] = converted_count
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"  Updated config.json with FP8 metadata")

    print(f"\n[FP8 Convert] Done! Model saved to: {output_path}")

    return converted_count


def save_sharded_model(state_dict: dict, output_path: Path, max_shard_size_gb: float = 4.0):
    """Save state dict as sharded safetensors files."""
    from safetensors.torch import save_file

    max_shard_bytes = int(max_shard_size_gb * 1024**3)

    # Sort keys for deterministic sharding
    sorted_keys = sorted(state_dict.keys())

    # Group into shards
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted_keys:
        tensor = state_dict[key]
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    print(f"  Saving {len(shards)} shard(s)...")

    # Save shards
    weight_map = {}
    total_size = 0

    for i, shard in enumerate(tqdm(shards, desc="Saving shards")):
        if len(shards) == 1:
            shard_name = "model.safetensors"
        else:
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"

        shard_path = output_path / shard_name

        # Move tensors to CPU and ensure contiguous
        shard_cpu = {}
        for key, tensor in shard.items():
            t = tensor.cpu().contiguous()
            shard_cpu[key] = t
            total_size += t.numel() * t.element_size()
            weight_map[key] = shard_name

        save_file(shard_cpu, shard_path)

    # Save index file if sharded
    if len(shards) > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        index_path = output_path / "model.safetensors.index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"  Saved index: {index_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HunyuanImage-3.0 model to FP8 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python convert_to_fp8.py --input models/LLM/HunyuanImage-3.0-Instruct --output models/LLM/HunyuanImage-3.0-Instruct-FP8

    # With verbose output
    python convert_to_fp8.py --input models/LLM/HunyuanImage-3.0-Instruct --output models/LLM/HunyuanImage-3.0-Instruct-FP8 --verbose
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input model directory"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output model directory"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for non-FP8 layers (default: bfloat16)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed conversion info"
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Run conversion
    convert_model(
        input_path=args.input,
        output_path=args.output,
        dtype=dtype,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
