#!/usr/bin/env python3
"""
Convert HunyuanImage-3.0 model to FP8 format.

This script processes weights shard-by-shard using the GPU for fast conversion,
converting eligible Linear layers to FP8 with per-tensor scaling.
Vision components are kept in bf16.

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


def should_convert_weight(name: str, skip_patterns: list[str]) -> bool:
    """Check if a weight tensor should be converted to FP8."""
    # Only convert .weight tensors from linear layers (not biases, not norms)
    if not name.endswith('.weight'):
        return False

    # Skip non-2D tensors (embeddings, norms, etc.)
    # We'll check tensor shape during conversion

    name_lower = name.lower()
    return not any(pattern in name_lower for pattern in skip_patterns)


def convert_tensor_to_fp8(weight: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a weight tensor to FP8 with per-tensor scaling on GPU.

    Returns:
        tuple of (fp8_weight, scale) on CPU
    """
    # Move to GPU for fast computation
    weight_gpu = weight.to(device).float()

    # Compute scale factor (FP8 E4M3 max value is ~448)
    abs_max = weight_gpu.abs().max()

    if abs_max == 0:
        abs_max = torch.tensor(1.0, dtype=torch.float32, device=device)

    scale = (abs_max / 448.0).float()

    # Convert to FP8
    fp8_weight = (weight_gpu / scale).to(torch.float8_e4m3fn)

    # Return on CPU to save GPU memory
    return fp8_weight.cpu(), scale.cpu()


def check_dynamic_range(weight: torch.Tensor, device: torch.device, threshold: float = 1e6) -> bool:
    """Check if weight has acceptable dynamic range for FP8."""
    weight_gpu = weight.to(device).float()
    abs_max = weight_gpu.abs().max()

    # Find non-zero minimum
    non_zero_mask = weight_gpu != 0
    if not non_zero_mask.any():
        return True

    abs_min = weight_gpu.abs()[non_zero_mask].min()

    if abs_max > 0 and abs_min > 0:
        dynamic_range = (abs_max / abs_min).item()
        return dynamic_range <= threshold

    return True


def convert_model_sharded(
    input_path: str,
    output_path: str,
    skip_patterns: Optional[list[str]] = None,
    device: str = "cuda",
    verbose: bool = False,
):
    """
    Convert a model to FP8 format by processing shards directly.

    This approach:
    - Loads one shard at a time
    - Converts weights on GPU
    - Saves converted shard
    - Uses minimal memory (~5GB GPU, ~10GB RAM)

    Args:
        input_path: Path to input model directory
        output_path: Path to output model directory
        skip_patterns: Layer name patterns to skip (keep in original precision)
        device: Device for conversion (cuda or cpu)
        verbose: Print detailed conversion info
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS

    # Normalize patterns to lowercase
    skip_patterns = [p.lower() for p in skip_patterns]

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[FP8 Convert] Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print(f"[FP8 Convert] Using CPU (slower)")

    print(f"[FP8 Convert] Input: {input_path}")
    print(f"[FP8 Convert] Output: {output_path}")
    print(f"[FP8 Convert] Skip patterns: {skip_patterns}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy all non-weight files (config, tokenizer, Python files, etc.)
    print("\n[FP8 Convert] Copying model files...")
    copied_files = []
    skipped_files = []

    for file in input_path.iterdir():
        if file.is_file():
            # Skip weight files (we'll create new ones)
            if file.name.endswith('.safetensors') or file.name.endswith('.bin'):
                skipped_files.append(file.name)
                continue
            # Skip index file (we'll regenerate it)
            if file.name == 'model.safetensors.index.json':
                skipped_files.append(file.name)
                continue

            # Copy everything else (Python files, configs, tokenizer, README, etc.)
            shutil.copy2(file, output_path / file.name)
            copied_files.append(file.name)

    # Copy subdirectories (excluding __pycache__ and .cache)
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            if subdir.name in ['__pycache__', '.cache']:
                continue
            dest = output_path / subdir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(subdir, dest, ignore=shutil.ignore_patterns('__pycache__', '.cache', '*.pyc'))
            copied_files.append(f"{subdir.name}/")

    print(f"  Copied {len(copied_files)} files/directories:")
    for f in sorted(copied_files):
        print(f"    - {f}")
    if verbose and skipped_files:
        print(f"  Skipped (will be regenerated): {', '.join(skipped_files)}")

    # Find all safetensor shards
    index_path = input_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # Get unique shard files in order
        shard_files = sorted(set(weight_map.values()))
    else:
        # Single file model
        single_file = input_path / "model.safetensors"
        if single_file.exists():
            shard_files = ["model.safetensors"]
            weight_map = None
        else:
            raise FileNotFoundError(f"No safetensors files found in {input_path}")

    print(f"\n[FP8 Convert] Found {len(shard_files)} shard(s) to process")

    # Statistics
    converted_count = 0
    skipped_pattern = 0
    skipped_shape = 0
    skipped_range = 0
    total_original_bytes = 0
    total_fp8_bytes = 0

    # New weight map for output
    new_weight_map = {}
    total_size = 0

    # Process each shard
    for shard_idx, shard_name in enumerate(shard_files):
        shard_path = input_path / shard_name
        output_shard_path = output_path / shard_name

        print(f"\n[FP8 Convert] Processing shard {shard_idx + 1}/{len(shard_files)}: {shard_name}")

        if not shard_path.exists():
            print(f"  ERROR: Shard file not found: {shard_path}")
            continue

        # Load shard
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
            print(f"  Tensors in shard: {len(tensor_names)}")

            # Process tensors
            output_tensors = {}
            shard_converted = 0

            # Debug: find and test a tensor that SHOULD convert
            if shard_idx == 0:
                print("  DEBUG: Looking for convertible tensors...")
                found_convertible = False
                for debug_name in tensor_names:
                    debug_tensor = f.get_tensor(debug_name)
                    should = should_convert_weight(debug_name, skip_patterns)
                    if should and debug_tensor.ndim == 2:
                        print(f"    FOUND: {debug_name}")
                        print(f"      should_convert={should}, shape={debug_tensor.shape}, dtype={debug_tensor.dtype}")
                        # Test dynamic range
                        range_ok = check_dynamic_range(debug_tensor, device)
                        print(f"      dynamic_range_ok={range_ok}")
                        found_convertible = True
                        break
                if not found_convertible:
                    print("    NO CONVERTIBLE TENSORS FOUND IN SHARD 1!")
                    # Show what's blocking conversion
                    for debug_name in tensor_names[:20]:
                        debug_tensor = f.get_tensor(debug_name)
                        should = should_convert_weight(debug_name, skip_patterns)
                        print(f"    {debug_name}: should={should}, ndim={debug_tensor.ndim}")

            for name in tqdm(tensor_names, desc=f"Shard {shard_idx + 1}", leave=False):
                tensor = f.get_tensor(name)

                # Check if this should be converted
                if should_convert_weight(name, skip_patterns):
                    # Must be 2D (linear layer weight)
                    if tensor.ndim != 2:
                        skipped_shape += 1
                        output_tensors[name] = tensor
                        if verbose:
                            print(f"  Skip (shape {tensor.shape}): {name}")
                    # Check dynamic range
                    elif not check_dynamic_range(tensor, device):
                        skipped_range += 1
                        output_tensors[name] = tensor
                        if verbose:
                            print(f"  Skip (range): {name}")
                    else:
                        # Convert to FP8
                        original_bytes = tensor.numel() * tensor.element_size()
                        total_original_bytes += original_bytes

                        fp8_weight, scale = convert_tensor_to_fp8(tensor, device)

                        fp8_bytes = fp8_weight.numel() * fp8_weight.element_size()
                        total_fp8_bytes += fp8_bytes + 4  # +4 for scale

                        # Store converted weight and scale
                        output_tensors[name] = fp8_weight
                        scale_name = name.replace('.weight', '.scale_weight')
                        output_tensors[scale_name] = scale.view(1)

                        converted_count += 1
                        shard_converted += 1

                        if verbose:
                            print(f"  Converted: {name}")
                else:
                    # Keep original
                    if name.endswith('.weight') and tensor.ndim == 2:
                        skipped_pattern += 1
                        if verbose:
                            print(f"  Skip (pattern): {name}")
                    output_tensors[name] = tensor

            print(f"  Converted {shard_converted} tensors in this shard")

            # Update weight map
            for tensor_name, tensor in output_tensors.items():
                new_weight_map[tensor_name] = shard_name
                total_size += tensor.numel() * tensor.element_size()

            # Save output shard
            print(f"  Saving to: {output_shard_path}")
            save_file(output_tensors, output_shard_path)
            print(f"  Saved successfully")

        # Clear GPU cache between shards
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save new index file
    if len(shard_files) > 1:
        new_index = {
            "metadata": {"total_size": total_size},
            "weight_map": new_weight_map
        }
        new_index_path = output_path / "model.safetensors.index.json"
        with open(new_index_path, 'w') as f:
            json.dump(new_index, f, indent=2)

    # Print summary
    print(f"\n[FP8 Convert] Conversion complete:")
    print(f"  Converted: {converted_count} linear layers to FP8")
    print(f"  Skipped (pattern): {skipped_pattern} layers")
    print(f"  Skipped (shape): {skipped_shape} tensors")
    print(f"  Skipped (range): {skipped_range} layers")

    if total_original_bytes > 0:
        original_gb = total_original_bytes / (1024**3)
        fp8_gb = total_fp8_bytes / (1024**3)
        reduction = (1 - fp8_gb / original_gb) * 100
        print(f"  Converted layers: {original_gb:.2f}GB -> {fp8_gb:.2f}GB ({reduction:.1f}% reduction)")

    # Update config to mark as FP8 and remove torch_dtype to prevent auto-conversion on load
    config_path = output_path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['fp8_converted'] = True
        config['fp8_skip_patterns'] = skip_patterns
        config['fp8_converted_layers'] = converted_count
        # CRITICAL: Remove torch_dtype so transformers doesn't convert FP8 back to bf16 on load
        if 'torch_dtype' in config:
            print(f"  Removing torch_dtype from config (was: {config['torch_dtype']})")
            del config['torch_dtype']
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"  Updated config.json with FP8 metadata")

    print(f"\n[FP8 Convert] Done! Model saved to: {output_path}")

    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert HunyuanImage-3.0 model to FP8 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion (uses GPU)
    python convert_to_fp8.py --input models/LLM/HunyuanImage-3.0-Instruct --output models/LLM/HunyuanImage-3.0-Instruct-FP8

    # With verbose output
    python convert_to_fp8.py --input models/LLM/HunyuanImage-3.0-Instruct --output models/LLM/HunyuanImage-3.0-Instruct-FP8 --verbose

    # Force CPU (slower but works without GPU)
    python convert_to_fp8.py --input models/LLM/HunyuanImage-3.0-Instruct --output models/LLM/HunyuanImage-3.0-Instruct-FP8 --device cpu
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
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for conversion (default: cuda)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed conversion info"
    )

    args = parser.parse_args()

    # Run conversion
    convert_model_sharded(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
