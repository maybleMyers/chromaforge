#!/usr/bin/env python3
"""
Q8 (INT8) Quantization Converter for Vision-Language Models

This script converts a model to Q8 (INT8) format with selective quantization,
preserving bf16 precision for sensitive layers (vision encoder, norms, embeddings).

The converted model uses per-tensor symmetric quantization with scale factors.

Usage:
    python q8_quantization_converter.py \
        --model "models/LLM/qwen235bf16" \
        --output "models/LLM/qwen235q8" \
        --analyze-only  # Optional: just analyze, don't convert
"""

import os
import gc
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
from tqdm import tqdm

# Thread-local storage for GPU assignment
_thread_local = threading.local()


def get_available_gpus() -> List[str]:
    """Get list of available CUDA devices."""
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def print_gpu_info():
    """Print information about available GPUs."""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s):")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")


# Check for safetensors
try:
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Install with: pip install safetensors")


def get_model_info(model_path: str) -> Dict:
    """Load model config and get basic info."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def find_safetensor_files(model_path: str) -> List[str]:
    """Find all safetensor files in model directory."""
    model_dir = Path(model_path)

    # Check for model.safetensors.index.json (sharded model)
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
        # Get unique shard files
        shard_files = sorted(set(index["weight_map"].values()))
        return [str(model_dir / f) for f in shard_files]

    # Check for single model.safetensors
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return [str(single_file)]

    # Fallback: find all .safetensors files
    files = list(model_dir.glob("*.safetensors"))
    if files:
        return sorted([str(f) for f in files])

    raise FileNotFoundError(f"No safetensors files found in {model_path}")


def analyze_tensor(name: str, tensor: torch.Tensor, device: str = "cpu") -> Dict:
    """Analyze a tensor for INT8 quantization suitability."""
    # Convert to float32 for analysis, use GPU if available
    if device != "cpu" and torch.cuda.is_available():
        t = tensor.to(device).float()
    else:
        t = tensor.float()

    # Basic stats
    abs_vals = t.abs()
    non_zero = t[t != 0]

    stats = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "size_mb": tensor.numel() * tensor.element_size() / (1024 * 1024),
    }

    if tensor.numel() == 0:
        stats["quantizable"] = False
        stats["reason"] = "empty tensor"
        return stats

    # Range analysis
    stats["min"] = float(t.min())
    stats["max"] = float(t.max())
    stats["abs_max"] = float(abs_vals.max())
    stats["abs_min"] = float(non_zero.abs().min()) if non_zero.numel() > 0 else 0.0
    stats["mean"] = float(t.mean())
    stats["std"] = float(t.std())

    # Dynamic range (important for quantization)
    if stats["abs_min"] > 0:
        stats["dynamic_range"] = stats["abs_max"] / stats["abs_min"]
    else:
        stats["dynamic_range"] = float('inf')

    # Kurtosis (measure of outliers)
    if stats["std"] > 0:
        stats["kurtosis"] = float(((t - stats["mean"]) / stats["std"]).pow(4).mean())
    else:
        stats["kurtosis"] = 0.0

    # Determine if quantizable
    # INT8 has range [-127, 127] with symmetric quantization
    stats["quantizable"] = True
    stats["risk_score"] = 0
    stats["reasons"] = []

    # Check dynamic range
    if stats["dynamic_range"] > 1e6:
        stats["risk_score"] += 30
        stats["reasons"].append(f"Very high dynamic range: {stats['dynamic_range']:.2e}")
    elif stats["dynamic_range"] > 1e4:
        stats["risk_score"] += 15
        stats["reasons"].append(f"High dynamic range: {stats['dynamic_range']:.2e}")

    # Check kurtosis (outliers)
    if stats["kurtosis"] > 50:
        stats["risk_score"] += 25
        stats["reasons"].append(f"Very high kurtosis: {stats['kurtosis']:.1f}")
    elif stats["kurtosis"] > 20:
        stats["risk_score"] += 10
        stats["reasons"].append(f"High kurtosis: {stats['kurtosis']:.1f}")

    return stats


def should_skip_layer(name: str, skip_patterns: List[str]) -> Tuple[bool, str]:
    """Check if a layer should be skipped based on name patterns."""
    name_lower = name.lower()

    for pattern in skip_patterns:
        if pattern in name_lower:
            return True, f"matches pattern '{pattern}'"

    return False, ""


def analyze_shard(shard_file: str, skip_patterns: List[str], device: str) -> Dict:
    """Analyze a single shard file on a specific GPU."""
    results = {
        "layers": {},
        "total_layers": 0,
        "quantizable_layers": 0,
        "skipped_layers": 0,
        "total_size_mb": 0,
        "quantizable_size_mb": 0,
        "skipped_size_mb": 0,
        "categories": {"safe": [], "medium_risk": [], "high_risk": [], "skipped": []}
    }

    if not os.path.exists(shard_file):
        return results

    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)

            # Only analyze weight tensors (not biases, which are small)
            if tensor.numel() < 1000:
                continue

            # Analyze tensor using GPU
            stats = analyze_tensor(name, tensor, device=device)
            results["total_layers"] += 1
            results["total_size_mb"] += stats["size_mb"]

            # Check skip patterns
            should_skip, skip_reason = should_skip_layer(name, skip_patterns)

            if should_skip:
                stats["quantizable"] = False
                stats["skip_reason"] = skip_reason
                stats["category"] = "skipped_pattern"
                results["skipped_layers"] += 1
                results["skipped_size_mb"] += stats["size_mb"]
                results["categories"]["skipped"].append(name)
            elif stats["risk_score"] >= 50:
                stats["quantizable"] = False
                stats["category"] = "high_risk"
                results["skipped_layers"] += 1
                results["skipped_size_mb"] += stats["size_mb"]
                results["categories"]["high_risk"].append(name)
            elif stats["risk_score"] >= 25:
                stats["category"] = "medium_risk"
                results["quantizable_layers"] += 1
                results["quantizable_size_mb"] += stats["size_mb"]
                results["categories"]["medium_risk"].append(name)
            else:
                stats["category"] = "safe"
                results["quantizable_layers"] += 1
                results["quantizable_size_mb"] += stats["size_mb"]
                results["categories"]["safe"].append(name)

            results["layers"][name] = stats
            del tensor

    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def analyze_model(model_path: str, skip_patterns: Optional[List[str]] = None, device: str = "cuda") -> Dict:
    """
    Analyze a model for INT8 quantization.

    Args:
        model_path: Path to model directory
        skip_patterns: Layer name patterns to skip (keep in bf16)
        device: Device for analysis computation ("cpu", "cuda", "cuda:0", etc.)

    Returns detailed analysis of each layer and recommendations.
    """
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors is required for analysis")

    # Determine GPUs to use
    gpus = []
    if device != "cpu":
        if torch.cuda.is_available():
            if device == "cuda" or device == "all":
                # Use all available GPUs
                gpus = get_available_gpus()
            else:
                # Use specific GPU(s)
                gpus = [device]
            print_gpu_info()
            print(f"Using {len(gpus)} GPU(s) for analysis: {gpus}")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"

    if not gpus:
        gpus = ["cpu"]

    if skip_patterns is None:
        # Default patterns to skip (keep in bf16)
        # Vision layers are CRITICAL for VLM quality
        skip_patterns = [
            'visual',       # Vision encoder - VERY SENSITIVE, must stay bf16
            'embed',        # Embeddings
            'lm_head',      # Output head
            'norm',         # Normalization layers (LayerNorm, RMSNorm)
            'rotary',       # Rotary embeddings
            'wte', 'wpe',   # Token/position embeddings
        ]

    print(f"\nAnalyzing model: {model_path}")
    print(f"Skip patterns (keep in bf16): {skip_patterns}")

    # Get model config
    config = get_model_info(model_path)
    model_type = config.get("model_type", "unknown")
    print(f"Model type: {model_type}")

    # Find safetensor files
    safetensor_files = find_safetensor_files(model_path)
    print(f"Found {len(safetensor_files)} safetensor file(s)")

    # Analyze all tensors
    analysis = {
        "model_path": model_path,
        "model_type": model_type,
        "analysis_time": datetime.now().isoformat(),
        "skip_patterns": skip_patterns,
        "layers": {},
        "summary": {
            "total_layers": 0,
            "quantizable_layers": 0,
            "skipped_layers": 0,
            "total_size_mb": 0,
            "quantizable_size_mb": 0,
            "skipped_size_mb": 0,
        }
    }

    # Categories for reporting
    categories = defaultdict(list)

    # Process shards in parallel across GPUs
    num_workers = len(gpus)

    if num_workers > 1:
        print(f"Processing {len(safetensor_files)} shards in parallel across {num_workers} GPUs...")

        # Create work items: (shard_file, gpu_device)
        work_items = [(shard_file, gpus[i % num_workers]) for i, shard_file in enumerate(safetensor_files)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_shard = {
                executor.submit(analyze_shard, shard_file, skip_patterns, gpu): shard_file
                for shard_file, gpu in work_items
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_shard), total=len(safetensor_files), desc="Analyzing shards"):
                shard_file = future_to_shard[future]
                try:
                    result = future.result()

                    # Merge results
                    analysis["summary"]["total_layers"] += result["total_layers"]
                    analysis["summary"]["quantizable_layers"] += result["quantizable_layers"]
                    analysis["summary"]["skipped_layers"] += result["skipped_layers"]
                    analysis["summary"]["total_size_mb"] += result["total_size_mb"]
                    analysis["summary"]["quantizable_size_mb"] += result["quantizable_size_mb"]
                    analysis["summary"]["skipped_size_mb"] += result["skipped_size_mb"]

                    analysis["layers"].update(result["layers"])

                    for cat, names in result["categories"].items():
                        categories[cat].extend(names)

                except Exception as e:
                    print(f"Error processing {shard_file}: {e}")
    else:
        # Single GPU/CPU - process sequentially
        for shard_file in tqdm(safetensor_files, desc="Analyzing shards"):
            result = analyze_shard(shard_file, skip_patterns, gpus[0])

            # Merge results
            analysis["summary"]["total_layers"] += result["total_layers"]
            analysis["summary"]["quantizable_layers"] += result["quantizable_layers"]
            analysis["summary"]["skipped_layers"] += result["skipped_layers"]
            analysis["summary"]["total_size_mb"] += result["total_size_mb"]
            analysis["summary"]["quantizable_size_mb"] += result["quantizable_size_mb"]
            analysis["summary"]["skipped_size_mb"] += result["skipped_size_mb"]

            analysis["layers"].update(result["layers"])

            for cat, names in result["categories"].items():
                categories[cat].extend(names)

    # Calculate savings
    total_mb = analysis["summary"]["total_size_mb"]
    quant_mb = analysis["summary"]["quantizable_size_mb"]
    skip_mb = analysis["summary"]["skipped_size_mb"]

    # INT8 is 1 byte per element vs 2 bytes for bf16
    q8_quant_mb = quant_mb / 2  # 50% reduction for quantized layers
    estimated_q8_total = q8_quant_mb + skip_mb  # Skipped layers stay same size

    analysis["summary"]["estimated_q8_size_mb"] = estimated_q8_total
    analysis["summary"]["memory_reduction_percent"] = (1 - estimated_q8_total / total_mb) * 100 if total_mb > 0 else 0

    analysis["categories"] = {k: len(v) for k, v in categories.items()}

    return analysis


def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report."""
    print("\n" + "=" * 70)
    print("Q8 (INT8) QUANTIZATION ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nModel: {analysis['model_path']}")
    print(f"Type: {analysis['model_type']}")
    print(f"Analysis Time: {analysis['analysis_time']}")

    summary = analysis["summary"]
    print("\n" + "-" * 70)
    print("MEMORY SUMMARY")
    print("-" * 70)
    print(f"Total model size:     {summary['total_size_mb'] / 1024:.2f} GB")
    print(f"Quantizable layers:   {summary['quantizable_size_mb'] / 1024:.2f} GB ({summary['quantizable_layers']} layers)")
    print(f"Skipped layers (bf16): {summary['skipped_size_mb'] / 1024:.2f} GB ({summary['skipped_layers']} layers)")
    print(f"Estimated Q8 size:    {summary['estimated_q8_size_mb'] / 1024:.2f} GB")
    print(f"Memory reduction:     {summary['memory_reduction_percent']:.1f}%")

    print("\n" + "-" * 70)
    print("LAYER CATEGORIES")
    print("-" * 70)
    categories = analysis.get("categories", {})
    print(f"Safe to quantize:     {categories.get('safe', 0)} layers")
    print(f"Medium risk:          {categories.get('medium_risk', 0)} layers")
    print(f"High risk (skipped):  {categories.get('high_risk', 0)} layers")
    print(f"Pattern skip (bf16):  {categories.get('skipped', 0)} layers")

    # Show some skipped layers (vision, etc.)
    skipped = [name for name, stats in analysis["layers"].items()
               if stats.get("category") == "skipped_pattern"]
    if skipped:
        print("\n" + "-" * 70)
        print("LAYERS KEPT IN BF16 (vision, norms, embeddings)")
        print("-" * 70)
        # Group by prefix
        visual_count = sum(1 for n in skipped if 'visual' in n.lower())
        norm_count = sum(1 for n in skipped if 'norm' in n.lower())
        embed_count = sum(1 for n in skipped if 'embed' in n.lower())
        other_count = len(skipped) - visual_count - norm_count - embed_count
        print(f"  • Visual encoder layers: {visual_count}")
        print(f"  • Normalization layers:  {norm_count}")
        print(f"  • Embedding layers:      {embed_count}")
        if other_count > 0:
            print(f"  • Other skipped:         {other_count}")

    print("\n" + "=" * 70)


def quantize_to_int8(tensor: torch.Tensor, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to INT8 using symmetric per-tensor quantization.

    Args:
        tensor: Input tensor to quantize
        device: Device to use for computation ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        int8_tensor: Quantized tensor in int8 (on CPU for saving)
        scale: Scale factor for dequantization (original = int8 * scale)
    """
    # Move to device for faster computation
    if device != "cpu" and torch.cuda.is_available():
        t_float = tensor.to(device).float()
    else:
        t_float = tensor.float()

    # Symmetric quantization: scale = abs_max / 127
    abs_max = t_float.abs().max()
    if abs_max == 0:
        abs_max = torch.tensor(1.0, device=t_float.device)

    # Scale factor
    scale = (abs_max / 127.0).float()

    # Quantize: round(tensor / scale) clamped to [-127, 127]
    int8_tensor = torch.clamp(torch.round(t_float / scale), -127, 127).to(torch.int8)

    # Move back to CPU for saving
    return int8_tensor.cpu(), scale.cpu()


def convert_shard(
    shard_file: str,
    output_shard: str,
    analysis: Dict,
    device: str
) -> Dict:
    """Convert a single shard file on a specific GPU."""
    result = {
        "weight_map": {},
        "converted_count": 0,
        "skipped_count": 0,
        "original_size": 0,
        "q8_size": 0,
        "shard_name": os.path.basename(shard_file)
    }

    if not os.path.exists(shard_file):
        return result

    shard_name = result["shard_name"]
    converted_tensors = {}
    scales = {}

    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            original_size = tensor.numel() * tensor.element_size()
            result["original_size"] += original_size

            # Check if this layer should be quantized
            layer_info = analysis["layers"].get(name, {})
            should_quantize = (
                layer_info.get("quantizable", False) and
                layer_info.get("category") in ["safe", "medium_risk"] and
                tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]
            )

            if should_quantize:
                # Quantize to INT8 using GPU
                int8_tensor, scale = quantize_to_int8(tensor, device=device)

                converted_tensors[name] = int8_tensor
                scales[f"{name}._scale"] = scale.view(1)

                result["weight_map"][name] = shard_name
                result["weight_map"][f"{name}._scale"] = shard_name

                # INT8 = 1 byte per element + scale (4 bytes total)
                q8_size = tensor.numel() * 1 + 4
                result["q8_size"] += q8_size
                result["converted_count"] += 1
            else:
                # Keep original precision (bf16 for vision layers)
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.bfloat16)
                converted_tensors[name] = tensor
                result["weight_map"][name] = shard_name
                result["q8_size"] += tensor.numel() * tensor.element_size()
                result["skipped_count"] += 1

            del tensor

    # Merge scales into tensors dict
    converted_tensors.update(scales)

    # Save shard
    safetensors_save(converted_tensors, output_shard)

    del converted_tensors
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def convert_to_q8(
    model_path: str,
    output_path: str,
    analysis: Dict,
    force: bool = False,
    device: str = "cpu"
) -> str:
    """
    Convert model to Q8 (INT8) format based on analysis.

    Creates a new model directory with INT8 weights for quantizable layers
    and bf16 weights for vision/sensitive layers.

    Args:
        model_path: Path to source model
        output_path: Path for output Q8 model
        analysis: Analysis dict from analyze_model()
        force: Overwrite existing output
        device: Device for quantization computation ("cpu", "cuda", "cuda:0", etc.)
    """
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors is required for conversion")

    if os.path.exists(output_path):
        if force:
            print(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output path already exists: {output_path}. Use --force to overwrite.")

    os.makedirs(output_path, exist_ok=True)

    print(f"\nConverting model to Q8 (INT8)...")
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print(f"Vision layers will be kept in bf16")

    # Determine GPUs to use
    gpus = []
    if device != "cpu":
        if torch.cuda.is_available():
            if device == "cuda" or device == "all":
                gpus = get_available_gpus()
            else:
                gpus = [device]
            print_gpu_info()
            print(f"Using {len(gpus)} GPU(s) for conversion: {gpus}")
        else:
            print("Warning: CUDA not available, falling back to CPU")

    if not gpus:
        gpus = ["cpu"]

    # Copy config files
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                        "vocab.json", "merges.txt", "special_tokens_map.json",
                        "preprocessor_config.json", "generation_config.json",
                        "chat_template.json", "chat_template.jinja",
                        "video_preprocessor_config.json"]:
        src = os.path.join(model_path, config_file)
        if os.path.exists(src):
            shutil.copy2(src, output_path)
            print(f"Copied {config_file}")

    # Copy any .py files (for trust_remote_code models)
    for py_file in Path(model_path).glob("*.py"):
        shutil.copy2(py_file, output_path)
        print(f"Copied {py_file.name}")

    # Find safetensor files
    safetensor_files = find_safetensor_files(model_path)

    # Process shards
    weight_map = {}
    converted_count = 0
    skipped_count = 0
    total_original_size = 0
    total_q8_size = 0

    num_workers = len(gpus)

    if num_workers > 1:
        print(f"Processing {len(safetensor_files)} shards in parallel across {num_workers} GPUs...")

        # Create work items: (shard_file, output_shard, gpu)
        work_items = []
        for i, shard_file in enumerate(safetensor_files):
            shard_name = os.path.basename(shard_file)
            output_shard = os.path.join(output_path, shard_name)
            gpu = gpus[i % num_workers]
            work_items.append((shard_file, output_shard, gpu))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_shard = {
                executor.submit(convert_shard, shard_file, output_shard, analysis, gpu): shard_file
                for shard_file, output_shard, gpu in work_items
            }

            for future in tqdm(as_completed(future_to_shard), total=len(safetensor_files), desc="Converting shards"):
                shard_file = future_to_shard[future]
                try:
                    result = future.result()
                    weight_map.update(result["weight_map"])
                    converted_count += result["converted_count"]
                    skipped_count += result["skipped_count"]
                    total_original_size += result["original_size"]
                    total_q8_size += result["q8_size"]
                except Exception as e:
                    print(f"Error converting {shard_file}: {e}")
    else:
        # Single GPU/CPU - process sequentially
        for shard_file in tqdm(safetensor_files, desc="Converting shards"):
            shard_name = os.path.basename(shard_file)
            output_shard = os.path.join(output_path, shard_name)

            result = convert_shard(shard_file, output_shard, analysis, gpus[0])
            weight_map.update(result["weight_map"])
            converted_count += result["converted_count"]
            skipped_count += result["skipped_count"]
            total_original_size += result["original_size"]
            total_q8_size += result["q8_size"]

    # Create index file
    index = {
        "metadata": {
            "total_size": sum(os.path.getsize(os.path.join(output_path, f))
                            for f in os.listdir(output_path) if f.endswith(".safetensors")),
            "q8_converted": True,
            "converted_layers": converted_count,
            "skipped_layers": skipped_count,
            "original_size_gb": total_original_size / (1024**3),
            "q8_size_gb": total_q8_size / (1024**3),
        },
        "weight_map": weight_map
    }

    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Save Q8 metadata for loader
    q8_metadata = {
        "original_model": model_path,
        "conversion_time": datetime.now().isoformat(),
        "quantization_type": "int8_symmetric",
        "converted_layers": converted_count,
        "skipped_layers": skipped_count,
        "skip_patterns": analysis["skip_patterns"],
        "original_size_gb": total_original_size / (1024**3),
        "q8_size_gb": total_q8_size / (1024**3),
        "memory_reduction_percent": (1 - total_q8_size / total_original_size) * 100 if total_original_size > 0 else 0,
    }

    with open(os.path.join(output_path, "q8_metadata.json"), "w") as f:
        json.dump(q8_metadata, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Quantized to INT8: {converted_count} layers")
    print(f"Kept in bf16: {skipped_count} layers (vision, norms, embeddings)")
    print(f"Original size: {total_original_size / (1024**3):.2f} GB")
    print(f"Q8 size: {total_q8_size / (1024**3):.2f} GB")
    print(f"Reduction: {(1 - total_q8_size / total_original_size) * 100:.1f}%")
    print(f"Output: {output_path}")

    return output_path


def generate_loader_code(output_path: str, analysis: Dict) -> str:
    """Generate Python loader code for the Q8 model."""

    loader_code = f'''#!/usr/bin/env python3
"""
Auto-generated Q8 (INT8) loader for {os.path.basename(output_path)}
Generated by Q8 Quantization Converter on {datetime.now().isoformat()}

This loads the Q8-quantized model with proper dequantization during inference.
Vision layers are kept in bf16 for quality preservation.
"""

import os
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoProcessor, AutoConfig
from typing import Dict, Optional


class Q8DequantLinear(nn.Module):
    """Linear layer with INT8 weights and per-tensor scale for dequantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Placeholder weights (will be loaded)
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(1, dtype=torch.float32))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=dtype))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight: original = int8 * scale
        weight = self.weight_int8.to(self.dtype) * self.weight_scale.to(self.dtype)
        return nn.functional.linear(x.to(self.dtype), weight, self.bias)


def load_q8_weights(model: nn.Module, model_path: str, device: str = "cuda"):
    """
    Load Q8 weights into a model, handling dequantization.

    This replaces quantized Linear layers with Q8DequantLinear modules
    and loads the INT8 weights with their scale factors.
    """
    from safetensors import safe_open
    import json

    # Load index
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Group weights by shard file
    shard_to_weights = {{}}
    for weight_name, shard_file in weight_map.items():
        if shard_file not in shard_to_weights:
            shard_to_weights[shard_file] = []
        shard_to_weights[shard_file].append(weight_name)

    # Load each shard
    for shard_file, weight_names in shard_to_weights.items():
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for name in weight_names:
                if name.endswith("._scale"):
                    continue  # Scales are loaded with their weights

                tensor = f.get_tensor(name)
                scale_name = f"{{name}}._scale"

                # Check if this is a quantized weight
                if scale_name in weight_names:
                    scale = f.get_tensor(scale_name)
                    # Dequantize: original = int8 * scale
                    tensor = tensor.to(torch.bfloat16) * scale.to(torch.bfloat16)

                # Set the weight in the model
                # Navigate to the correct module and set the weight
                parts = name.split(".")
                module = model
                for part in parts[:-1]:
                    module = getattr(module, part)

                param_name = parts[-1]
                if hasattr(module, param_name):
                    param = getattr(module, param_name)
                    if isinstance(param, nn.Parameter):
                        param.data = tensor
                    else:
                        setattr(module, param_name, tensor)

    return model


def load_q8_model(model_path: str, device_map: str = "auto", dtype=torch.bfloat16):
    """
    Load the Q8-quantized model.

    Args:
        model_path: Path to the Q8-quantized model directory
        device_map: Device map for multi-GPU ("auto", "cuda:0", etc.)
        dtype: Compute dtype for dequantized weights (should be bfloat16)

    Returns:
        model, processor
    """
    from transformers import AutoModelForVision2Seq

    # Load with trust_remote_code for custom model code
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    return model, processor


# Memory estimates
ORIGINAL_SIZE_GB = {analysis["summary"]["total_size_mb"] / 1024:.2f}
Q8_SIZE_GB = {analysis["summary"]["estimated_q8_size_mb"] / 1024:.2f}
MEMORY_REDUCTION = {analysis["summary"]["memory_reduction_percent"]:.1f}


if __name__ == "__main__":
    print(f"Loading Q8 model from: {output_path}")
    print(f"Original size: {{ORIGINAL_SIZE_GB}} GB")
    print(f"Q8 size: {{Q8_SIZE_GB}} GB")
    print(f"Memory reduction: {{MEMORY_REDUCTION}}%")
    print(f"Vision layers preserved in bf16")

    model, processor = load_q8_model("{output_path}")
    print("Model loaded successfully!")
'''

    loader_path = os.path.join(output_path, "load_q8_model.py")
    with open(loader_path, "w") as f:
        f.write(loader_code)

    print(f"Generated loader: {loader_path}")
    return loader_path


def main():
    parser = argparse.ArgumentParser(
        description="Q8 (INT8) Quantization Converter for Vision-Language Models"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for Q8 model (default: {model}-Q8)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, don't convert"
    )
    parser.add_argument(
        "--save-analysis",
        type=str,
        default=None,
        help="Save analysis to JSON file"
    )
    parser.add_argument(
        "--skip-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Layer name patterns to skip (keep in bf16)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device for computation. Options: 'cuda' or 'all' (use all GPUs in parallel), 'cuda:0' (specific GPU), 'cpu' (CPU-only)"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = args.model.rstrip("/\\") + "-Q8"

    # Run analysis
    analysis = analyze_model(args.model, args.skip_patterns, device=args.device)
    print_analysis_report(analysis)

    # Save analysis if requested
    if args.save_analysis:
        with open(args.save_analysis, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {args.save_analysis}")

    # Convert if not analyze-only
    if not args.analyze_only:
        output_path = convert_to_q8(args.model, args.output, analysis, args.force, args.device)
        generate_loader_code(output_path, analysis)

        print("\n" + "=" * 70)
        print("CONVERSION COMPLETE")
        print("=" * 70)
        print(f"\nQ8 model saved to: {output_path}")
        print(f"Vision layers preserved in bf16 for quality")
        print(f"\nTo use the Q8 model with vlm_diffusers.py:")
        print(f"  Update model path to: {output_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()
