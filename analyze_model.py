#!/usr/bin/env python3
"""
Analyze model weights to understand FP8 conversion eligibility.
Run this on the machine with the model weights.
"""

import json
from pathlib import Path
from collections import defaultdict

# Skip patterns from convert_to_fp8.py
SKIP_PATTERNS = [
    'vae', 'vision_model', 'vision_aligner',
    'final_layer', 'patch_embed',
    'time_embed', 'time_embed_2', 'timestep_emb', 'guidance_emb', 'timestep_r_emb',
    'wg', 'wte', 'lm_head',
    'ln_f', 'layernorm', 'input_layernorm', 'post_attention_layernorm',
    'key_layernorm', 'query_layernorm',
]

def analyze_model(model_path: str):
    from safetensors import safe_open

    model_path = Path(model_path)
    skip_patterns = [p.lower() for p in SKIP_PATTERNS]

    # Find safetensors files
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        print(f"Found sharded model with {len(shard_files)} shards")
    else:
        single_file = model_path / "model.safetensors"
        if single_file.exists():
            shard_files = ["model.safetensors"]
            print("Found single safetensors file")
        else:
            print(f"ERROR: No safetensors files found in {model_path}")
            return

    # Analyze tensors
    stats = {
        'would_convert': [],
        'skip_pattern': defaultdict(list),
        'skip_not_weight': [],
        'skip_not_2d': [],
        'total_tensors': 0,
        'dtypes': defaultdict(int),
    }

    # Only analyze first shard for speed (patterns are consistent across shards)
    shard_path = model_path / shard_files[0]
    print(f"\nAnalyzing first shard: {shard_path.name}")

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        tensor_names = list(f.keys())
        print(f"Tensors in shard: {len(tensor_names)}")

        for name in tensor_names:
            tensor = f.get_tensor(name)
            stats['total_tensors'] += 1
            stats['dtypes'][str(tensor.dtype)] += 1

            # Check if it ends with .weight
            if not name.endswith('.weight'):
                stats['skip_not_weight'].append(name)
                continue

            # Check shape (must be 2D for linear layers)
            if tensor.ndim != 2:
                stats['skip_not_2d'].append((name, tensor.shape))
                continue

            # Check skip patterns
            name_lower = name.lower()
            matched_pattern = None
            for pattern in skip_patterns:
                if pattern in name_lower:
                    matched_pattern = pattern
                    break

            if matched_pattern:
                stats['skip_pattern'][matched_pattern].append(name)
            else:
                stats['would_convert'].append((name, tensor.shape, tensor.dtype))

    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    print(f"\nTotal tensors in first shard: {stats['total_tensors']}")

    print(f"\nDtypes found:")
    for dtype, count in sorted(stats['dtypes'].items()):
        print(f"  {dtype}: {count}")

    print(f"\nSkipped (not .weight): {len(stats['skip_not_weight'])}")
    if stats['skip_not_weight'][:3]:
        print(f"  Examples: {stats['skip_not_weight'][:3]}")

    print(f"\nSkipped (not 2D): {len(stats['skip_not_2d'])}")
    for name, shape in stats['skip_not_2d'][:5]:
        print(f"  {name}: shape={shape}")

    print(f"\nSkipped by pattern:")
    for pattern, names in sorted(stats['skip_pattern'].items()):
        print(f"  '{pattern}': {len(names)} tensors")
        if names[:2]:
            for n in names[:2]:
                print(f"    - {n}")

    print(f"\nWOULD CONVERT: {len(stats['would_convert'])} tensors")
    if stats['would_convert']:
        print("Examples:")
        for name, shape, dtype in stats['would_convert'][:10]:
            print(f"  {name}: shape={shape}, dtype={dtype}")
    else:
        print("  *** NO TENSORS ELIGIBLE FOR CONVERSION ***")
        print("\n  This means either:")
        print("  1. All .weight tensors match a skip pattern")
        print("  2. All .weight tensors are not 2D")
        print("  3. There are no .weight tensors")

    # Estimate total conversions across all shards
    if shard_files and stats['would_convert']:
        # Rough estimate: multiply by number of shards
        estimated_total = len(stats['would_convert']) * len(shard_files)
        print(f"\nEstimated total conversions across {len(shard_files)} shards: ~{estimated_total}")

    # Show some tensor names for debugging
    print("\n" + "="*60)
    print("SAMPLE TENSOR NAMES (first 20)")
    print("="*60)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for i, name in enumerate(list(f.keys())[:20]):
            tensor = f.get_tensor(name)
            print(f"  {name}")
            print(f"    shape={tensor.shape}, dtype={tensor.dtype}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_model.py <model_path>")
        print("Example: python analyze_model.py models/LLM/HunyuanImage-3.0-Instruct")
        sys.exit(1)

    analyze_model(sys.argv[1])
