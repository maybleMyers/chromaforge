"""
AuraFlow T5 Safetensors Analysis and Merger
Analyzes, compares, and merges T5 encoder weights in AuraFlow checkpoints
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse

try:
    from safetensors.torch import load_file, save_file
    import torch
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("pip install safetensors torch")
    exit(1)


def get_t5_keys(state_dict: Dict) -> List[str]:
    """Extract all T5-related keys from a state dict."""
    t5_prefixes = ['text_encoder', 't5', 'conditioner', 'encoder']
    t5_keys = []
    
    for key in state_dict.keys():
        key_lower = key.lower()
        if any(prefix in key_lower for prefix in t5_prefixes):
            t5_keys.append(key)
    
    return sorted(t5_keys)


def analyze_checkpoint(file_path: str) -> Tuple[Dict, List[str], Dict]:
    """Analyze a safetensors checkpoint and return T5 info."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*60}")
    
    state_dict = load_file(file_path)
    t5_keys = get_t5_keys(state_dict)
    
    # Calculate sizes
    total_params = sum(v.numel() for v in state_dict.values())
    t5_params = sum(state_dict[k].numel() for k in t5_keys) if t5_keys else 0
    
    # Get shapes info
    shapes_info = {k: tuple(state_dict[k].shape) for k in t5_keys}
    
    print(f"\nTotal keys in checkpoint: {len(state_dict)}")
    print(f"T5 encoder keys found: {len(t5_keys)}")
    print(f"Total parameters: {total_params:,}")
    print(f"T5 parameters: {t5_params:,} ({t5_params/total_params*100:.1f}%)")
    
    if t5_keys:
        print(f"\nT5 Key Prefixes (first 10):")
        for key in t5_keys[:10]:
            shape = state_dict[key].shape
            dtype = state_dict[key].dtype
            print(f"  {key}: {shape} [{dtype}]")
        if len(t5_keys) > 10:
            print(f"  ... and {len(t5_keys) - 10} more keys")
    
    return state_dict, t5_keys, shapes_info


def compare_t5_encoders(file1: str, file2: str):
    """Compare T5 encoders between two checkpoints."""
    print(f"\n{'#'*60}")
    print("COMPARING T5 ENCODERS")
    print(f"{'#'*60}")
    
    sd1, keys1, shapes1 = analyze_checkpoint(file1)
    sd2, keys2, shapes2 = analyze_checkpoint(file2)
    
    # Compare key sets
    keys1_set = set(keys1)
    keys2_set = set(keys2)
    
    common_keys = keys1_set & keys2_set
    only_in_1 = keys1_set - keys2_set
    only_in_2 = keys2_set - keys1_set
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Common T5 keys: {len(common_keys)}")
    print(f"Only in file 1: {len(only_in_1)}")
    print(f"Only in file 2: {len(only_in_2)}")
    
    if only_in_1:
        print(f"\nKeys only in {Path(file1).name} (first 5):")
        for key in sorted(only_in_1)[:5]:
            print(f"  {key}")
    
    if only_in_2:
        print(f"\nKeys only in {Path(file2).name} (first 5):")
        for key in sorted(only_in_2)[:5]:
            print(f"  {key}")
    
    # Compare weights for common keys
    if common_keys:
        print(f"\nWeight Comparison (sample of 5 common keys):")
        for key in sorted(common_keys)[:5]:
            tensor1 = sd1[key]
            tensor2 = sd2[key]
            
            if tensor1.shape == tensor2.shape:
                diff = (tensor1 - tensor2).abs().mean().item()
                max_diff = (tensor1 - tensor2).abs().max().item()
                identical = torch.equal(tensor1, tensor2)
                print(f"\n  {key}:")
                print(f"    Shape: {tensor1.shape}")
                print(f"    Identical: {identical}")
                print(f"    Mean diff: {diff:.6f}")
                print(f"    Max diff: {max_diff:.6f}")
            else:
                print(f"\n  {key}: SHAPE MISMATCH")
                print(f"    File 1: {tensor1.shape}")
                print(f"    File 2: {tensor2.shape}")


def merge_t5_encoder(source_file: str, target_file: str, output_file: str, 
                     prefix_map: Dict[str, str] = None):
    """
    Merge T5 encoder from source into target checkpoint.
    
    Args:
        source_file: Path to checkpoint with T5 to extract
        target_file: Path to checkpoint to merge T5 into
        output_file: Path for output merged checkpoint
        prefix_map: Optional mapping of key prefixes for renaming
    """
    print(f"\n{'#'*60}")
    print("MERGING T5 ENCODER")
    print(f"{'#'*60}")
    
    # Load both checkpoints
    print("\nLoading source checkpoint...")
    source_sd = load_file(source_file)
    source_t5_keys = get_t5_keys(source_sd)
    
    print("Loading target checkpoint...")
    target_sd = load_file(target_file)
    target_t5_keys = get_t5_keys(target_sd)
    
    if not source_t5_keys:
        print("ERROR: No T5 keys found in source file!")
        return
    
    print(f"\nFound {len(source_t5_keys)} T5 keys in source")
    print(f"Found {len(target_t5_keys)} T5 keys in target (will be replaced)")
    
    # Create merged state dict
    merged_sd = target_sd.copy()
    
    # Remove old T5 keys from target
    for key in target_t5_keys:
        if key in merged_sd:
            del merged_sd[key]
    
    # Add T5 keys from source (with optional renaming)
    keys_added = 0
    for key in source_t5_keys:
        new_key = key
        
        # Apply prefix mapping if provided
        if prefix_map:
            for old_prefix, new_prefix in prefix_map.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
        
        merged_sd[new_key] = source_sd[key]
        keys_added += 1
    
    print(f"\nMerge summary:")
    print(f"  Removed {len(target_t5_keys)} old T5 keys")
    print(f"  Added {keys_added} new T5 keys")
    print(f"  Total keys in output: {len(merged_sd)}")
    
    # Save merged checkpoint
    print(f"\nSaving merged checkpoint to: {output_file}")
    save_file(merged_sd, output_file)
    
    print("✓ Merge complete!")
    
    # Verify output
    verify_sd = load_file(output_file)
    verify_t5_keys = get_t5_keys(verify_sd)
    print(f"\nVerification: Output file has {len(verify_t5_keys)} T5 keys")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and merge T5 encoders in AuraFlow safetensors checkpoints"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a checkpoint')
    analyze_parser.add_argument('file', help='Path to safetensors file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two checkpoints')
    compare_parser.add_argument('file1', help='First safetensors file')
    compare_parser.add_argument('file2', help='Second safetensors file')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge T5 encoder')
    merge_parser.add_argument('source', help='Source file (with T5 to extract)')
    merge_parser.add_argument('target', help='Target file (to merge T5 into)')
    merge_parser.add_argument('output', help='Output file path')
    merge_parser.add_argument('--prefix-from', help='Old key prefix to replace')
    merge_parser.add_argument('--prefix-to', help='New key prefix')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_checkpoint(args.file)
    
    elif args.command == 'compare':
        compare_t5_encoders(args.file1, args.file2)
    
    elif args.command == 'merge':
        prefix_map = None
        if args.prefix_from and args.prefix_to:
            prefix_map = {args.prefix_from: args.prefix_to}
        merge_t5_encoder(args.source, args.target, args.output, prefix_map)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()