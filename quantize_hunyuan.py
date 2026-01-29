#!/usr/bin/env python3
"""
Quantize HunyuanImage-3.0 with Quanto int4, layer by layer on GPU.
Saves quantized weights for fast loading in hunyuan_image.py.

Usage:
    python quantize_hunyuan.py --model-path models/LLM/HunyuanImage-3.0-Instruct

Requirements:
    pip install optimum-quanto safetensors transformers torch tqdm
"""

import os
import shutil
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from optimum.quanto import quantize, qint4, freeze, quantization_map

# Modules to keep in full precision (vision, VAE, embeddings, gates, layernorms)
# These are small but critical for image quality and numerical stability
SKIP_MODULES = [
    # VAE - pixel reconstruction quality (encoder + decoder)
    'vae',
    # Vision Encoder - semantic understanding (SigLIP2)
    'vision_model', 'vision_aligner',
    # Image Generation Head - latent output
    'final_layer', 'patch_embed',
    # Timestep/Guidance Embeddings - diffusion timing
    'time_embed', 'time_embed_2', 'timestep_emb', 'guidance_emb', 'timestep_r_emb',
    # MoE Router Gates - MUST stay FP32 for numerical stability
    'wg',
    # Token Embeddings
    'wte', 'lm_head',
    # All LayerNorms - must stay FP32 for stability
    'ln_f', 'layernorm', 'input_layernorm', 'post_attention_layernorm',
    'key_layernorm', 'query_layernorm',  # QK attention layernorms
]


def should_skip(name):
    """Check if module should be skipped (kept in full precision)."""
    return any(skip in name for skip in SKIP_MODULES)


def quantize_on_gpu(model, device='cuda'):
    """Quantize model layer by layer on GPU for speed."""
    device = torch.device(device)

    # Get all named modules that have parameters
    modules_to_quantize = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if not should_skip(name):
                modules_to_quantize.append((name, module))

    print(f"Found {len(modules_to_quantize)} modules to quantize")
    skipped = sum(1 for name, _ in model.named_modules()
                  if hasattr(_, 'weight') and _.weight is not None and should_skip(name))
    print(f"Skipping {skipped} modules (vision/VAE/embeddings/gates)")

    for name, module in tqdm(modules_to_quantize, desc="Quantizing on GPU"):
        try:
            # Move to GPU
            module.to(device)

            # Quantize this module
            quantize(module, weights=qint4)
            freeze(module)

            # Move back to CPU to free GPU memory
            module.to('cpu')
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nWarning: Could not quantize {name}: {e}")
            # Move back to CPU even on error
            try:
                module.to('cpu')
            except:
                pass

    return model


def main():
    parser = argparse.ArgumentParser(description='Quantize HunyuanImage-3.0 with Quanto int4')
    parser.add_argument('--model-path', required=True, help='Path to HunyuanImage model')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: model_path/quanto_int4)')
    parser.add_argument('--device', default='cuda', help='Device for quantization (default: cuda)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model_path, 'quanto_int4')
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, 'model_quanto_int4.safetensors')
    map_path = os.path.join(output_dir, 'quantization_map.json')

    # Check if already quantized
    if os.path.exists(weights_path) and os.path.exists(map_path):
        print(f"Quantized model already exists at {output_dir}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    print(f"Loading model from {args.model_path}...")
    print("This will load to CPU first, then quantize each layer on GPU.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='cpu',  # Load to CPU first
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"\nModel loaded. Starting GPU quantization...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model = quantize_on_gpu(model, device=args.device)

    print(f"\nSaving quantized weights to {output_dir}...")

    # Save quantized state dict
    print("Saving state dict (this may take a while for large models)...")
    state_dict = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in model.state_dict().items()}
    save_file(state_dict, weights_path)
    print(f"Saved: {weights_path}")

    # Save quantization map
    qmap = quantization_map(model)
    with open(map_path, 'w') as f:
        json.dump(qmap, f, indent=2)
    print(f"Saved: {map_path}")

    # Copy config and model files to make a standalone model folder
    print("\nCopying config and model files...")
    model_path = Path(args.model_path)
    output_path = Path(output_dir)

    # Config files to copy
    config_files = [
        'config.json', 'generation_config.json',
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    ]
    for f in config_files:
        src = model_path / f
        if src.exists():
            shutil.copy(src, output_path / f)
            print(f"  Copied: {f}")

    # Copy Python model files (needed for trust_remote_code)
    for py_file in model_path.glob('*.py'):
        shutil.copy(py_file, output_path / py_file.name)
        print(f"  Copied: {py_file.name}")

    # Print file sizes
    weights_size = os.path.getsize(weights_path) / (1024**3)
    print(f"\nQuantized model size: {weights_size:.2f} GB")
    print(f"Output folder: {output_dir}")
    print("Done! You can now load the quantized model in hunyuan_image.py")


if __name__ == '__main__':
    main()
