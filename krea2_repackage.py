#!/usr/bin/env python
"""Repackage a Krea 2 Diffusers checkpoint folder into single safetensors files
that ChromaForge can load with the 'krea2' UI preset.

Produces:
  models/Stable-diffusion/<name>.safetensors   transformer (reference key naming)
  models/text_encoder/<name>-qwen3-vl-4b.safetensors   Qwen3-VL text encoder
  models/VAE/qwen-image-vae.safetensors        Qwen-Image VAE

Usage:
  python krea2_repackage.py --source krea-2/Krea-2-Turbo --name krea2-turbo

Then in the UI: pick the 'krea2' preset, select the transformer as checkpoint and
the text encoder + VAE files in the "VAE / Text Encoder" dropdown.
"""

import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def repackage_transformer(source: str, output: str):
    import sys
    sys.path.insert(0, REPO_ROOT)
    from backend.nn.krea2 import convert_diffusers_krea2_state_dict

    transformer_dir = os.path.join(source, 'transformer')
    index_path = os.path.join(transformer_dir, 'diffusion_pytorch_model.safetensors.index.json')

    if os.path.exists(index_path):
        with open(index_path, 'rt', encoding='utf-8') as f:
            index = json.load(f)
        shards = sorted(set(index['weight_map'].values()))
    else:
        shards = ['diffusion_pytorch_model.safetensors']

    state_dict = {}
    for shard in shards:
        print(f'Reading {shard} ...')
        state_dict.update(load_file(os.path.join(transformer_dir, shard)))

    print(f'Converting {len(state_dict)} keys to reference (SingleStreamDiT) naming ...')
    state_dict = convert_diffusers_krea2_state_dict(state_dict)

    print(f'Saving {output} ...')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_file(state_dict, output, metadata={'format': 'pt', 'modelspec.architecture': 'krea-2'})
    print(f'Transformer saved: {output} ({os.path.getsize(output) / 1024**3:.2f} GB)')


def copy_single_file(src: str, dst: str, what: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f'{what} already exists: {dst}')
        return
    print(f'Copying {what} to {dst} ...')
    shutil.copyfile(src, dst)
    print(f'{what} saved: {dst} ({os.path.getsize(dst) / 1024**3:.2f} GB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default=os.path.join(REPO_ROOT, 'krea-2', 'Krea-2-Turbo'),
                        help='Path to the cloned Krea 2 Diffusers checkpoint folder')
    parser.add_argument('--name', default=None,
                        help='Base name for output files (default: derived from source folder name)')
    parser.add_argument('--models-dir', default=os.path.join(REPO_ROOT, 'models'),
                        help='ChromaForge models directory')
    parser.add_argument('--skip-transformer', action='store_true')
    parser.add_argument('--skip-text-encoder', action='store_true')
    parser.add_argument('--skip-vae', action='store_true')
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    if not os.path.isdir(os.path.join(source, 'transformer')):
        raise SystemExit(f'{source} does not look like a Krea 2 checkpoint folder (no transformer/ subfolder)')

    name = args.name or os.path.basename(source.rstrip('/')).lower()

    if not args.skip_transformer:
        repackage_transformer(source, os.path.join(args.models_dir, 'Stable-diffusion', f'{name}.safetensors'))

    if not args.skip_text_encoder:
        copy_single_file(
            os.path.join(source, 'text_encoder', 'model.safetensors'),
            os.path.join(args.models_dir, 'text_encoder', f'{name}-qwen3-vl-4b.safetensors'),
            'Text encoder',
        )

    if not args.skip_vae:
        copy_single_file(
            os.path.join(source, 'vae', 'diffusion_pytorch_model.safetensors'),
            os.path.join(args.models_dir, 'VAE', 'qwen-image-vae.safetensors'),
            'VAE',
        )

    print('Done.')


if __name__ == '__main__':
    main()
