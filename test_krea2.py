"""Smoke test for the Krea 2 integration: loads the repackaged single-file models,
runs text encoding, a VAE round trip, and one transformer forward pass.

Run: venv/bin/python test_krea2.py
"""

import os
import sys
import time

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'packages_3rdparty'))

import types

# Outside the webui, memory_management's CPU-swap path can't import the real
# main_entry (circular import through modules.shared); stub the one attribute it reads.
_main_entry_stub = types.ModuleType('modules_forge.main_entry')
_main_entry_stub.user_specified_model_memory = None
sys.modules['modules_forge.main_entry'] = _main_entry_stub

import torch

from backend import memory_management
from backend.diffusion_engine.krea2 import Krea2
from backend.utils import load_torch_file


def main():
    t0 = time.time()
    print('=== Loading state dicts ===')
    state_dicts = {
        'transformer': load_torch_file('models/Stable-diffusion/krea2-turbo.safetensors'),
        'text_encoder': load_torch_file('models/text_encoder/krea2-qwen3-vl-4b.safetensors'),
        'vae': load_torch_file('models/VAE/qwen-image-vae.safetensors'),
    }
    print(f'state dicts loaded in {time.time() - t0:.1f}s')

    t0 = time.time()
    print('=== Building engine ===')
    engine = Krea2(state_dicts=state_dicts)
    del state_dicts
    print(f'engine built in {time.time() - t0:.1f}s')

    t0 = time.time()
    print('=== Text encoding ===')
    cond = engine.get_learned_conditioning(['a fox walking in the snow'])
    print('crossattn:', tuple(cond['crossattn'].shape), cond['crossattn'].dtype)
    print('attention_mask:', tuple(cond['attention_mask'].shape), 'valid tokens:', int(cond['attention_mask'].sum()))
    assert cond['crossattn'].shape[1] == 512 and cond['crossattn'].shape[2] == 12
    assert torch.isfinite(cond['crossattn'].float()).all(), 'NaN/Inf in text conditioning!'
    print(f'text encoded in {time.time() - t0:.1f}s')

    t0 = time.time()
    print('=== VAE round trip (256x256) ===')
    device = memory_management.get_torch_device()
    x = torch.rand(1, 3, 256, 256) * 2.0 - 1.0
    z = engine.encode_first_stage(x)
    print('latent:', tuple(z.shape), z.dtype)
    assert z.shape == (1, 16, 32, 32)
    assert torch.isfinite(z.float()).all(), 'NaN/Inf in latent!'
    decoded = engine.decode_first_stage(z)
    print('decoded:', tuple(decoded.shape), 'range:', float(decoded.min()), float(decoded.max()))
    assert decoded.shape == (1, 3, 256, 256)
    print(f'vae round trip in {time.time() - t0:.1f}s')

    t0 = time.time()
    print('=== Transformer forward (512x512 latent grid) ===')
    unet = engine.forge_objects.unet
    memory_management.load_models_gpu([unet], memory_required=unet.model.memory_required((1, 16, 64, 64)))

    latent = torch.randn(1, 16, 64, 64, device=device)
    sigma = torch.tensor([1.0], device=device)
    c_crossattn = {
        'crossattn': cond['crossattn'].to(device),
        'attention_mask': cond['attention_mask'].to(device),
    }
    with torch.inference_mode():
        denoised = unet.model.apply_model(latent, sigma, c_crossattn=c_crossattn)
    print('denoised:', tuple(denoised.shape), 'std:', float(denoised.std()))
    assert denoised.shape == latent.shape
    assert torch.isfinite(denoised).all(), 'NaN/Inf in transformer output!'
    print(f'transformer forward in {time.time() - t0:.1f}s')

    print('=== ALL OK ===')


if __name__ == '__main__':
    main()
