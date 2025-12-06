from __future__ import annotations

import os
import re
import torch
import network
import functools

from backend.args import dynamic_args
from modules import shared, sd_models, errors, scripts
from backend.utils import load_torch_file
from backend.patcher.lora import model_lora_keys_clip, model_lora_keys_unet, load_lora
from backend.lora.zimage_lora import load_zimage_lora_patches, get_lora_manager, clear_lora_manager


def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename='default', online_mode=False):
    model_flag = type(model.model).__name__ if model is not None else 'default'

    # Check if this is a Z-Image model (detect by wrapper class or model config)
    is_zimage = False
    if model is not None:
        model_config = getattr(model.model, 'config', None)
        if model_config is not None and getattr(model_config, 'is_zimage', False):
            is_zimage = True
            print(f'[LORA] Z-Image model detected via config.is_zimage')
        # Also check by class name pattern
        if 'ZImage' in model_flag:
            is_zimage = True
            print(f'[LORA] Z-Image model detected via class name: {model_flag}')

    # Use dedicated Z-Image LoRA loader if applicable
    if is_zimage and model is not None:
        print(f'[LORA] Using Z-Image LoRA loader for {filename}')
        lora_unet, lora_unmatch = load_zimage_lora_patches(lora, model.model)
        clip_keys = model_lora_keys_clip(clip.cond_stage_model) if clip is not None else {}
        lora_clip, lora_unmatch = load_lora(lora_unmatch, clip_keys)
    else:
        unet_keys = model_lora_keys_unet(model.model) if model is not None else {}
        clip_keys = model_lora_keys_clip(clip.cond_stage_model) if clip is not None else {}

        lora_unmatch = lora
        lora_unet, lora_unmatch = load_lora(lora_unmatch, unet_keys)
        lora_clip, lora_unmatch = load_lora(lora_unmatch, clip_keys)

    #if len(lora_unmatch) > 12:
        #print(f'[LORA] LoRA version mismatch for {model_flag}: {filename}')
        #return model, clip

    if len(lora_unmatch) > 0:
        print(f'[LORA] Loading {filename} for {model_flag} with unmatched keys {list(lora_unmatch.keys())}')

    new_model = model.clone() if model is not None else None
    new_clip = clip.clone() if clip is not None else None

    if new_model is not None and len(lora_unet) > 0:
        loaded_keys = new_model.add_patches(filename=filename, patches=lora_unet, strength_patch=strength_model, online_mode=online_mode)
        skipped_keys = [item for item in lora_unet if item not in loaded_keys]
        if len(skipped_keys) > 12:
            print(f'[LORA] Mismatch {filename} for {model_flag}-UNet with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys')
        else:
            print(f'[LORA] Loaded {filename} for {model_flag}-UNet with {len(loaded_keys)} keys at weight {strength_model} (skipped {len(skipped_keys)} keys) with on_the_fly = {online_mode}')
            model = new_model

    if new_clip is not None and len(lora_clip) > 0:
        loaded_keys = new_clip.add_patches(filename=filename, patches=lora_clip, strength_patch=strength_clip, online_mode=online_mode)
        skipped_keys = [item for item in lora_clip if item not in loaded_keys]
        if len(skipped_keys) > 12:
            print(f'[LORA] Mismatch {filename} for {model_flag}-CLIP with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys')
        else:
            print(f'[LORA] Loaded {filename} for {model_flag}-CLIP with {len(loaded_keys)} keys at weight {strength_clip} (skipped {len(skipped_keys)} keys) with on_the_fly = {online_mode}')
            clip = new_clip

    return model, clip


@functools.lru_cache(maxsize=5)
def load_lora_state_dict(filename):
    return load_torch_file(filename, safe_load=True)


def load_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    return net


def _is_zimage_model(model):
    """Check if the model is a Z-Image model."""
    if model is None:
        return False
    model_config = getattr(model.model, 'config', None)
    if model_config is not None and getattr(model_config, 'is_zimage', False):
        return True
    model_flag = type(model.model).__name__
    if 'ZImage' in model_flag:
        return True
    return False


def _load_zimage_loras_direct(current_sd, compiled_lora_targets):
    """
    Load LoRAs for Z-Image models using direct weight merge.

    This approach modifies model weights directly instead of using patches,
    which is more reliable for multiple LoRAs.
    """
    model = current_sd.forge_objects.unet
    clip = current_sd.forge_objects.clip

    # Get or create LoRA manager for the model
    lora_manager = get_lora_manager(model.model)

    # Prepare LoRA configs for the transformer (UNet equivalent)
    unet_lora_configs = []
    clip_lora_configs = []

    for filename, strength_model, strength_clip, online_mode in compiled_lora_targets:
        if strength_model != 0:
            unet_lora_configs.append((filename, strength_model))
        if strength_clip != 0:
            clip_lora_configs.append((filename, strength_clip))

    # Apply LoRAs to transformer using direct merge
    if unet_lora_configs:
        print(f"[Z-Image LoRA] Using direct merge for {len(unet_lora_configs)} LoRA(s)")
        stats = lora_manager.apply_loras(unet_lora_configs)
        print(f"[Z-Image LoRA] Direct merge complete: {stats['applied']} layers modified, "
              f"{stats['unmatched']} unmatched keys")

    # Apply CLIP LoRAs using the standard patch system (works fine for single models)
    if clip_lora_configs and clip is not None:
        new_clip = clip.clone()
        for filename, strength_clip in clip_lora_configs:
            lora_sd = load_lora_state_dict(filename)
            clip_keys = model_lora_keys_clip(clip.cond_stage_model)
            lora_clip, lora_unmatch = load_lora(lora_sd, clip_keys)

            if len(lora_clip) > 0:
                loaded_keys = new_clip.add_patches(
                    filename=filename,
                    patches=lora_clip,
                    strength_patch=strength_clip,
                    online_mode=False  # Always use offline mode for CLIP
                )
                print(f"[Z-Image LoRA] Loaded CLIP LoRA from {filename} with {len(loaded_keys)} keys")

        current_sd.forge_objects.clip = new_clip


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    global lora_state_dict_cache

    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return

    loaded_networks.clear()

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        try:
            net = load_network(name, network_on_disk)
        except Exception as e:
            errors.display(e, f"loading network {network_on_disk.filename}")
            continue
        net.mentioned_name = name
        network_on_disk.read_hash()
        loaded_networks.append(net)

    online_mode = dynamic_args.get('online_lora', False)

    if current_sd.forge_objects.unet.model.storage_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        online_mode = False

    compiled_lora_targets = []
    for a, b, c in zip(networks_on_disk, unet_multipliers, te_multipliers):
        compiled_lora_targets.append([a.filename, b, c, online_mode])

    compiled_lora_targets_hash = str(compiled_lora_targets)

    if current_sd.current_lora_hash == compiled_lora_targets_hash:
        return

    current_sd.current_lora_hash = compiled_lora_targets_hash

    # Check if this is a Z-Image model - use direct merge approach
    is_zimage = _is_zimage_model(current_sd.forge_objects.unet)

    if is_zimage:
        # Z-Image: Use direct merge for transformer, standard patches for CLIP
        print("[Z-Image LoRA] Using direct merge approach for multiple LoRA support")

        # Debug: Check model identity
        model = current_sd.forge_objects.unet.model
        print(f"[Z-Image LoRA DEBUG] Model id: {id(model)}, type: {type(model).__name__}")

        # Clear any existing LoRA manager state
        lora_manager = get_lora_manager(model)
        print(f"[Z-Image LoRA DEBUG] Manager has {len(lora_manager.weight_backup)} backed up weights, "
              f"{len(lora_manager.applied_loras)} applied LoRAs")
        lora_manager.clear_loras()

        # Reset CLIP to original (we'll re-apply CLIP LoRAs)
        current_sd.forge_objects.clip = current_sd.forge_objects_original.clip

        # IMPORTANT: Clear the patcher's lora_patches to prevent interference with direct merge
        # The patcher's refresh_loras() is called when loading to GPU - we don't want it doing anything
        if hasattr(current_sd.forge_objects.unet, 'lora_patches'):
            if current_sd.forge_objects.unet.lora_patches:
                print(f"[Z-Image LoRA DEBUG] Clearing {len(current_sd.forge_objects.unet.lora_patches)} patches from patcher")
            current_sd.forge_objects.unet.lora_patches = {}

        # Also reset the lora_loader's state to prevent it from restoring/patching
        if hasattr(current_sd.forge_objects.unet, 'lora_loader'):
            loader = current_sd.forge_objects.unet.lora_loader
            if loader.backup:
                print(f"[Z-Image LoRA DEBUG] Clearing {len(loader.backup)} backups from lora_loader")
                loader.backup = {}
            loader.loaded_hash = str([])  # Reset hash so it doesn't skip refresh

        # Apply LoRAs using direct merge
        _load_zimage_loras_direct(current_sd, compiled_lora_targets)

        # Final verification: sample a weight to confirm modification
        try:
            sample_key = list(lora_manager.weight_backup.keys())[0] if lora_manager.weight_backup else None
            if sample_key:
                param = lora_manager._get_parameter(sample_key)
                backup = lora_manager.weight_backup[sample_key]
                # Compare first 3 values
                current_vals = param.data.flatten()[:3].tolist()
                backup_vals = backup.flatten()[:3].tolist()
                print(f"[Z-Image LoRA DEBUG] Final verification after direct merge:")
                print(f"[Z-Image LoRA DEBUG] Sample key: {sample_key[:50]}...")
                print(f"[Z-Image LoRA DEBUG] Current (with LoRA): {[f'{v:.8f}' for v in current_vals]}")
                print(f"[Z-Image LoRA DEBUG] Backup (original):   {[f'{v:.8f}' for v in backup_vals]}")
                print(f"[Z-Image LoRA DEBUG] Model device: {param.device}, patcher state: lora_patches={len(getattr(current_sd.forge_objects.unet, 'lora_patches', {}))}")
        except Exception as e:
            print(f"[Z-Image LoRA DEBUG] Verification error: {e}")
    else:
        # Standard models: Use the original patch-based approach
        current_sd.forge_objects.unet = current_sd.forge_objects_original.unet
        current_sd.forge_objects.clip = current_sd.forge_objects_original.clip

        for filename, strength_model, strength_clip, online_mode in compiled_lora_targets:
            lora_sd = load_lora_state_dict(filename)
            current_sd.forge_objects.unet, current_sd.forge_objects.clip = load_lora_for_models(
                current_sd.forge_objects.unet, current_sd.forge_objects.clip, lora_sd, strength_model, strength_clip,
                filename=filename, online_mode=online_mode)

    current_sd.forge_objects_after_applying_lora = current_sd.forge_objects.shallow_copy()
    return


def process_network_files(names: list[str] | None = None):
    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    for filename in candidates:
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
