"""
Z-Image LoRA Loader

A dedicated LoRA loader for Z-Image models that handles the specific key mappings
required between LoRA files and the Z-Image transformer architecture.

Key Mappings:
  LoRA: diffusion_model.layers.N.attention.out -> Model: layers.N.attention.to_out.0.weight
  LoRA: diffusion_model.layers.N.attention.to_q -> Model: layers.N.attention.to_q.weight
  LoRA: diffusion_model.layers.N.feed_forward.w1 -> Model: layers.N.feed_forward.w1.weight
  etc.
"""

import torch
try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


def lora_key_to_model_key(lora_key: str) -> str:
    """
    Convert a LoRA key to the corresponding model key.

    Example:
        'diffusion_model.layers.0.attention.out' -> 'layers.0.attention.to_out.0.weight'
        'diffusion_model.layers.0.attention.to_q' -> 'layers.0.attention.to_q.weight'
    """
    # Remove diffusion_model. prefix
    if lora_key.startswith('diffusion_model.'):
        model_key = lora_key[len('diffusion_model.'):]
    else:
        model_key = lora_key

    # Handle attention.out -> attention.to_out.0 transformation
    model_key = model_key.replace('.attention.out', '.attention.to_out.0')

    # Add .weight suffix
    model_key = model_key + '.weight'

    return model_key


def load_zimage_lora_patches(lora_state: dict, model) -> tuple[dict, dict]:
    """
    Create LoRA patches for a Z-Image model in Forge's patch format.

    This function is compatible with Forge's LoRA loading system and returns
    patches in the same format as the regular load_lora function.

    Args:
        lora_state: Dictionary of LoRA weights (already loaded from file)
        model: The Z-Image model (KModel wrapping ZImageTransformerWrapper)

    Returns:
        (patch_dict, remaining_dict) where:
        - patch_dict: {model_key: ("lora", (lora_up, lora_down, alpha, mid, dora_scale))}
        - remaining_dict: Unmatched LoRA keys
    """
    # Get model state dict keys for matching
    # KModel structure: KModel.diffusion_model (wrapper) -> wrapper.transformer (actual model)
    model_state_keys = set(model.state_dict().keys())

    # Debug: Print sample keys to understand structure
    sample_keys = list(model_state_keys)[:5]
    print(f"[Z-Image LoRA] Model state dict sample keys: {sample_keys}")

    # Group LoRA keys by base key
    lora_groups = {}
    for key in lora_state.keys():
        result = extract_lora_base_key(key)
        if result:
            base_key, key_type = result
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key][key_type] = key  # Store the full key name

    patch_dict = {}
    remaining_dict = {}
    loaded_keys = set()

    for lora_base_key, lora_key_names in lora_groups.items():
        # Convert LoRA key to model key (returns key without diffusion_model prefix)
        # e.g., 'diffusion_model.layers.0.attention.out' -> 'layers.0.attention.to_out.0.weight'
        model_key = lora_key_to_model_key(lora_base_key)

        # Try multiple prefixes to find the key in the model
        # KModel wraps: diffusion_model.transformer.{actual_key}
        actual_model_key = None
        prefixes_to_try = [
            '',  # Direct match
            'diffusion_model.',  # Standard prefix
            'diffusion_model.transformer.',  # Wrapped transformer
        ]

        for prefix in prefixes_to_try:
            candidate = prefix + model_key
            if candidate in model_state_keys:
                actual_model_key = candidate
                break

        if actual_model_key is None:
            # Key not found - add all related keys to remaining
            for key_type, full_key in lora_key_names.items():
                remaining_dict[full_key] = lora_state.get(full_key)
            continue

        # Get LoRA components
        up_key = lora_key_names.get('up')
        down_key = lora_key_names.get('down')
        alpha_key = lora_key_names.get('alpha')

        if up_key is None or down_key is None:
            continue

        lora_up = lora_state[up_key]
        lora_down = lora_state[down_key]
        alpha = lora_state[alpha_key].item() if alpha_key else None

        # Mark keys as loaded
        loaded_keys.add(up_key)
        loaded_keys.add(down_key)
        if alpha_key:
            loaded_keys.add(alpha_key)

        # Create patch in Forge format: ("lora", (up, down, alpha, mid, dora_scale))
        patch_dict[actual_model_key] = ("lora", (lora_up, lora_down, alpha, None, None))

    # Add unloaded keys to remaining_dict
    for key in lora_state.keys():
        if key not in loaded_keys:
            remaining_dict[key] = lora_state[key]

    print(f"[Z-Image LoRA] Created {len(patch_dict)} patches, {len(remaining_dict)} unmatched keys")

    return patch_dict, remaining_dict


def extract_lora_base_key(full_key: str) -> tuple[str, str] | None:
    """
    Extract the base key and type from a full LoRA key.

    Returns (base_key, type) where type is 'up', 'down', or 'alpha'
    Returns None if not a valid LoRA key.

    Supports two LoRA formats:
    - Kohya/WebUI format: .lora_up.weight, .lora_down.weight
    - PEFT/HuggingFace format: .lora_B.weight (up), .lora_A.weight (down)
    """
    # Kohya/WebUI format
    if full_key.endswith('.lora_up.weight'):
        return full_key[:-len('.lora_up.weight')], 'up'
    elif full_key.endswith('.lora_down.weight'):
        return full_key[:-len('.lora_down.weight')], 'down'
    # PEFT/HuggingFace format (lora_B = up, lora_A = down)
    elif full_key.endswith('.lora_B.weight'):
        return full_key[:-len('.lora_B.weight')], 'up'
    elif full_key.endswith('.lora_A.weight'):
        return full_key[:-len('.lora_A.weight')], 'down'
    elif full_key.endswith('.alpha'):
        return full_key[:-len('.alpha')], 'alpha'
    return None


def load_zimage_lora(lora_path: str, model: torch.nn.Module, strength: float = 1.0) -> dict:
    """
    Load a LoRA file and apply it to a Z-Image model.

    Args:
        lora_path: Path to the LoRA safetensors file
        model: The Z-Image transformer model
        strength: LoRA strength multiplier (default 1.0)

    Returns:
        dict with 'matched', 'unmatched', and 'applied' counts
    """
    # Load LoRA weights
    lora_state = load_file(lora_path)

    # Get model state dict
    model_state = model.state_dict()

    # Group LoRA keys by base key
    lora_groups = {}
    for key in lora_state.keys():
        result = extract_lora_base_key(key)
        if result:
            base_key, key_type = result
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key][key_type] = lora_state[key]

    matched = 0
    unmatched = 0
    applied = 0
    unmatched_keys = []

    # Apply LoRA weights
    for lora_base_key, lora_weights in lora_groups.items():
        model_key = lora_key_to_model_key(lora_base_key)

        if model_key not in model_state:
            unmatched += 1
            unmatched_keys.append(lora_base_key)
            continue

        matched += 1

        # Get LoRA components
        lora_up = lora_weights.get('up')
        lora_down = lora_weights.get('down')
        alpha = lora_weights.get('alpha')

        if lora_up is None or lora_down is None:
            continue

        # Calculate LoRA delta: up @ down * scale
        # Alpha scaling: if alpha exists, scale = alpha / rank
        rank = lora_down.shape[0]
        if alpha is not None:
            scale = alpha.item() / rank
        else:
            scale = 1.0

        scale *= strength

        # Compute the delta weight
        # lora_up shape: [out_features, rank]
        # lora_down shape: [rank, in_features]
        # delta shape: [out_features, in_features]
        delta = (lora_up @ lora_down) * scale

        # Get the parameter and apply delta
        param_name = model_key

        # Navigate to the actual parameter in the model
        parts = param_name.split('.')
        target = model
        for part in parts[:-1]:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)

        param = getattr(target, parts[-1])

        # Apply the delta
        with torch.no_grad():
            if param.shape == delta.shape:
                param.add_(delta.to(param.device, param.dtype))
                applied += 1
            else:
                print(f"[Z-Image LoRA] Shape mismatch for {model_key}: param={param.shape}, delta={delta.shape}")

    if unmatched_keys:
        print(f"[Z-Image LoRA] Unmatched keys ({len(unmatched_keys)}): {unmatched_keys[:5]}...")

    return {
        'matched': matched,
        'unmatched': unmatched,
        'applied': applied,
        'total_lora_layers': len(lora_groups)
    }


def apply_zimage_lora_to_state_dict(lora_path: str, model_state: dict, strength: float = 1.0) -> tuple[dict, dict]:
    """
    Apply LoRA weights to a model state dict (without requiring the model instance).

    Args:
        lora_path: Path to the LoRA safetensors file
        model_state: The model's state dict
        strength: LoRA strength multiplier

    Returns:
        (modified_state_dict, stats_dict)
    """
    lora_state = load_file(lora_path)

    # Group LoRA keys by base key
    lora_groups = {}
    for key in lora_state.keys():
        result = extract_lora_base_key(key)
        if result:
            base_key, key_type = result
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key][key_type] = lora_state[key]

    matched = 0
    applied = 0
    unmatched_keys = []

    # Create a copy of the state dict
    new_state = {k: v.clone() for k, v in model_state.items()}

    for lora_base_key, lora_weights in lora_groups.items():
        model_key = lora_key_to_model_key(lora_base_key)

        if model_key not in new_state:
            unmatched_keys.append(lora_base_key)
            continue

        matched += 1

        lora_up = lora_weights.get('up')
        lora_down = lora_weights.get('down')
        alpha = lora_weights.get('alpha')

        if lora_up is None or lora_down is None:
            continue

        rank = lora_down.shape[0]
        scale = (alpha.item() / rank if alpha is not None else 1.0) * strength

        delta = (lora_up @ lora_down) * scale

        if new_state[model_key].shape == delta.shape:
            new_state[model_key] = new_state[model_key] + delta.to(new_state[model_key].dtype)
            applied += 1

    stats = {
        'matched': matched,
        'unmatched': len(unmatched_keys),
        'applied': applied,
        'unmatched_keys': unmatched_keys[:10] if unmatched_keys else []
    }

    return new_state, stats


# Standalone test
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python zimage_lora.py <model.safetensors> <lora.safetensors> [strength]")
        sys.exit(1)

    model_path = sys.argv[1]
    lora_path = sys.argv[2]
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    print(f"Loading model: {model_path}")
    model_state = load_file(model_path)

    print(f"Applying LoRA: {lora_path} (strength={strength})")
    new_state, stats = apply_zimage_lora_to_state_dict(lora_path, model_state, strength)

    print(f"\nResults:")
    print(f"  Matched: {stats['matched']}")
    print(f"  Applied: {stats['applied']}")
    print(f"  Unmatched: {stats['unmatched']}")
    if stats['unmatched_keys']:
        print(f"  Sample unmatched: {stats['unmatched_keys']}")
