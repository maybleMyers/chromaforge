"""
Z-Image LoRA Loader

A dedicated LoRA loader for Z-Image models that handles the specific key mappings
required between LoRA files and the Z-Image transformer architecture.

Key Mappings:
  LoRA: diffusion_model.layers.N.attention.out -> Model: layers.N.attention.to_out.0.weight
  LoRA: diffusion_model.layers.N.attention.to_q -> Model: layers.N.attention.to_q.weight
  LoRA: diffusion_model.layers.N.feed_forward.w1 -> Model: layers.N.feed_forward.w1.weight
  etc.

This module provides two approaches for loading LoRAs:
1. Patch-based (original): Creates patches for Forge's LoRA system
2. Direct merge (new): Directly modifies model weights, better for multiple LoRAs
"""

import torch
from typing import Optional
try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


class ZImageLoRAManager:
    """
    Manages direct LoRA merging for Z-Image models.

    This approach directly modifies model weights instead of using patches,
    which is more reliable for multiple LoRAs. It maintains a backup of
    original weights for restoration.

    Usage:
        manager = ZImageLoRAManager(model)
        manager.apply_loras([
            ('lora1.safetensors', 0.8),
            ('lora2.safetensors', 0.6),
        ])
        # Later, to switch LoRAs:
        manager.clear_loras()
        manager.apply_loras([...])
    """

    def __init__(self, model):
        """
        Initialize the LoRA manager.

        Args:
            model: The Z-Image model (ZImageTransformerWrapper or the underlying transformer)
        """
        self.model = model
        self.weight_backup = {}  # {key: original_weight_tensor}
        self.applied_loras = []  # [(filename, strength), ...]
        self._model_state_keys = None

    def _get_model_state_keys(self):
        """Get and cache model state dict keys."""
        if self._model_state_keys is None:
            self._model_state_keys = set(self.model.state_dict().keys())
        return self._model_state_keys

    def _find_model_key(self, lora_base_key: str) -> Optional[str]:
        """
        Find the actual model key for a LoRA key, trying multiple prefixes.

        Args:
            lora_base_key: The base key from the LoRA file

        Returns:
            The actual model key, or None if not found
        """
        model_keys = self._get_model_state_keys()
        model_key = lora_key_to_model_key(lora_base_key)

        # Try multiple prefixes to find the key in the model
        prefixes_to_try = [
            '',  # Direct match
            'diffusion_model.',  # Standard prefix
            'diffusion_model.transformer.',  # Wrapped transformer
            'transformer.',  # Just transformer prefix
        ]

        for prefix in prefixes_to_try:
            candidate = prefix + model_key
            if candidate in model_keys:
                return candidate

        return None

    def _get_parameter(self, key: str) -> torch.nn.Parameter:
        """Navigate to and return a parameter by its key path."""
        parts = key.split('.')
        target = self.model
        for part in parts[:-1]:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)
        return getattr(target, parts[-1])

    def _set_parameter(self, key: str, value: torch.Tensor):
        """Set a parameter by its key path."""
        parts = key.split('.')
        target = self.model
        for part in parts[:-1]:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)

        # Convert to parameter if needed
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value, requires_grad=False)
        setattr(target, parts[-1], value)

    def _backup_weight(self, key: str):
        """Backup a weight if not already backed up."""
        if key not in self.weight_backup:
            param = self._get_parameter(key)
            # Store a copy on CPU to save GPU memory
            self.weight_backup[key] = param.data.clone().cpu()

    def clear_loras(self):
        """
        Remove all applied LoRAs by restoring original weights.
        """
        if not self.weight_backup:
            print("[Z-Image LoRA] clear_loras called but no backup exists")
            return

        print(f"[Z-Image LoRA] Restoring {len(self.weight_backup)} original weights")

        # Debug: Sample a weight before and after restoration
        sample_key = next(iter(self.weight_backup.keys()))
        sample_backup = self.weight_backup[sample_key]
        try:
            param = self._get_parameter(sample_key)
            print(f"[Z-Image LoRA DEBUG] Before restore - {sample_key}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            print(f"[Z-Image LoRA DEBUG] Backup values - mean={sample_backup.mean().item():.6f}, std={sample_backup.std().item():.6f}")
        except Exception as e:
            print(f"[Z-Image LoRA DEBUG] Error sampling before restore: {e}")

        restored_count = 0
        for key, original_weight in self.weight_backup.items():
            try:
                param = self._get_parameter(key)
                device = param.device
                dtype = param.dtype
                # Restore with proper dtype conversion
                param.data.copy_(original_weight.to(device=device, dtype=dtype))
                restored_count += 1
            except Exception as e:
                print(f"[Z-Image LoRA] Warning: Failed to restore {key}: {e}")

        # Debug: Verify restoration
        try:
            param = self._get_parameter(sample_key)
            print(f"[Z-Image LoRA DEBUG] After restore - {sample_key}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        except Exception as e:
            print(f"[Z-Image LoRA DEBUG] Error sampling after restore: {e}")

        print(f"[Z-Image LoRA] Successfully restored {restored_count}/{len(self.weight_backup)} weights")

        self.weight_backup.clear()
        self.applied_loras.clear()
        self._model_state_keys = None  # Clear cache in case model structure changed

        print("[Z-Image LoRA] All LoRAs cleared, original weights restored")

    def apply_loras(self, lora_configs: list[tuple[str, float]], computation_dtype=torch.float32) -> dict:
        """
        Apply multiple LoRAs directly to model weights.

        This method first restores original weights (if any LoRAs were previously applied),
        then applies all specified LoRAs in order.

        Args:
            lora_configs: List of (lora_path, strength) tuples
            computation_dtype: Dtype for intermediate computations (default float32)

        Returns:
            Dict with statistics about the merge operation
        """
        # Clear any previously applied LoRAs first
        if self.applied_loras:
            self.clear_loras()

        total_applied = 0
        total_matched = 0
        total_unmatched = 0

        # Track first modified key for debugging
        first_modified_key = None
        first_modified_before = None

        # Track which keys are modified by each LoRA
        keys_modified_by_lora = {}
        overlapping_keys = set()

        for lora_idx, (lora_path, strength) in enumerate(lora_configs):
            print(f"[Z-Image LoRA] Direct merge: {lora_path} (strength={strength})")

            # Load LoRA state dict
            if load_file is None:
                raise ImportError("safetensors is required for LoRA loading")

            try:
                lora_state = load_file(lora_path)
            except Exception as e:
                print(f"[Z-Image LoRA] Failed to load {lora_path}: {e}")
                continue

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

            for lora_base_key, lora_weights in lora_groups.items():
                # Find the actual model key
                model_key = self._find_model_key(lora_base_key)

                if model_key is None:
                    unmatched_keys.append(lora_base_key)
                    continue

                matched += 1

                # Get LoRA components
                lora_up = lora_weights.get('up')
                lora_down = lora_weights.get('down')
                alpha = lora_weights.get('alpha')

                if lora_up is None or lora_down is None:
                    continue

                # Backup original weight before first modification
                self._backup_weight(model_key)

                # Get current parameter
                param = self._get_parameter(model_key)
                device = param.device
                dtype = param.dtype

                # Calculate scale: alpha / rank * strength
                rank = lora_down.shape[0]
                if alpha is not None:
                    alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                    scale = (alpha_val / rank) * strength
                else:
                    scale = strength

                # Move LoRA weights to computation device/dtype
                lora_up = lora_up.to(device=device, dtype=computation_dtype)
                lora_down = lora_down.to(device=device, dtype=computation_dtype)

                # Compute delta: up @ down * scale
                # Handle different tensor shapes (Linear vs Conv2d)
                if len(lora_up.shape) == 4:
                    # Conv2d: [out, rank, 1, 1] @ [rank, in, kh, kw]
                    delta = torch.mm(
                        lora_up.squeeze(3).squeeze(2),
                        lora_down.squeeze(3).squeeze(2)
                    ).unsqueeze(2).unsqueeze(3) * scale
                else:
                    # Linear: [out, rank] @ [rank, in]
                    delta = (lora_up @ lora_down) * scale

                # Apply delta to weight
                with torch.no_grad():
                    if param.shape == delta.shape:
                        # Track overlapping keys
                        if model_key in keys_modified_by_lora:
                            overlapping_keys.add(model_key)
                            # Log when same key is modified by second LoRA
                            if len(overlapping_keys) == 1:  # First overlap detected
                                current_val = (param.data.mean().item(), param.data.std().item())
                                print(f"[Z-Image LoRA DEBUG] OVERLAP: {model_key} modified by LoRA {lora_idx+1}")
                                print(f"[Z-Image LoRA DEBUG] Current (after LoRA1): mean={current_val[0]:.6f}, std={current_val[1]:.6f}")
                                print(f"[Z-Image LoRA DEBUG] Adding delta: mean={delta.mean().item():.6f}, std={delta.std().item():.6f}")
                        keys_modified_by_lora[model_key] = lora_idx

                        # Debug: Track first modified key across all LoRAs
                        if first_modified_key is None:
                            first_modified_key = model_key
                            first_modified_before = (param.data.mean().item(), param.data.std().item(),
                                                     param.data.min().item(), param.data.max().item())
                            print(f"[Z-Image LoRA DEBUG] First modified key: {model_key}")
                            print(f"[Z-Image LoRA DEBUG] Before: mean={first_modified_before[0]:.6f}, "
                                  f"std={first_modified_before[1]:.6f}, min={first_modified_before[2]:.6f}, "
                                  f"max={first_modified_before[3]:.6f}")

                        # Log delta statistics for first delta of each LoRA
                        if applied == 0:
                            print(f"[Z-Image LoRA DEBUG] Delta stats for {model_key}: "
                                  f"mean={delta.mean().item():.6f}, std={delta.std().item():.6f}, "
                                  f"min={delta.min().item():.6f}, max={delta.max().item():.6f}, "
                                  f"scale={scale:.6f}, rank={rank}")
                            if torch.isnan(delta).any() or torch.isinf(delta).any():
                                print(f"[Z-Image LoRA DEBUG] ERROR: Delta contains NaN or Inf!")

                        # IMPORTANT: Do the addition in float32 to avoid bfloat16 precision loss
                        # bfloat16 has only 7 bits of mantissa, so small deltas get lost
                        if applied == 0:
                            # Debug: Show specific value before/after for first weight
                            val_before = param.data[0, 0].item() if len(param.shape) >= 2 else param.data[0].item()
                            print(f"[Z-Image LoRA DEBUG] dtype={dtype}, param device={param.device}")
                            print(f"[Z-Image LoRA DEBUG] Single value before: {val_before:.10f}")

                        if dtype == torch.bfloat16 or dtype == torch.float16:
                            new_data = (param.data.float() + delta).to(dtype=dtype)
                            param.data.copy_(new_data)
                        else:
                            param.data.add_(delta.to(dtype=dtype))

                        if applied == 0:
                            val_after = param.data[0, 0].item() if len(param.shape) >= 2 else param.data[0].item()
                            delta_val = delta[0, 0].item() if len(delta.shape) >= 2 else delta[0].item()
                            print(f"[Z-Image LoRA DEBUG] Delta value: {delta_val:.10f}")
                            print(f"[Z-Image LoRA DEBUG] Single value after: {val_after:.10f}")
                            print(f"[Z-Image LoRA DEBUG] Actual change: {val_after - val_before:.10f}")

                        applied += 1
                    else:
                        print(f"[Z-Image LoRA] Shape mismatch for {model_key}: "
                              f"param={param.shape}, delta={delta.shape}")

            if unmatched_keys:
                print(f"[Z-Image LoRA] {len(unmatched_keys)} unmatched keys in {lora_path}")
                if len(unmatched_keys) <= 5:
                    print(f"[Z-Image LoRA] Unmatched: {unmatched_keys}")

            print(f"[Z-Image LoRA] Applied {applied}/{matched} layers from {lora_path}")

            self.applied_loras.append((lora_path, strength))
            total_applied += applied
            total_matched += matched
            total_unmatched += len(unmatched_keys)

        # Debug: Check the first modified weight after all modifications
        if first_modified_key:
            try:
                param = self._get_parameter(first_modified_key)
                after = (param.data.mean().item(), param.data.std().item(),
                         param.data.min().item(), param.data.max().item())
                print(f"[Z-Image LoRA DEBUG] After all LoRAs - {first_modified_key}:")
                print(f"[Z-Image LoRA DEBUG] After: mean={after[0]:.6f}, std={after[1]:.6f}, "
                      f"min={after[2]:.6f}, max={after[3]:.6f}")

                # Show the change
                if first_modified_before:
                    mean_diff = after[0] - first_modified_before[0]
                    std_diff = after[1] - first_modified_before[1]
                    print(f"[Z-Image LoRA DEBUG] Change: mean_diff={mean_diff:.6f}, std_diff={std_diff:.6f}")

                # Check if values are reasonable
                if abs(after[0]) > 100 or after[1] > 100:
                    print(f"[Z-Image LoRA DEBUG] WARNING: Weight values seem abnormally large!")
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    print(f"[Z-Image LoRA DEBUG] ERROR: Weight contains NaN or Inf values!")
            except Exception as e:
                print(f"[Z-Image LoRA DEBUG] Error checking weight: {e}")

        # Debug: Check backup status and overlaps
        print(f"[Z-Image LoRA DEBUG] Backup contains {len(self.weight_backup)} weights")
        print(f"[Z-Image LoRA DEBUG] Total unique keys modified: {len(keys_modified_by_lora)}")
        print(f"[Z-Image LoRA DEBUG] Overlapping keys (modified by multiple LoRAs): {len(overlapping_keys)}")

        return {
            'applied': total_applied,
            'matched': total_matched,
            'unmatched': total_unmatched,
            'loras_loaded': len(self.applied_loras),
        }

    def get_applied_loras(self) -> list[tuple[str, float]]:
        """Return list of currently applied LoRAs."""
        return self.applied_loras.copy()


# Global manager instance per model (weak reference would be better but this is simpler)
_lora_managers = {}


def get_lora_manager(model) -> ZImageLoRAManager:
    """
    Get or create a LoRA manager for a model.

    Args:
        model: The Z-Image model

    Returns:
        ZImageLoRAManager instance for this model
    """
    model_id = id(model)
    if model_id not in _lora_managers:
        _lora_managers[model_id] = ZImageLoRAManager(model)
    return _lora_managers[model_id]


def verify_lora_weights(model, sample_key: str = None) -> dict:
    """
    Verify that LoRA modifications are still present on the model.
    Call this during inference to ensure weights haven't been reset.

    Args:
        model: The model to verify
        sample_key: Specific key to check (optional)

    Returns:
        Dict with verification results
    """
    model_id = id(model)
    if model_id not in _lora_managers:
        return {'status': 'no_manager', 'message': 'No LoRA manager found for this model'}

    manager = _lora_managers[model_id]
    if not manager.weight_backup:
        return {'status': 'no_backup', 'message': 'No weight backup exists (no LoRAs applied?)'}

    results = {
        'status': 'ok',
        'loras_applied': len(manager.applied_loras),
        'weights_backed_up': len(manager.weight_backup),
        'sample_checks': []
    }

    # Check a few sample weights
    keys_to_check = list(manager.weight_backup.keys())[:3]
    if sample_key and sample_key in manager.weight_backup:
        keys_to_check = [sample_key] + [k for k in keys_to_check if k != sample_key][:2]

    for key in keys_to_check:
        try:
            param = manager._get_parameter(key)
            backup = manager.weight_backup[key]

            # Move backup to same device for comparison
            backup_on_device = backup.to(device=param.device)

            current_mean = param.data.mean().item()
            backup_mean = backup.mean().item()

            # Check individual values (first few elements)
            current_flat = param.data.flatten()[:5]
            backup_flat = backup_on_device.flatten()[:5]

            # Calculate actual difference
            diff = (param.data.float() - backup_on_device.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Check if current weights differ from backup (meaning LoRAs are applied)
            is_modified = max_diff > 1e-7

            results['sample_checks'].append({
                'key': key,
                'current_mean': current_mean,
                'backup_mean': backup_mean,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'current_sample': [v.item() for v in current_flat],
                'backup_sample': [v.item() for v in backup_flat],
                'is_modified': is_modified,
                'device': str(param.device)
            })
        except Exception as e:
            results['sample_checks'].append({
                'key': key,
                'error': str(e)
            })

    return results


def clear_lora_manager(model):
    """Remove the LoRA manager for a model (call when model is unloaded)."""
    model_id = id(model)
    if model_id in _lora_managers:
        _lora_managers[model_id].clear_loras()
        del _lora_managers[model_id]


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
