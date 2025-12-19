import torch

from backend import memory_management, attention
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler


# Import verification function for LoRA debugging
try:
    from backend.lora.zimage_lora import verify_lora_weights
except ImportError:
    verify_lora_weights = None


def _get_torch_compile_settings():
    """Get torch.compile settings from shared options."""
    try:
        from modules.shared import opts
        compile_mode = getattr(opts, 'torch_compile_mode', 'Disabled')
        return compile_mode if compile_mode != 'Disabled' else None
    except Exception:
        return None


def _apply_torch_compile(model, mode, model_name="model"):
    """Apply torch.compile to a model with error handling."""
    if mode is None:
        return model

    try:
        import torch._dynamo
        import sys

        # Check PyTorch version supports torch.compile
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version < (2, 0):
            print(f'[torch.compile] PyTorch {torch.__version__} does not support torch.compile (requires 2.0+)')
            return model

        # Determine backend based on platform
        backend = "inductor"  # Default backend

        # On Windows, inductor requires Triton which has compatibility issues
        # Use cudagraphs backend instead which works natively
        if sys.platform == "win32":
            # Check if triton is properly available
            try:
                from triton.compiler.compiler import triton_key
            except ImportError:
                print(f'[torch.compile] Triton not fully available on Windows, using cudagraphs backend')
                backend = "cudagraphs"

        print(f'[torch.compile] Compiling {model_name} with backend="{backend}"...')

        # Build compile kwargs based on backend
        compile_kwargs = {
            'fullgraph': False,  # Allow graph breaks for dynamic control flow
            'backend': backend,
        }

        # 'mode' is only supported by inductor backend
        if backend == "inductor":
            compile_kwargs['mode'] = mode

        # Use dynamic=True for PyTorch 2.3+ for better variable shape handling
        if torch_version >= (2, 3):
            compile_kwargs['dynamic'] = True

        compiled_model = torch.compile(model, **compile_kwargs)
        print(f'[torch.compile] {model_name} compilation scheduled (will compile on first forward pass)')
        return compiled_model

    except Exception as e:
        print(f'[torch.compile] Failed to compile {model_name}: {e}')
        return model


class KModel(torch.nn.Module):
    def __init__(self, model, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        print(f'K-Model Created: {dict(storage_dtype=self.storage_dtype, computation_dtype=self.computation_dtype)}')

        # Check if torch.compile is enabled
        compile_mode = _get_torch_compile_settings()
        if compile_mode:
            self.diffusion_model = _apply_torch_compile(model, compile_mode, "diffusion_model")
            self._is_compiled = True
        else:
            self.diffusion_model = model
            self._is_compiled = False

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t

        # DEBUG: Print sigma info on first call
        if not hasattr(self, '_debug_sigma_printed'):
            self._debug_sigma_printed = True
            print(f"\n=== KModel Debug (first call) ===")
            print(f"Sigma (t) value: {sigma}")
            print(f"Predictor type: {type(self.predictor).__name__}")
            print(f"Predictor sigmas range: [{self.predictor.sigma_min:.6f}, {self.predictor.sigma_max:.6f}]")

            # Verify LoRA weights are still applied
            if verify_lora_weights is not None:
                lora_check = verify_lora_weights(self)
                print(f"\n=== LoRA Verification at Inference ===")
                print(f"Status: {lora_check.get('status')}")
                if lora_check.get('status') == 'ok':
                    print(f"LoRAs applied: {lora_check.get('loras_applied')}")
                    print(f"Weights backed up: {lora_check.get('weights_backed_up')}")
                    for check in lora_check.get('sample_checks', []):
                        if 'error' in check:
                            print(f"  {check['key']}: ERROR - {check['error']}")
                        else:
                            status = "MODIFIED" if check['is_modified'] else "UNCHANGED (weights reset!)"
                            print(f"  {check['key'][:60]}...")
                            print(f"    max_diff={check['max_diff']:.8f}, mean_diff={check['mean_diff']:.8f} [{status}]")
                            print(f"    device={check['device']}")
                            print(f"    current[0:5]: {[f'{v:.6f}' for v in check['current_sample']]}")
                            print(f"    backup[0:5]:  {[f'{v:.6f}' for v in check['backup_sample']]}")
                else:
                    print(f"Message: {lora_check.get('message')}")
                print(f"======================================\n")
            print(f"\n=== KModel c_crossattn Debug ===")
            print(f"c_crossattn type: {type(c_crossattn)}")
            if isinstance(c_crossattn, dict):
                print(f"c_crossattn keys: {c_crossattn.keys()}")
                if 'attention_mask' in c_crossattn:
                    print(f"attention_mask shape: {c_crossattn['attention_mask'].shape}")
            elif hasattr(c_crossattn, 'shape'):
                print(f"c_crossattn shape: {c_crossattn.shape}")

        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        # Handle context which may now be a dict with crossattn and attention_mask
        if isinstance(c_crossattn, dict):
            context = c_crossattn['crossattn']
            attention_mask = c_crossattn.get('attention_mask', None)
        else:
            # Backward compatibility: if context is just a tensor
            context = c_crossattn
            attention_mask = None

        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)

        # Keep attention mask as boolean if present
        if attention_mask is not None:
            # Store attention mask in transformer options for potential future use
            transformer_options = transformer_options.copy() if transformer_options else {}
            transformer_options['attention_mask'] = attention_mask

        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        denoised = self.predictor.calculate_denoised(sigma, model_output, x)

        # DEBUG: Check denoised output on first call
        if not hasattr(self, '_debug_denoised_printed'):
            self._debug_denoised_printed = True
            print(f"\n=== KModel Denoised Debug ===")
            print(f"model_output stats: min={model_output.min().item():.4f}, max={model_output.max().item():.4f}")
            print(f"model_output has NaN: {torch.isnan(model_output).any().item()}")
            print(f"denoised stats: min={denoised.min().item():.4f}, max={denoised.max().item():.4f}")
            print(f"denoised has NaN: {torch.isnan(denoised).any().item()}")
            print(f"==============================\n")

        return denoised

    def memory_required(self, input_shape):
        area = input_shape[0] * input_shape[2] * input_shape[3]
        dtype_size = memory_management.dtype_size(self.computation_dtype)

        if attention.attention_function in [attention.attention_pytorch, attention.attention_xformers]:
            scaler = 1.28
        else:
            scaler = 1.65
            if attention.get_attn_precision() == torch.float32:
                dtype_size = 4

        is_chromadct = hasattr(self.diffusion_model, 'nerf_blocks')
        if is_chromadct:
            patch_size = getattr(self.diffusion_model, 'patch_size', 16)
            h_patches = input_shape[2] // patch_size
            w_patches = input_shape[3] // patch_size
            seq_len = h_patches * w_patches + 512
            scaler = 1.1
            nerf_overhead = seq_len * 49152 * dtype_size * input_shape[0]
            return scaler * area * dtype_size * 16384 + nerf_overhead

        return scaler * area * dtype_size * 16384
