import torch

from backend import memory_management, attention
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler


class KModel(torch.nn.Module):
    def __init__(self, model, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        print(f'K-Model Created: {dict(storage_dtype=self.storage_dtype, computation_dtype=self.computation_dtype)}')

        self.diffusion_model = model

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape):
        area = input_shape[0] * input_shape[2] * input_shape[3]
        dtype_size = memory_management.dtype_size(self.computation_dtype)

        if attention.attention_function in [attention.attention_pytorch, attention.attention_xformers]:
            scaler = 1.28
        else:
            scaler = 1.65
            if attention.get_attn_precision() == torch.float32:
                dtype_size = 4

        # Check if this is a ChromaDCT model and apply optimized memory calculation
        try:
            # ChromaDCT models have distinctive components - check for img_in_patch
            if hasattr(self.diffusion_model, 'img_in_patch'):
                # ChromaDCT models process RGB (3 channels) directly vs latent space (16 channels)
                # They also use patch-based processing which is more memory efficient
                # Reduce multiplier from 16384 to 2048 (8x reduction)
                
                # Only print detection message once per model instance
                if not hasattr(self, '_chromadct_detected'):
                    print("Detected ChromaDCT model - using optimized memory estimation (2048 vs 16384)")
                    self._chromadct_detected = True
                
                return scaler * area * dtype_size * 2048
        except:
            pass

        return scaler * area * dtype_size * 16384

    def enable_block_swap(self, num_blocks: int, device):
        """Delegate block swapping to underlying diffusion model."""
        if hasattr(self.diffusion_model, 'enable_block_swap'):
            print(f"[KModel] Delegating enable_block_swap({num_blocks}) to {self.diffusion_model.__class__.__name__}")
            return self.diffusion_model.enable_block_swap(num_blocks, device)
        else:
            raise AttributeError(f"Underlying model {self.diffusion_model.__class__.__name__} does not support block swapping")

    def prepare_block_swap_before_forward(self):
        """Delegate block swap preparation to underlying diffusion model."""
        if hasattr(self.diffusion_model, 'prepare_block_swap_before_forward'):
            return self.diffusion_model.prepare_block_swap_before_forward()

    @property
    def blocks_to_swap(self):
        """Access blocks_to_swap from underlying diffusion model."""
        if hasattr(self.diffusion_model, 'blocks_to_swap'):
            return self.diffusion_model.blocks_to_swap
        return None

    @property
    def double_blocks(self):
        """Access double_blocks from underlying diffusion model."""
        if hasattr(self.diffusion_model, 'double_blocks'):
            return self.diffusion_model.double_blocks
        return None

    @property
    def single_blocks(self):
        """Access single_blocks from underlying diffusion model."""
        if hasattr(self.diffusion_model, 'single_blocks'):
            return self.diffusion_model.single_blocks
        return None
