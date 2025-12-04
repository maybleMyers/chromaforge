import os
import torch

from huggingface_guess.detection import unet_config_from_diffusers_unet, model_config_from_unet
from huggingface_guess.utils import unet_to_diffusers
from backend import memory_management
from backend.operations import using_forge_operations
from backend.nn.cnets import cldm
from backend.patcher.controlnet import ControlLora, ControlNet, load_t2i_adapter, apply_controlnet_advanced
from modules_forge.shared import add_supported_control_model


class ControlModelPatcher:
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        return None

    def __init__(self, model_patcher=None):
        self.model_patcher = model_patcher
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0
        self.positive_advanced_weighting = None
        self.negative_advanced_weighting = None
        self.advanced_frame_weighting = None
        self.advanced_sigma_weighting = None
        self.advanced_mask_weighting = None

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        return

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        return


class ControlNetPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(controlnet_data, ckpt_path):
        if "lora_controlnet" in controlnet_data:
            return ControlNetPatcher(ControlLora(controlnet_data))

        controlnet_config = None
        if "controlnet_cond_embedding.conv_in.weight" in controlnet_data:  # diffusers format
            unet_dtype = memory_management.unet_dtype()
            controlnet_config = unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
            diffusers_keys = unet_to_diffusers(controlnet_config)
            diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
            diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

            count = 0
            loop = True
            while loop:
                suffix = [".weight", ".bias"]
                for s in suffix:
                    k_in = "controlnet_down_blocks.{}{}".format(count, s)
                    k_out = "zero_convs.{}.0{}".format(count, s)
                    if k_in not in controlnet_data:
                        loop = False
                        break
                    diffusers_keys[k_in] = k_out
                count += 1

            count = 0
            loop = True
            while loop:
                suffix = [".weight", ".bias"]
                for s in suffix:
                    if count == 0:
                        k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                    else:
                        k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                    k_out = "input_hint_block.{}{}".format(count * 2, s)
                    if k_in not in controlnet_data:
                        k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                        loop = False
                    diffusers_keys[k_in] = k_out
                count += 1

            new_sd = {}
            for k in diffusers_keys:
                if k in controlnet_data:
                    new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

            leftover_keys = controlnet_data.keys()
            if len(leftover_keys) > 0:
                print("leftover keys:", leftover_keys)
            controlnet_data = new_sd

        pth_key = 'control_model.zero_convs.0.0.weight'
        pth = False
        key = 'zero_convs.0.0.weight'
        if pth_key in controlnet_data:
            pth = True
            key = pth_key
            prefix = "control_model."
        elif key in controlnet_data:
            prefix = ""
        else:
            net = load_t2i_adapter(controlnet_data)
            if net is None:
                return None
            return ControlNetPatcher(net)

        if controlnet_config is None:
            unet_dtype = memory_management.unet_dtype()
            controlnet_config = model_config_from_unet(controlnet_data, prefix, True).unet_config
            controlnet_config['dtype'] = unet_dtype

        load_device = memory_management.get_torch_device()
        computation_dtype = memory_management.get_computation_dtype(load_device)

        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]

        with using_forge_operations(dtype=unet_dtype, manual_cast_enabled=computation_dtype != unet_dtype):
            control_model = cldm.ControlNet(**controlnet_config).to(dtype=unet_dtype)

        if pth:
            if 'difference' in controlnet_data:
                print("WARNING: Your controlnet model is diff version rather than official float16 model. "
                      "Please use an official float16/float32 model for robust performance.")

            class WeightsLoader(torch.nn.Module):
                pass

            w = WeightsLoader()
            w.control_model = control_model
            missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
        else:
            missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
        print(missing, unexpected)

        global_average_pooling = False
        filename = os.path.splitext(ckpt_path)[0]
        if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"):
            # TODO: smarter way of enabling global_average_pooling
            global_average_pooling = True

        control = ControlNet(control_model, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=computation_dtype)
        return ControlNetPatcher(control)

    def __init__(self, model_patcher):
        super().__init__(model_patcher)

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = apply_controlnet_advanced(
            unet=unet,
            controlnet=self.model_patcher,
            image_bchw=cond,
            strength=self.strength,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            positive_advanced_weighting=self.positive_advanced_weighting,
            negative_advanced_weighting=self.negative_advanced_weighting,
            advanced_frame_weighting=self.advanced_frame_weighting,
            advanced_sigma_weighting=self.advanced_sigma_weighting,
            advanced_mask_weighting=self.advanced_mask_weighting
        )

        process.sd_model.forge_objects.unet = unet
        return


class ZImageControlNetPatcher(ControlModelPatcher):
    """
    ControlNet patcher for Z-Image models (Z-Image-Turbo-Fun-Controlnet-Union).

    This ControlNet works differently from standard SD ControlNets:
    - Control components are added directly to the transformer
    - Uses VAE-encoded control images as context
    - Injects hints at 15 transformer layers (every 2nd layer)
    """

    # Control weights are approximately 3GB
    CONTROL_WEIGHTS_SIZE_BYTES = 3 * 1024 * 1024 * 1024  # 3GB

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        """Detect Z-Image ControlNet by checking for control-specific keys"""
        # Z-Image ControlNet has specific keys for control layers
        control_keys = [k for k in state_dict.keys() if any(
            pattern in k for pattern in [
                'control_layers.',
                'control_all_x_embedder.',
                'control_noise_refiner.'
            ]
        )]

        if not control_keys:
            return None

        print(f"Detected Z-Image ControlNet with {len(control_keys)} control keys")
        return ZImageControlNetPatcher(state_dict, ckpt_path)

    def __init__(self, state_dict, ckpt_path):
        super().__init__(model_patcher=None)
        self.control_state_dict = state_dict
        self.ckpt_path = ckpt_path
        self._control_loaded = False

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        """
        Called before sampling starts.
        - Encodes control image to VAE latents
        - Loads control layers into transformer if not already loaded
        - Sets transformer_options for control
        """
        sd_model = process.sd_model

        # Check if this is a Z-Image model
        if not self._is_zimage_model(sd_model):
            print("WARNING: Z-Image ControlNet requires Z-Image model, skipping control")
            return

        # Get the UNet patcher
        unet = sd_model.forge_objects.unet.clone()

        # 1. Encode control image to VAE latents
        control_latents = self._encode_control_image(cond, sd_model)

        # 2. Load control layers into transformer if not already loaded
        self._ensure_control_layers_loaded(sd_model, unet)

        # 3. Set transformer options for control
        unet.set_transformer_option('control_context', control_latents)
        unet.set_transformer_option('control_context_scale', self.strength)
        unet.set_transformer_option('z_image_controlnet_active', True)

        # 4. Reserve memory for control components (3GB)
        unet.add_extra_preserved_memory_during_sampling(self.CONTROL_WEIGHTS_SIZE_BYTES)

        # Update the UNet
        process.sd_model.forge_objects.unet = unet

        print(f"Z-Image ControlNet applied: scale={self.strength:.2f}, "
              f"control_latents shape={control_latents.shape}")

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        """Clean up control state after sampling"""
        unet = process.sd_model.forge_objects.unet

        # Clear transformer options
        if hasattr(unet, 'model_options') and 'transformer_options' in unet.model_options:
            to = unet.model_options['transformer_options']
            to.pop('control_context', None)
            to.pop('control_context_scale', None)
            to.pop('z_image_controlnet_active', None)

    def _is_zimage_model(self, sd_model):
        """Check if the current model is a Z-Image model"""
        try:
            unet = sd_model.forge_objects.unet
            config = unet.model.config if hasattr(unet.model, 'config') else None
            if config is not None:
                return getattr(config, 'is_zimage', False)

            # Also check by class name
            model_class = type(sd_model).__name__
            return 'ZImage' in model_class
        except Exception:
            return False

    def _encode_control_image(self, control_image, sd_model):
        """
        Encode control image to VAE latents matching Z-Image format.

        Args:
            control_image: [B, C, H, W] tensor, values in [0, 1] or [0, 255]
            sd_model: The stable diffusion model

        Returns:
            Control latents [B, C, 1, H, W] for Z-Image in the model's dtype
        """
        vae = sd_model.forge_objects.vae

        # Ensure control_image is a tensor
        if not isinstance(control_image, torch.Tensor):
            control_image = torch.tensor(control_image)

        # Normalize to [0, 1] if needed
        if control_image.max() > 1.0:
            control_image = control_image / 255.0

        # Get target device and dtype from the transformer (not VAE, as transformer may be bfloat16)
        try:
            unet = sd_model.forge_objects.unet
            transformer = unet.model.diffusion_model.transformer
            target_dtype = next(transformer.parameters()).dtype
            target_device = next(transformer.parameters()).device
        except Exception:
            # Fallback to VAE device/dtype
            target_device = next(vae.first_stage_model.parameters()).device
            target_dtype = next(vae.first_stage_model.parameters()).dtype

        # Move to VAE device and dtype for encoding
        vae_device = next(vae.first_stage_model.parameters()).device
        vae_dtype = next(vae.first_stage_model.parameters()).dtype
        control_image = control_image.to(device=vae_device, dtype=vae_dtype)

        # VAE expects [B, H, W, C] format
        if control_image.dim() == 4 and control_image.shape[1] in [1, 3, 4]:
            # Input is [B, C, H, W], convert to [B, H, W, C]
            control_image_vae = control_image.permute(0, 2, 3, 1)
        else:
            control_image_vae = control_image

        # Encode through VAE
        with torch.no_grad():
            latents = vae.encode(control_image_vae)

        # Apply Z-Image VAE scaling factors
        # scale_factor = 0.3611, shift_factor = 0.1159
        # Formula: (latents - shift) * scale
        latents = (latents - 0.1159) * 0.3611

        # Add frame dimension: [B, C, H, W] -> [B, C, 1, H, W]
        latents = latents.unsqueeze(2)

        # Cast to transformer dtype (e.g., bfloat16) for control processing
        latents = latents.to(dtype=target_dtype)

        return latents

    def _ensure_control_layers_loaded(self, sd_model, unet):
        """Load control weights into transformer if not already loaded"""
        try:
            # Get the wrapped transformer
            wrapped_model = unet.model.diffusion_model
            transformer = wrapped_model.transformer if hasattr(wrapped_model, 'transformer') else wrapped_model

            # Always check if the CURRENT transformer has control layers loaded
            # (model may have been reloaded, creating a new transformer without control layers)
            if hasattr(transformer, '_control_layers_loaded') and transformer._control_layers_loaded:
                self._control_loaded = True
                return

            # Import control module
            from backend.nn.zimage_control import add_zimage_control_components

            # Get device and dtype from transformer
            device = next(transformer.parameters()).device
            dtype = next(transformer.parameters()).dtype

            # Add control components and load weights
            print(f"Loading Z-Image ControlNet components ({self.CONTROL_WEIGHTS_SIZE_BYTES / (1024**3):.1f}GB)...")
            add_zimage_control_components(transformer, self.control_state_dict, device=device, dtype=dtype)

            self._control_loaded = True
            print("Z-Image ControlNet components loaded successfully")

        except Exception as e:
            print(f"ERROR loading Z-Image ControlNet: {e}")
            import traceback
            traceback.print_exc()


# Register Z-Image ControlNet patcher first (more specific detection)
add_supported_control_model(ZImageControlNetPatcher)
# Register standard ControlNet patcher second
add_supported_control_model(ControlNetPatcher)
