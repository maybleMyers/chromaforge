import torch

from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management


class ZImageOmniLatentFormat:
    """Latent format for Z-Image-Omni models (16-channel latents using FLUX VAE)"""
    # Computed via least squares regression from latent/RGB pairs
    # Maps 16-channel latents to RGB for cheap preview approximation
    latent_rgb_factors = [
        [-0.037223, -0.005345,  0.027556],
        [ 0.016390,  0.037071,  0.064793],
        [ 0.019696, -0.027534, -0.014296],
        [-0.006327,  0.014695,  0.036488],
        [ 0.068934,  0.055449,  0.033998],
        [ 0.013588,  0.014694,  0.000826],
        [ 0.074462,  0.109467,  0.114805],
        [-0.024411, -0.022535, -0.025943],
        [-0.023577,  0.013515,  0.077390],
        [ 0.062267,  0.037371, -0.032887],
        [-0.052327, -0.016201, -0.019524],
        [ 0.068548,  0.035523,  0.014399],
        [ 0.014019,  0.010056,  0.018069],
        [-0.063674, -0.015887, -0.049793],
        [-0.005128, -0.034811, -0.009309],
        [-0.064089, -0.038018, -0.024993],
    ]

    def __init__(self):
        self.scale_factor = 0.3611  # Z-Image VAE scale factor
        self.shift_factor = 0.1159  # Z-Image VAE shift factor

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class ZImageOmni(ForgeDiffusionEngine):
    def __init__(self, components_dict, estimated_config=None):
        # Create minimal config if not provided
        if estimated_config is None:
            class MinimalConfig:
                def inpaint_model(self):
                    return False
            estimated_config = MinimalConfig()

        super().__init__(estimated_config, components_dict)
        self.is_inpaint = False

        # Add latent_format to model_config for cheap preview approximation
        if not hasattr(self.model_config, 'latent_format'):
            self.model_config.latent_format = ZImageOmniLatentFormat()

        # Wrap Qwen encoder in CLIP interface
        clip = CLIP(
            model_dict={
                'qwen': components_dict['text_encoder']
            },
            tokenizer_dict={
                'qwen': components_dict['tokenizer']
            }
        )

        vae = VAE(model=components_dict['vae'])

        # Ensure VAE latent channels match Transformer input channels
        transformer_config = components_dict['transformer'].config
        transformer_channels = None

        if isinstance(transformer_config, dict):
            transformer_channels = transformer_config.get('in_channels')
        elif hasattr(transformer_config, 'in_channels'):
            transformer_channels = transformer_config.in_channels

        if transformer_channels is not None:
            if vae.latent_channels != transformer_channels:
                print(f"Correction: VAE latent_channels ({vae.latent_channels}) != Transformer in_channels ({transformer_channels}). Updating VAE to match Transformer.")
                vae.latent_channels = transformer_channels
                if hasattr(vae.first_stage_model.config, 'latent_channels'):
                    vae.first_stage_model.config.latent_channels = transformer_channels

        # Set Z-Image specific VAE scaling factors
        vae.first_stage_model.scaling_factor = 0.3611
        vae.first_stage_model.shift_factor = 0.1159
        if hasattr(vae.first_stage_model.config, 'scaling_factor'):
            vae.first_stage_model.config.scaling_factor = 0.3611
        if hasattr(vae.first_stage_model.config, 'shift_factor'):
            vae.first_stage_model.config.shift_factor = 0.1159

        # Store SigLIP components if available
        self.siglip = components_dict.get('siglip')
        self.siglip_processor = components_dict.get('siglip_processor')

        # Wrap the transformer to adapt parameter names
        class ZImageOmniTransformerWrapper(torch.nn.Module):
            def __init__(wrapper_self, transformer, siglip=None, siglip_processor=None):
                super().__init__()
                wrapper_self.transformer = transformer
                wrapper_self.siglip = siglip
                wrapper_self.siglip_processor = siglip_processor

            def forward(wrapper_self, x, timestep, context=None, transformer_options=None, **kwargs):
                import torch

                # Get condition images from transformer_options
                condition_images = transformer_options.get('condition_images') if transformer_options else None
                condition_latents = transformer_options.get('condition_latents') if transformer_options else None
                siglip_embeds = transformer_options.get('siglip_embeds') if transformer_options else None

                if not isinstance(x, list):
                    # Input should be [batch, channels, height, width] (4D)
                    # Need to add frame dimension and split into list
                    if len(x.shape) == 4:  # [B, C, H, W]
                        x = x.unsqueeze(2)  # [B, C, 1, H, W] - add frame dimension
                        x = list(x.unbind(dim=0))  # List of [C, 1, H, W]
                    else:
                        raise ValueError(f"Unexpected input shape: {x.shape}. Expected 4D tensor [B, C, H, W]")

                bsz = len(x)

                # Convert context to list of lists format for Omni model
                if context is not None and not isinstance(context, list):
                    attention_mask = None
                    if transformer_options is not None and 'attention_mask' in transformer_options:
                        attention_mask = transformer_options['attention_mask']

                    if len(context.shape) == 3:  # [batch, seq_len, features]
                        if attention_mask is not None:
                            # Filter by attention mask
                            context_list = []
                            for i in range(context.shape[0]):
                                filtered = context[i][attention_mask[i]]
                                # Omni model expects list of lists: [[cap_feat1, cap_feat2, ...], ...]
                                context_list.append([filtered])
                            context = context_list
                        else:
                            context = [[context[i]] for i in range(context.shape[0])]
                    elif len(context.shape) == 2:  # [seq_len, features] - single item
                        context = [[context]]
                    else:
                        raise ValueError(f"Unexpected context shape: {context.shape}")

                # Invert timestep: Forge uses sigma in [1->0] (noisy->clean)
                # but Z-Image expects t in [0->1] (noisy->clean)
                timestep = 1.0 - timestep
                timestep = torch.clamp(timestep, min=1e-6, max=1.0)

                # Ensure pad tokens are on the same device as input
                target_device = x[0].device
                if hasattr(wrapper_self.transformer, 'x_pad_token') and wrapper_self.transformer.x_pad_token.device != target_device:
                    wrapper_self.transformer.x_pad_token.data = wrapper_self.transformer.x_pad_token.data.to(target_device)
                if hasattr(wrapper_self.transformer, 'cap_pad_token') and wrapper_self.transformer.cap_pad_token.device != target_device:
                    wrapper_self.transformer.cap_pad_token.data = wrapper_self.transformer.cap_pad_token.data.to(target_device)
                if hasattr(wrapper_self.transformer, 'siglip_pad_token') and wrapper_self.transformer.siglip_pad_token.device != target_device:
                    wrapper_self.transformer.siglip_pad_token.data = wrapper_self.transformer.siglip_pad_token.data.to(target_device)

                # Prepare condition latents (empty list if not provided)
                if condition_latents is None:
                    condition_latents = [[] for _ in range(bsz)]

                # Prepare siglip embeds (None list if not provided)
                if siglip_embeds is None:
                    siglip_embeds = [None for _ in range(bsz)]

                # Call transformer with Omni model signature
                result = wrapper_self.transformer(
                    x=x,
                    t=timestep,
                    cap_feats=context,
                    cond_latents=condition_latents,
                    siglip_feats=siglip_embeds,
                    patch_size=2,
                    f_patch_size=1,
                    return_dict=False
                )
                output_list = result[0] if isinstance(result, tuple) else result

                # Handle Transformer2DModelOutput from diffusers
                if hasattr(output_list, 'sample'):
                    output_list = output_list.sample

                # Convert list of [C, F, H, W] tensors back to batched tensor
                if isinstance(output_list, list):
                    output = torch.stack(output_list, dim=0).squeeze(2)
                    return -output
                else:
                    return -output_list

            def __getattr__(wrapper_self, name):
                # Pass through all other attributes to the underlying transformer
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(wrapper_self.transformer, name)

        wrapped_transformer = ZImageOmniTransformerWrapper(
            components_dict['transformer'],
            siglip=self.siglip,
            siglip_processor=self.siglip_processor
        )

        # Z-Image uses dynamic resolution-dependent shift (like Flux)
        k_predictor = PredictionFlux(
            seq_len=4096,
            base_seq_len=256,
            max_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
        )

        # Create config object for Z-Image-Omni identification
        class ZImageOmniModelConfig:
            is_zimage = True
            is_zimage_omni = True
            huggingface_repo = 'Z-Image-Omni'

        unet = UnetPatcher.from_model(
            model=wrapped_transformer,
            diffusers_scheduler=components_dict['scheduler'],
            k_predictor=k_predictor,
            config=ZImageOmniModelConfig()
        )

        self.text_processing_engine_qwen = QwenTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen,
            tokenizer=clip.tokenizer.qwen,
            emphasis_name=dynamic_args['emphasis_name'],
            min_length=1
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def set_clip_skip(self, clip_skip):
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        # Process prompts through Qwen text encoder
        tokenizer = self.text_processing_engine_qwen.tokenizer
        text_encoder = self.text_processing_engine_qwen.text_encoder

        formatted_prompts = []
        for prompt_item in prompt:
            messages = [{"role": "user", "content": prompt_item}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted_prompts.append(formatted_prompt)

        # Tokenize
        text_inputs = tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        device = text_encoder.embed_tokens.weight.device
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Encode (use hidden_states[-2] like official implementation)
        outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-2]

        return {'crossattn': prompt_embeds, 'attention_mask': prompt_masks}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_qwen.tokenize([prompt])[0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def prepare_condition_images(self, images, device, dtype):
        """
        Prepare condition images for the Omni model.
        Encodes images to VAE latents and extracts SigLIP features.

        Args:
            images: List of PIL images or tensors
            device: Target device
            dtype: Target dtype

        Returns:
            Tuple of (condition_latents, siglip_embeds)
        """
        if images is None or len(images) == 0:
            return None, None

        condition_latents = []
        siglip_embeds = []

        for image in images:
            # Encode to VAE latent
            if isinstance(image, torch.Tensor):
                image_tensor = image
            else:
                # Convert PIL to tensor
                import numpy as np
                from PIL import Image
                if isinstance(image, Image.Image):
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")

            image_tensor = image_tensor.to(device=device, dtype=dtype)

            # Encode with VAE
            vae = self.forge_objects.vae.first_stage_model
            latent = vae.encode(image_tensor).latent_dist.mode()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
            latent = latent.unsqueeze(1)  # Add frame dimension
            condition_latents.append(latent)

            # Extract SigLIP features if available
            if self.siglip is not None and self.siglip_processor is not None:
                siglip_inputs = self.siglip_processor(images=[image], return_tensors="pt").to(device)
                shape = siglip_inputs.spatial_shapes[0]
                hidden_state = self.siglip(**siglip_inputs).last_hidden_state
                B, N, C = hidden_state.shape
                hidden_state = hidden_state[:, :shape[0] * shape[1]]
                hidden_state = hidden_state.view(shape[0], shape[1], C)
                siglip_embeds.append(hidden_state.to(dtype))
            else:
                siglip_embeds.append(None)

        return condition_latents, siglip_embeds

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        decoded = self.forge_objects.vae.decode(sample)
        # VAE outputs [0, 1], convert to [-1, 1]
        result = decoded.movedim(-1, 1) * 2.0 - 1.0
        return result.to(x)
