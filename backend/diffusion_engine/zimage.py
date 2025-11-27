import torch

from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionZImage
from backend import memory_management

class ZImage(ForgeDiffusionEngine):
    def __init__(self, components_dict, estimated_config=None):
        # Create minimal config if not provided
        if estimated_config is None:
            class MinimalConfig:
                def inpaint_model(self):
                    return False
            estimated_config = MinimalConfig()

        super().__init__(estimated_config, components_dict)
        self.is_inpaint = False

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
        # This is necessary because the VAE config might be inferred incorrectly (e.g. 8 channels)
        # while the Transformer expects 16 channels.
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
        # Default is 0.18215 / 0.0, but Z-Image needs 0.3611 / 0.1159
        vae.first_stage_model.scaling_factor = 0.3611
        vae.first_stage_model.shift_factor = 0.1159
        if hasattr(vae.first_stage_model.config, 'scaling_factor'):
            vae.first_stage_model.config.scaling_factor = 0.3611
        if hasattr(vae.first_stage_model.config, 'shift_factor'):
            vae.first_stage_model.config.shift_factor = 0.1159

        # Wrap the transformer to adapt parameter names
        class ZImageTransformerWrapper(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer

            def forward(self, x, timestep, context=None, transformer_options=None, **kwargs):
                # Convert input to list format matching official diffusers implementation
                # Official: latent [B,C,H,W] -> unsqueeze(2) -> [B,C,1,H,W] -> unbind(0) -> List of [C,1,H,W]
                import torch

                # DEBUG: Print info on first call only
                if not hasattr(self, '_debug_printed'):
                    self._debug_printed = True
                    print(f"\n=== Z-Image Wrapper Debug (first call) ===")
                    print(f"Input x shape: {x.shape}, dtype: {x.dtype}")
                    print(f"Input x stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
                    print(f"Input x has NaN: {torch.isnan(x).any().item()}")
                    print(f"Timestep (raw sigma): {timestep}")
                    if context is not None:
                        print(f"Context shape: {context.shape}, dtype: {context.dtype}")

                if not isinstance(x, list):
                    # Input should be [batch, channels, height, width] (4D)
                    # Need to add frame dimension and split into list
                    if len(x.shape) == 4:  # [B, C, H, W]
                        x = x.unsqueeze(2)  # [B, C, 1, H, W] - add frame dimension
                        x = list(x.unbind(dim=0))  # List of [C, 1, H, W]
                    else:
                        raise ValueError(f"Unexpected input shape: {x.shape}. Expected 4D tensor [B, C, H, W]")

                # Convert context to list format, filtering by attention mask if available
                if context is not None and not isinstance(context, list):
                    # Check if attention_mask is available in transformer_options
                    attention_mask = None
                    if transformer_options is not None and 'attention_mask' in transformer_options:
                        attention_mask = transformer_options['attention_mask']

                    # DEBUG: Check attention mask
                    if not hasattr(self, '_debug_context_printed'):
                        self._debug_context_printed = True
                        print(f"\n=== Context/Attention Mask Debug ===")
                        print(f"Context shape before filtering: {context.shape}")
                        print(f"Attention mask present: {attention_mask is not None}")
                        if attention_mask is not None:
                            print(f"Attention mask shape: {attention_mask.shape}")
                            print(f"Attention mask sum (non-padded tokens): {attention_mask.sum(dim=1).tolist()}")

                    if len(context.shape) == 3:  # [batch, seq_len, features]
                        if attention_mask is not None:
                            # Filter by attention mask to create variable-length embeddings (like official)
                            context_list = []
                            for i in range(context.shape[0]):
                                # Only keep non-padded tokens
                                filtered = context[i][attention_mask[i]]
                                context_list.append(filtered)
                                if not hasattr(self, '_debug_filtered_printed'):
                                    self._debug_filtered_printed = True
                                    print(f"Filtered context[{i}] shape: {filtered.shape} (from {context[i].shape})")
                            context = context_list
                        else:
                            # No attention mask, just split batch
                            context = [context[i] for i in range(context.shape[0])]
                            if not hasattr(self, '_debug_nofilter_printed'):
                                self._debug_nofilter_printed = True
                                print(f"WARNING: No attention mask - using full padded context!")
                    elif len(context.shape) == 2:  # [seq_len, features] - single item
                        context = [context]
                    else:
                        raise ValueError(f"Unexpected context shape: {context.shape}")

                    if not hasattr(self, '_debug_final_context_printed'):
                        self._debug_final_context_printed = True
                        print(f"Final context: list of {len(context)} tensors, shapes: {[c.shape for c in context]}")
                        print(f"====================================\n")

                # Invert timestep: Forge uses sigma in [1->0] (noisy->clean)
                # but Z-Image expects t in [0->1] (noisy->clean)
                # The transformer then multiplies by t_scale=1000 internally
                original_timestep = timestep
                timestep = 1.0 - timestep
                # Clamp to avoid exactly 0.0 which can cause numerical issues
                timestep = torch.clamp(timestep, min=1e-6, max=1.0)

                # DEBUG: Print transformed timestep on first call
                if hasattr(self, '_debug_printed') and not hasattr(self, '_debug_timestep_printed'):
                    self._debug_timestep_printed = True
                    print(f"Transformed timestep: {original_timestep} -> {timestep}")
                    print(f"(Official would use: (1000 - t)/1000 where t ≈ sigma*1000)")

                # Ensure pad tokens are on the same device as input (fixes CPU/GPU mismatch when model is partially offloaded)
                target_device = x[0].device
                if hasattr(self.transformer, 'x_pad_token') and self.transformer.x_pad_token.device != target_device:
                    self.transformer.x_pad_token.data = self.transformer.x_pad_token.data.to(target_device)
                if hasattr(self.transformer, 'cap_pad_token') and self.transformer.cap_pad_token.device != target_device:
                    self.transformer.cap_pad_token.data = self.transformer.cap_pad_token.data.to(target_device)

                # Call transformer with correct format
                # patch_size=2 and f_patch_size=1 are the only supported values per config
                # Transformer returns (list_of_tensors, {}), extract the list
                result = self.transformer(x=x, t=timestep, cap_feats=context, patch_size=2, f_patch_size=1)
                output_list = result[0] if isinstance(result, tuple) else result

                # Convert list of [C, F, H, W] tensors back to batched [B, C, F, H, W] tensor
                # Then squeeze frame dimension to get [B, C, H, W]
                if isinstance(output_list, list):
                    output = torch.stack(output_list, dim=0)  # [B, C, F, H, W]
                    output = output.squeeze(2)  # [B, C, H, W] - remove frame dimension

                    # DEBUG: Check transformer output
                    if hasattr(self, '_debug_printed') and not hasattr(self, '_debug_output_printed'):
                        self._debug_output_printed = True
                        print(f"\n=== Transformer Output Debug ===")
                        print(f"Output shape: {output.shape}, dtype: {output.dtype}")
                        print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
                        print(f"Output has NaN: {torch.isnan(output).any().item()}")
                        print(f"Output has Inf: {torch.isinf(output).any().item()}")
                        print(f"=================================\n")

                    return -output
                else:
                    return -output_list

            def __getattr__(self, name):
                # Pass through all other attributes to the underlying transformer
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.transformer, name)

        wrapped_transformer = ZImageTransformerWrapper(components_dict['transformer'])

        # Z-Image uses static shift=3.0 (from scheduler config)
        # This matches the formula: sigmas = shift * t / (1 + (shift-1) * t)
        k_predictor = PredictionZImage(
            shift=3.0,
            timesteps=1000
        )

        unet = UnetPatcher.from_model(
            model=wrapped_transformer,
            diffusers_scheduler=components_dict['scheduler'],
            k_predictor=k_predictor,
            config=None
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
        # Format prompts with chat template like official implementation
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

        # Move to device
        device = text_encoder.device
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Encode (use hidden_states[-2] like official implementation)
        outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-2]

        # Filter by attention mask to get variable-length embeddings (official approach)
        # This removes padding tokens, creating a list of 2D tensors with different lengths
        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        # However, for Forge backend compatibility, we need to return a batched tensor
        # So we'll return the full padded embeddings with attention mask
        # The wrapper will handle splitting into list format
        return {'crossattn': prompt_embeds, 'attention_mask': prompt_masks}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_qwen.tokenize([prompt])[0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        print(f"\n=== VAE Decode Debug ===")
        print(f"Input latents: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")

        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        print(f"After process_out: min={sample.min().item():.4f}, max={sample.max().item():.4f}")

        decoded = self.forge_objects.vae.decode(sample)
        print(f"VAE decode output: min={decoded.min().item():.4f}, max={decoded.max().item():.4f}")
        print(f"VAE decode output shape: {decoded.shape}")

        # Note: If VAE outputs [-1, 1], we should NOT do * 2.0 - 1.0
        # If VAE outputs [0, 1], we need * 2.0 - 1.0 to convert to [-1, 1]
        sample = decoded.movedim(-1, 1) * 2.0 - 1.0
        print(f"After * 2.0 - 1.0: min={sample.min().item():.4f}, max={sample.max().item():.4f}")
        print(f"=========================\n")

        return sample.to(x)
