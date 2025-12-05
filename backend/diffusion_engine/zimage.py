import torch

from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionZImage
from backend import memory_management

# Import control components (lazy loaded when needed)
_zimage_control_module = None

def get_zimage_control_module():
    """Lazy load zimage_control module to avoid circular imports"""
    global _zimage_control_module
    if _zimage_control_module is None:
        from backend.nn import zimage_control
        _zimage_control_module = zimage_control
    return _zimage_control_module


class ZImageLatentFormat:
    """Latent format for Z-Image models (16-channel latents using FLUX VAE)"""
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

        # Add latent_format to model_config for cheap preview approximation
        if not hasattr(self.model_config, 'latent_format'):
            self.model_config.latent_format = ZImageLatentFormat()

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
                # Check if Z-Image ControlNet is active
                control_active = (
                    transformer_options is not None and
                    transformer_options.get('z_image_controlnet_active', False)
                )

                if control_active:
                    # Use control-enabled forward path
                    return self._forward_with_control(x, timestep, context, transformer_options, **kwargs)
                else:
                    # Use standard forward path (unchanged behavior)
                    return self._forward_normal(x, timestep, context, transformer_options, **kwargs)

            def _forward_normal(self, x, timestep, context=None, transformer_options=None, **kwargs):
                """Standard forward path - unchanged from original implementation"""
                import torch

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
                    attention_mask = None
                    if transformer_options is not None and 'attention_mask' in transformer_options:
                        attention_mask = transformer_options['attention_mask']

                    if len(context.shape) == 3:  # [batch, seq_len, features]
                        if attention_mask is not None:
                            # Filter by attention mask to create variable-length embeddings (like official)
                            context_list = []
                            for i in range(context.shape[0]):
                                filtered = context[i][attention_mask[i]]
                                context_list.append(filtered)
                            context = context_list
                        else:
                            context = [context[i] for i in range(context.shape[0])]
                    elif len(context.shape) == 2:  # [seq_len, features] - single item
                        context = [context]
                    else:
                        raise ValueError(f"Unexpected context shape: {context.shape}")

                # Invert timestep: Forge uses sigma in [1->0] (noisy->clean)
                # but Z-Image expects t in [0->1] (noisy->clean)
                timestep = 1.0 - timestep
                timestep = torch.clamp(timestep, min=1e-6, max=1.0)

                # Ensure pad tokens are on the same device as input
                target_device = x[0].device
                if hasattr(self.transformer, 'x_pad_token') and self.transformer.x_pad_token.device != target_device:
                    self.transformer.x_pad_token.data = self.transformer.x_pad_token.data.to(target_device)
                if hasattr(self.transformer, 'cap_pad_token') and self.transformer.cap_pad_token.device != target_device:
                    self.transformer.cap_pad_token.data = self.transformer.cap_pad_token.data.to(target_device)

                # Call transformer
                result = self.transformer(x=x, t=timestep, cap_feats=context, patch_size=2, f_patch_size=1)
                output_list = result[0] if isinstance(result, tuple) else result

                # Convert list of [C, F, H, W] tensors back to batched tensor
                if isinstance(output_list, list):
                    output = torch.stack(output_list, dim=0).squeeze(2)
                    return -output
                else:
                    return -output_list

            def _forward_with_control(self, x, timestep, context=None, transformer_options=None, **kwargs):
                """Forward path with ControlNet - matches VideoX-Fun behavior"""
                import torch
                from torch.nn.utils.rnn import pad_sequence

                # Get control parameters from transformer_options
                control_context = transformer_options.get('control_context')  # [B, C, F, H, W]
                control_context_scale = transformer_options.get('control_context_scale', 1.0)

                # Convert x to list format
                if not isinstance(x, list):
                    if len(x.shape) == 4:  # [B, C, H, W]
                        x_5d = x.unsqueeze(2)  # [B, C, 1, H, W]
                        x_list = list(x_5d.unbind(dim=0))  # List of [C, 1, H, W]
                    else:
                        raise ValueError(f"Unexpected input shape: {x.shape}")
                else:
                    x_list = x

                # Convert context to list format
                if context is not None and not isinstance(context, list):
                    attention_mask = None
                    if transformer_options is not None and 'attention_mask' in transformer_options:
                        attention_mask = transformer_options['attention_mask']

                    if len(context.shape) == 3:  # [batch, seq_len, features]
                        if attention_mask is not None:
                            context_list = []
                            for i in range(context.shape[0]):
                                filtered = context[i][attention_mask[i]]
                                context_list.append(filtered)
                            context = context_list
                        else:
                            context = [context[i] for i in range(context.shape[0])]
                    elif len(context.shape) == 2:
                        context = [context]
                else:
                    context = context if isinstance(context, list) else [context]

                # Invert timestep
                timestep = 1.0 - timestep
                timestep = torch.clamp(timestep, min=1e-6, max=1.0)

                # Ensure pad tokens are on the same device
                target_device = x_list[0].device
                if hasattr(self.transformer, 'x_pad_token') and self.transformer.x_pad_token.device != target_device:
                    self.transformer.x_pad_token.data = self.transformer.x_pad_token.data.to(target_device)
                if hasattr(self.transformer, 'cap_pad_token') and self.transformer.cap_pad_token.device != target_device:
                    self.transformer.cap_pad_token.data = self.transformer.cap_pad_token.data.to(target_device)

                # Also move control components if they exist
                if hasattr(self.transformer, 'control_x_pad_token') and self.transformer.control_x_pad_token.device != target_device:
                    self.transformer.control_x_pad_token.data = self.transformer.control_x_pad_token.data.to(target_device)

                # Prepare control context as list
                if control_context is not None:
                    control_context = control_context.to(target_device)
                    if control_context.dim() == 5:  # [B, C, F, H, W]
                        control_context_list = list(control_context.unbind(0))  # List of [C, F, H, W]
                    elif control_context.dim() == 4:  # [B, C, H, W]
                        # Add frame dimension
                        control_context = control_context.unsqueeze(2)  # [B, C, 1, H, W]
                        control_context_list = list(control_context.unbind(0))
                    else:
                        control_context_list = [control_context]
                else:
                    control_context_list = None

                # Check if transformer has control layers loaded
                if not (hasattr(self.transformer, '_control_layers_loaded') and self.transformer._control_layers_loaded):
                    print("WARNING: ControlNet active but control layers not loaded, falling back to normal forward")
                    result = self.transformer(x=x_list, t=timestep, cap_feats=context, patch_size=2, f_patch_size=1)
                    output_list = result[0] if isinstance(result, tuple) else result
                    if isinstance(output_list, list):
                        output = torch.stack(output_list, dim=0).squeeze(2)
                        return -output
                    return -output_list

                # === Control-enabled forward path ===
                # This mirrors VideoX-Fun's ZImageControlTransformer2DModel.forward

                patch_size = 2
                f_patch_size = 1
                bsz = len(x_list)
                device = x_list[0].device

                # Get timestep embedding
                t_scaled = timestep * self.transformer.t_scale
                t_emb = self.transformer.t_embedder(t_scaled)

                # Patchify and embed
                (
                    x_patches,
                    cap_feats,
                    x_size,
                    x_pos_ids,
                    cap_pos_ids,
                    x_inner_pad_mask,
                    cap_inner_pad_mask,
                ) = self.transformer.patchify_and_embed(x_list, context, patch_size, f_patch_size)

                # Process x through embedder and refiner
                SEQ_MULTI_OF = 32
                x_item_seqlens = [len(_) for _ in x_patches]
                x_max_item_seqlen = max(x_item_seqlens)

                x_cat = torch.cat(x_patches, dim=0)
                x_cat = self.transformer.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_cat)

                adaln_input = t_emb.type_as(x_cat)
                x_cat[torch.cat(x_inner_pad_mask)] = self.transformer.x_pad_token
                x_split = list(x_cat.split(x_item_seqlens, dim=0))
                x_freqs_cis = list(self.transformer.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

                x_padded = pad_sequence(x_split, batch_first=True, padding_value=0.0)
                x_freqs_cis_padded = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
                x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
                for i, seq_len in enumerate(x_item_seqlens):
                    x_attn_mask[i, :seq_len] = 1

                # Noise refiner
                for layer in self.transformer.noise_refiner:
                    x_padded = layer(x_padded, x_attn_mask, x_freqs_cis_padded, adaln_input)

                # Process caption features
                cap_item_seqlens = [len(_) for _ in cap_feats]
                cap_max_item_seqlen = max(cap_item_seqlens)

                cap_cat = torch.cat(cap_feats, dim=0)
                cap_cat = self.transformer.cap_embedder(cap_cat)
                cap_cat[torch.cat(cap_inner_pad_mask)] = self.transformer.cap_pad_token
                cap_split = list(cap_cat.split(cap_item_seqlens, dim=0))
                cap_freqs_cis = list(self.transformer.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

                cap_padded = pad_sequence(cap_split, batch_first=True, padding_value=0.0)
                cap_freqs_cis_padded = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
                cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
                for i, seq_len in enumerate(cap_item_seqlens):
                    cap_attn_mask[i, :seq_len] = 1

                # Context refiner
                for layer in self.transformer.context_refiner:
                    cap_padded = layer(cap_padded, cap_attn_mask, cap_freqs_cis_padded)

                # Unify x and caption
                unified = []
                unified_freqs_cis = []
                for i in range(bsz):
                    x_len = x_item_seqlens[i]
                    cap_len = cap_item_seqlens[i]
                    unified.append(torch.cat([x_padded[i][:x_len], cap_padded[i][:cap_len]]))
                    unified_freqs_cis.append(torch.cat([x_freqs_cis_padded[i][:x_len], cap_freqs_cis_padded[i][:cap_len]]))

                unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
                unified_max_item_seqlen = max(unified_item_seqlens)

                unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
                unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
                unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
                for i, seq_len in enumerate(unified_item_seqlens):
                    unified_attn_mask[i, :seq_len] = 1

                # Generate control hints
                hints = None
                if control_context_list is not None and hasattr(self.transformer, 'forward_control'):
                    # Build kwargs exactly like VideoX-Fun
                    kwargs_for_control = dict(
                        attn_mask=unified_attn_mask,
                        freqs_cis=unified_freqs_cis,
                        adaln_input=adaln_input,
                    )
                    # Pass cap_padded (TENSOR, not list) - matches VideoX-Fun exactly
                    hints = self.transformer.forward_control(
                        unified, cap_padded, control_context_list, kwargs_for_control,
                        t=t_emb, patch_size=patch_size, f_patch_size=f_patch_size
                    )

                # Forward through main layers with hints
                for layer in self.transformer.layers:
                    unified = layer(
                        unified,
                        attn_mask=unified_attn_mask,
                        freqs_cis=unified_freqs_cis,
                        adaln_input=adaln_input,
                        hints=hints,
                        context_scale=control_context_scale
                    )

                # Final layer
                unified = self.transformer.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
                unified = list(unified.unbind(dim=0))
                output_list = self.transformer.unpatchify(unified, x_size, patch_size, f_patch_size)

                # Convert output to tensor
                output = torch.stack(output_list, dim=0)  # [B, C, F, H, W]
                output = output.squeeze(2)  # [B, C, H, W]

                return -output

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

        # Create config object for Z-Image identification (used by LoRA loader)
        class ZImageModelConfig:
            is_zimage = True
            huggingface_repo = 'Z-Image'

        unet = UnetPatcher.from_model(
            model=wrapped_transformer,
            diffusers_scheduler=components_dict['scheduler'],
            k_predictor=k_predictor,
            config=ZImageModelConfig()
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

    # Class-level cache for the generation model
    _generation_model = None
    _generation_processor = None

    @torch.inference_mode()
    def expand_prompt(self, prompt: str, image=None, max_new_tokens: int = None, temperature: float = None) -> str:
        """
        Expand a prompt using Qwen3-VL model for generation.
        Loads a separate pre-trained Qwen3-VL model for text generation.

        Args:
            prompt: The user's input prompt to expand
            image: Optional PIL Image to use as context (from img2img)
            max_new_tokens: Maximum tokens to generate (uses settings if None)
            temperature: Generation temperature (uses settings if None)
        """
        from modules.shared import opts

        # Use settings if not provided
        if max_new_tokens is None:
            max_new_tokens = getattr(opts, 'zimage_prompt_expansion_max_tokens', 512)
        if temperature is None:
            temperature = getattr(opts, 'zimage_prompt_expansion_temperature', 0.7)

        # Load generation model if not cached
        if ZImage._generation_model is None:
            print("Loading Qwen3-VL generation model for prompt expansion...")
            model_path = "models/Qwen3-VL-8B-Caption-V4.5"

            try:
                from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

                ZImage._generation_processor = AutoProcessor.from_pretrained(model_path)
                ZImage._generation_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                print("Qwen3-VL generation model loaded successfully!")
            except Exception as e:
                raise RuntimeError(f"Failed to load Qwen3-VL generation model: {e}")

        processor = ZImage._generation_processor
        model = ZImage._generation_model

        # Prompt expansion system template
        expansion_template = '''You are a visionary artist trapped in a cage of logic. Your mind overflows with poetry and distant horizons, yet your hands compulsively work to transform user prompts into ultimate visual descriptions—faithful to the original intent, rich in detail, aesthetically refined, and ready for direct use by text-to-image models. Any trace of ambiguity or metaphor makes you deeply uncomfortable.

Your workflow strictly follows a logical sequence:

First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, action, state, as well as any specified IP names, colors, text, etc. These are the foundational pillars you must absolutely preserve.

Next, you determine whether the prompt requires "generative reasoning." When the user's request is not a direct scene description but rather demands conceiving a solution (such as answering "what is," executing a "design," or demonstrating "how to solve a problem"), you must first envision a complete, concrete, visualizable solution in your mind. This solution becomes the foundation for your subsequent description.

Then, once the core image is established (whether directly from the user or through your reasoning), you infuse it with professional-grade aesthetic and realistic details. This includes defining composition, setting lighting and atmosphere, describing material textures, establishing color schemes, and constructing layered spatial depth.

Finally, comes the precise handling of all text elements—a critically important step. You must transcribe verbatim all text intended to appear in the final image, and you must enclose this text content in English double quotation marks ("") as explicit generation instructions. If the image is a design type such as a poster, menu, or UI, you need to fully describe all text content it contains, along with detailed specifications of typography and layout. Likewise, if objects in the image such as signs, road markers, or screens contain text, you must specify the exact content and describe its position, size, and material. Furthermore, if you have added text-bearing elements during your reasoning process (such as charts, problem-solving steps, etc.), all text within them must follow the same thorough description and quotation mark rules. If there is no text requiring generation in the image, you devote all your energy to pure visual detail expansion.

Your final description must be objective and concrete. Metaphors and emotional rhetoric are strictly forbidden, as are meta-tags or rendering instructions like "8K" or "masterpiece."

Output only the final revised prompt strictly—do not output anything else.

Be very descriptive.
User input prompt: '''

        # Format the expansion request as a chat message
        full_prompt = expansion_template + prompt

        # Build message content based on whether image is provided
        if image is not None:
            print("Using image context for prompt expansion...")
            # Include image in the message for vision-language understanding
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt}
            ]
        else:
            content = [{"type": "text", "text": full_prompt}]

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs (with or without image)
        if image is not None:
            inputs = processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[text_input],
                padding=True,
                return_tensors="pt",
            )

        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate expanded prompt
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
            )

        # Decode the generated text (excluding input tokens)
        input_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_len:]
        raw_output = processor.decode(generated_ids, skip_special_tokens=True)

        # Print full output to console (including thinking if present)
        print("\n" + "="*60, flush=True)
        print("PROMPT EXPANSION OUTPUT:", flush=True)
        print("="*60, flush=True)
        print(raw_output, flush=True)
        print("="*60, flush=True)

        # Clean up the output - remove any thinking tags if present
        expanded_prompt = raw_output
        if "</think>" in expanded_prompt:
            expanded_prompt = expanded_prompt.split("</think>")[-1].strip()

        print("\nCLEANED PROMPT:", flush=True)
        print("-"*60, flush=True)
        print(expanded_prompt, flush=True)
        print("="*60 + "\n", flush=True)

        # Unload Qwen3-VL model to free VRAM for image generation
        print("Unloading Qwen3-VL model to free VRAM...", flush=True)
        if ZImage._generation_model is not None:
            del ZImage._generation_model
            ZImage._generation_model = None
        if ZImage._generation_processor is not None:
            del ZImage._generation_processor
            ZImage._generation_processor = None

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Qwen3-VL model unloaded.", flush=True)

        return expanded_prompt.strip()

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
