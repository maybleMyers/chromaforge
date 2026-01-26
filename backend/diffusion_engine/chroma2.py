import torch
from einops import rearrange

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management


class Flux2TransformerWrapper(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        # Storage for reference image latents (set before sampling)
        self._ref_latents = None
        self._ref_latent_ids = None

    def set_reference_latents(self, ref_latents, ref_latent_ids):
        """Store reference latents for use during forward passes."""
        self._ref_latents = ref_latents
        self._ref_latent_ids = ref_latent_ids

    def clear_reference_latents(self):
        """Clear reference latents after generation."""
        self._ref_latents = None
        self._ref_latent_ids = None

    @property
    def storage_dtype(self):
        return getattr(self.transformer, 'storage_dtype', next(self.transformer.parameters()).dtype)

    @property
    def dtype(self):
        return getattr(self.transformer, 'dtype', next(self.transformer.parameters()).dtype)

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def __getattr__(self, name):
        if name in ['transformer', '_modules', '_parameters', '_buffers']:
            return super().__getattr__(name)
        try:
            return getattr(self.transformer, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, x, timestep, context, **kwargs):
        bs, c, h, w = x.shape
        input_device = x.device
        input_dtype = x.dtype
        patch_size = 2

        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

        hidden_states = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        del x

        h_len = (h + pad_h) // patch_size
        w_len = (w + pad_w) // patch_size
        noise_seq_len = hidden_states.shape[1]

        # Prepare img_ids for noise latents (T=0)
        img_ids = torch.zeros((h_len * w_len, 4), device=input_device, dtype=input_dtype)
        for i in range(h_len):
            for j in range(w_len):
                idx = i * w_len + j
                img_ids[idx, 1] = i
                img_ids[idx, 2] = j

        # Use stored reference latents (set via set_reference_latents before sampling)
        ref_latents = self._ref_latents
        ref_latent_ids = self._ref_latent_ids

        if ref_latents is not None and ref_latent_ids is not None:
            # Ensure ref_latents matches batch size, device, and dtype
            ref_latents_batch = ref_latents.to(device=input_device, dtype=input_dtype)
            if ref_latents_batch.shape[0] != bs:
                ref_latents_batch = ref_latents_batch.expand(bs, -1, -1)

            # Concatenate noise latents with reference latents
            hidden_states = torch.cat([hidden_states, ref_latents_batch], dim=1)
            img_ids = torch.cat([img_ids, ref_latent_ids.squeeze(0).to(device=input_device, dtype=input_dtype)], dim=0)

        # txt_ids format: [time=0, height=0, width=0, seq_position]
        txt_seq_len = context.shape[1]
        txt_ids = torch.zeros((txt_seq_len, 4), device=input_device, dtype=input_dtype)
        txt_ids[:, 3] = torch.arange(txt_seq_len, device=input_device, dtype=input_dtype)

        guidance = torch.full((bs,), 3.0, device=input_device, dtype=input_dtype)

        result = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False
        )

        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result

        # Only take the noise prediction part (not reference latents)
        out = out[:, :noise_seq_len, :]

        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)
        out = out[:, :, :h, :w]
        return out


class Chroma2Klein(ForgeDiffusionEngine):
    matched_guesses = [model_list.Chroma2Klein]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(
            model_dict={
                'qwen': huggingface_components['text_encoder']
            },
            tokenizer_dict={
                'qwen': huggingface_components['tokenizer']
            }
        )

        vae_model = huggingface_components['vae']
        vae = VAE(model=vae_model)
        vae.latent_channels = 32

        # Store batch norm stats for FLUX.2 latent normalization
        # AutoencoderKLFlux2 has a built-in bn layer
        if hasattr(vae_model, 'bn') and hasattr(vae_model.bn, 'running_mean'):
            self.bn_mean = vae_model.bn.running_mean
            self.bn_var = vae_model.bn.running_var
            self.bn_eps = getattr(vae_model.config, 'batch_norm_eps', 1e-4)
            print("DEBUG: Using FLUX.2 VAE batch norm from model")
        elif 'vae_bn_mean' in huggingface_components and 'vae_bn_var' in huggingface_components:
            self.bn_mean = huggingface_components['vae_bn_mean']
            self.bn_var = huggingface_components['vae_bn_var']
            self.bn_eps = 1e-4
            print("DEBUG: Using FLUX.2 VAE batch norm from state dict")
        else:
            self.bn_mean = None
            self.bn_var = None
            self.bn_eps = 1e-4
            print("DEBUG: No VAE batch norm stats found")

        k_predictor = PredictionFlux(mu=1.0)

        wrapped_transformer = Flux2TransformerWrapper(huggingface_components['transformer'])

        unet = UnetPatcher.from_model(
            model=wrapped_transformer,
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
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

        tokenizer = self.text_processing_engine_qwen.tokenizer
        text_encoder = self.text_processing_engine_qwen.text_encoder

        # Format prompts as chat messages and apply chat template (like official pipelines)
        formatted_prompts = []
        for p in prompt:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted)

        inputs = tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        # Use memory management to get correct device (handles partial offloading)
        device = memory_management.text_encoder_device()

        # Sync embedding layer to device (like T5Engine does for partial offloading)
        text_encoder.embed_tokens.to(device=device)

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # FLUX.2 uses specific hidden state layers (10, 20, 30), NOT the last 3 layers
        # Use direct indexing as in Flux2Pipeline
        hidden_states_layers = (10, 20, 30)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Stack and reshape: (B, 3, seq, hidden) -> (B, seq, 3*hidden)
        out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
        hidden_states = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        cond = {'crossattn': hidden_states, 'attention_mask': attention_mask}
        cond['guidance'] = torch.FloatTensor([3.0] * len(prompt))
        return cond

    def set_reference_images(self, images, device=None, dtype=None):
        """
        Set reference images for FLUX.2 image conditioning.
        Called before sampling starts. Stores latents directly on transformer wrapper.

        Args:
            images: List of PIL images to use as reference
            device: Target device (defaults to model device)
            dtype: Target dtype (defaults to model dtype)
        """
        if images is None or len(images) == 0:
            self.forge_objects.unet.model.diffusion_model.clear_reference_latents()
            return

        if device is None:
            device = self.forge_objects.unet.model.diffusion_model.device
        if dtype is None:
            dtype = self.forge_objects.unet.model.diffusion_model.dtype

        ref_latents, ref_latent_ids = self.prepare_reference_latents(images, device, dtype)

        # Store directly on transformer wrapper (bypasses condition dict)
        self.forge_objects.unet.model.diffusion_model.set_reference_latents(ref_latents, ref_latent_ids)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_qwen.tokenize([prompt])[0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def prepare_reference_latents(self, images, device, dtype):
        """
        Encode reference images and prepare their latents and IDs for FLUX.2 conditioning.

        Args:
            images: List of PIL images or tensors (B, C, H, W) in range [0, 1]
            device: Target device
            dtype: Target dtype

        Returns:
            ref_latents: (1, N*seq_len, 128) - packed reference latents
            ref_latent_ids: (1, N*seq_len, 4) - position IDs with unique time coords
        """
        if images is None or len(images) == 0:
            return None, None

        patch_size = 2
        ref_scale = 10  # Time coordinate scale for reference images

        all_packed_latents = []
        all_ids = []

        for img_idx, image in enumerate(images):
            # Convert PIL to tensor if needed
            if hasattr(image, 'convert'):
                import torchvision.transforms.functional as TF
                image = TF.to_tensor(image).unsqueeze(0)  # (1, 3, H, W)

            # Ensure correct format
            if image.dim() == 3:
                image = image.unsqueeze(0)

            image = image.to(device=device, dtype=dtype)

            # Encode through VAE (image should be in [0, 1] range, convert to [-1, 1])
            image_input = image * 2.0 - 1.0
            latent = self.forge_objects.vae.encode(image_input.movedim(1, -1) * 0.5 + 0.5)

            # Apply batch norm normalization
            if self.bn_mean is not None:
                batch_size, channels, h, w = latent.shape

                # Patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)
                latent_patchified = latent.view(batch_size, channels, h // 2, 2, w // 2, 2)
                latent_patchified = latent_patchified.permute(0, 1, 3, 5, 2, 4)
                latent_patchified = latent_patchified.reshape(batch_size, channels * 4, h // 2, w // 2)

                # Apply batch norm normalization: (x - mean) / std
                bn_mean = self.bn_mean.view(1, -1, 1, 1).to(latent_patchified.device, dtype=latent_patchified.dtype)
                bn_std = torch.sqrt(self.bn_var.view(1, -1, 1, 1).to(latent_patchified.device, dtype=latent_patchified.dtype) + self.bn_eps)
                latent_patchified = (latent_patchified - bn_mean) / bn_std

                # Pack latents: (B, 128, H/2, W/2) -> (B, seq, 128)
                _, c_patch, h_patch, w_patch = latent_patchified.shape
                packed = latent_patchified.permute(0, 2, 3, 1).reshape(1, h_patch * w_patch, c_patch)
            else:
                # Fallback: pack without BN
                _, channels, h, w = latent.shape
                packed = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
                h_patch = h // patch_size
                w_patch = w // patch_size

            # Create position IDs with unique time coordinate for this reference image
            # Format: [T, H, W, L] where T = ref_scale + ref_scale * img_idx
            t_coord = ref_scale + ref_scale * img_idx
            ref_ids = torch.zeros((h_patch * w_patch, 4), device=device, dtype=dtype)
            for i in range(h_patch):
                for j in range(w_patch):
                    idx = i * w_patch + j
                    ref_ids[idx, 0] = t_coord  # Time coordinate
                    ref_ids[idx, 1] = i        # Height
                    ref_ids[idx, 2] = j        # Width
                    # ref_ids[idx, 3] = 0      # Layer (already 0)

            all_packed_latents.append(packed.squeeze(0))  # (seq, 128)
            all_ids.append(ref_ids)

        # Concatenate all reference latents and IDs
        ref_latents = torch.cat(all_packed_latents, dim=0).unsqueeze(0)  # (1, total_seq, 128)
        ref_latent_ids = torch.cat(all_ids, dim=0).unsqueeze(0)  # (1, total_seq, 4)

        return ref_latents, ref_latent_ids

    @torch.inference_mode()
    def encode_first_stage(self, x):
        # Encode image to latents
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)

        # For FLUX.2 VAE, apply batch norm normalization (inverse of decode denormalization)
        if self.bn_mean is not None:
            batch_size, channels, h, w = sample.shape

            # Patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)
            sample_patchified = sample.view(batch_size, channels, h // 2, 2, w // 2, 2)
            sample_patchified = sample_patchified.permute(0, 1, 3, 5, 2, 4)
            sample_patchified = sample_patchified.reshape(batch_size, channels * 4, h // 2, w // 2)

            # Apply batch norm normalization: (x - mean) / std
            bn_mean = self.bn_mean.view(1, -1, 1, 1).to(sample_patchified.device, dtype=sample_patchified.dtype)
            bn_std = torch.sqrt(self.bn_var.view(1, -1, 1, 1).to(sample_patchified.device, dtype=sample_patchified.dtype) + self.bn_eps)
            sample_patchified = (sample_patchified - bn_mean) / bn_std

            # Unpatchify: (B, 128, H/2, W/2) -> (B, 32, H, W)
            sample = sample_patchified.reshape(batch_size, channels, 2, 2, h // 2, w // 2)
            sample = sample.permute(0, 1, 4, 2, 5, 3)
            sample = sample.reshape(batch_size, channels, h, w)

        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        # x is in unpatchified format: (B, 32, H, W)
        # FLUX.2 VAE requires batch norm denormalization in patchified space

        if self.bn_mean is not None:
            batch_size, channels, h, w = x.shape

            # Patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)
            # Following Flux2Pipeline._patchify_latents
            x_patchified = x.view(batch_size, channels, h // 2, 2, w // 2, 2)
            x_patchified = x_patchified.permute(0, 1, 3, 5, 2, 4)
            x_patchified = x_patchified.reshape(batch_size, channels * 4, h // 2, w // 2)

            # Apply batch norm denormalization
            bn_mean = self.bn_mean.view(1, -1, 1, 1).to(x_patchified.device, dtype=x_patchified.dtype)
            bn_std = torch.sqrt(self.bn_var.view(1, -1, 1, 1).to(x_patchified.device, dtype=x_patchified.dtype) + self.bn_eps)
            x_patchified = x_patchified * bn_std + bn_mean

            # Unpatchify: (B, 128, H/2, W/2) -> (B, 32, H, W)
            # Following Flux2Pipeline._unpatchify_latents
            x = x_patchified.reshape(batch_size, channels, 2, 2, h // 2, w // 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.reshape(batch_size, channels, h, w)

        # VAE decode
        sample = self.forge_objects.vae.decode(x).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
