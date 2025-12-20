import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.t5_engine import T5TextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management

class Chroma(ForgeDiffusionEngine):
    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        # Cache for prompt embeddings to avoid reloading text encoder for identical prompts
        self._prompt_cache = {}
        self._prompt_cache_max_size = 20  # Limit cache size to prevent memory bloat

        clip = CLIP(
            model_dict={
                't5xxl': huggingface_components['text_encoder']
            },
            tokenizer_dict={
                't5xxl': huggingface_components['tokenizer']
            }
        )

        vae = VAE(model=huggingface_components['vae'])
        k_predictor = PredictionFlux(
            mu=1.0
        )
        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )

        self.text_processing_engine_t5 = T5TextProcessingEngine(
            text_encoder=clip.cond_stage_model.t5xxl,
            tokenizer=clip.tokenizer.t5xxl,
            emphasis_name=dynamic_args['emphasis_name'],
            min_length=1
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def set_clip_skip(self, clip_skip):
        pass

    def clear_prompt_cache(self):
        """Clear the prompt embedding cache."""
        self._prompt_cache.clear()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        # Create cache key from prompt list
        cache_key = tuple(prompt)

        # Check cache first
        if cache_key in self._prompt_cache:
            cached = self._prompt_cache[cache_key]
            # Move to end (LRU behavior) and return clone on GPU
            del self._prompt_cache[cache_key]
            self._prompt_cache[cache_key] = cached
            device = memory_management.text_encoder_device()
            return {
                'crossattn': cached['crossattn'].clone().to(device),
                'attention_mask': cached['attention_mask'].clone().to(device)
            }

        # Cache miss - need to load text encoder and encode
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        # Get embeddings with attention mask
        embeddings, attention_mask = self.text_processing_engine_t5(prompt, return_attention_mask=True)

        # Cache the result (store on CPU to save GPU memory)
        cached_embeddings = embeddings.cpu()
        cached_mask = attention_mask.cpu()

        # Evict oldest entry if cache is full
        if len(self._prompt_cache) >= self._prompt_cache_max_size:
            oldest_key = next(iter(self._prompt_cache))
            del self._prompt_cache[oldest_key]

        self._prompt_cache[cache_key] = {
            'crossattn': cached_embeddings,
            'attention_mask': cached_mask
        }

        # Store attention mask in a dict along with embeddings
        # Use 'crossattn' key to match the conditioning system's expectations
        return {'crossattn': embeddings, 'attention_mask': attention_mask}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)        
