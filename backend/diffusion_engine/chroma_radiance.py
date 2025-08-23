import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.text_processing.t5_engine import T5TextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management


class ChromaRadianceVAEStub:
    """Minimal VAE interface for ChromaRadiance compatibility with processing pipeline."""
    def __init__(self):
        self.latent_channels = 3  # ChromaRadiance works directly with 3-channel RGB
        self.downscale_ratio = 1  # No scaling - direct RGB processing
        
    def clone(self):
        return ChromaRadianceVAEStub()


class ChromaRadiance(ForgeDiffusionEngine):
    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(
            model_dict={
                't5xxl': huggingface_components['text_encoder']
            },
            tokenizer_dict={
                't5xxl': huggingface_components['tokenizer']
            }
        )

        # ChromaRadiance works directly with RGB but needs minimal VAE interface
        # for compatibility with processing pipeline
        vae = ChromaRadianceVAEStub()
        
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
        
    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_t5(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        """
        ChromaRadiance works directly with RGB input - no encoding needed.
        Just normalize to [-1, 1] range expected by the model.
        """
        # Normalize RGB from [0, 1] to [-1, 1] for model input
        if x.min() >= 0 and x.max() <= 1.1:  # Input in [0, 1] range
            x = x * 2.0 - 1.0  # Convert to [-1, 1]
        return x

    @torch.inference_mode() 
    def decode_first_stage(self, x):
        """
        ChromaRadiance outputs RGB directly - just denormalize from [-1, 1] to [0, 1].
        """
        # Convert from [-1, 1] to [0, 1] range
        sample = (x + 1.0) * 0.5
        return sample.clamp(0, 1)

    def has_vae(self):
        """ChromaRadiance works directly with RGB - no separate VAE."""
        return False
        
    def is_radiance_model(self):
        """Identifier for radiance models."""
        return True