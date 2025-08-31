import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.text_processing.t5_engine import T5TextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management

class ChromaDCT(ForgeDiffusionEngine):
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

        # ChromaDCT operates in pixel space, so no VAE needed
        # vae = None  # DCT model operates directly in pixel space
        
        k_predictor = PredictionFlux(
            mu=1.0
        )
        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )
        
        # Import and set ChromaDCT-specific memory estimation to prevent overestimation
        from backend.nn.model_dct import chroma_dct_memory_estimation
        unet.set_memory_peak_estimation_modifier(chroma_dct_memory_estimation)
        
        # Enable optimized offloading for ChromaDCT models
        self.use_optimized_offloading = True

        self.text_processing_engine_t5 = T5TextProcessingEngine(
            text_encoder=clip.cond_stage_model.t5xxl,
            tokenizer=clip.tokenizer.t5xxl,
            emphasis_name=dynamic_args['emphasis_name'],
            min_length=1
        )

        # Create forge objects without VAE since DCT operates in pixel space
        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=None, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def set_clip_skip(self, clip_skip):
        pass
        
    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        print(f"[CHROMADCT DEBUG] Loading CLIP model for text conditioning...")
        
        # Check memory before loading
        if hasattr(memory_management, 'get_free_memory'):
            free_mem = memory_management.get_free_memory()
            print(f"[CHROMADCT DEBUG] Free memory before CLIP load: {free_mem / (1024**2):.1f} MB")
        
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        
        # Check memory after loading
        if hasattr(memory_management, 'get_free_memory'):
            free_mem = memory_management.get_free_memory()
            print(f"[CHROMADCT DEBUG] Free memory after CLIP load: {free_mem / (1024**2):.1f} MB")
        
        print(f"[CHROMADCT DEBUG] Processing text conditioning for {len(prompt)} prompts...")
        result = self.text_processing_engine_t5(prompt)
        print(f"[CHROMADCT DEBUG] Text conditioning complete, result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        return result

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        # DCT model operates directly in pixel space
        # Convert from [-1, 1] to [0, 1] range for pixel values
        # Input x is expected to be in [-1, 1] range from UI
        # DCT model expects pixel values in [0, 1] or [-1, 1] depending on training
        # Based on the training code, it seems to expect values in standard range
        
        print(f"[CHROMADCT DEBUG] Encode first stage - input shape: {x.shape}, dtype: {x.dtype}, range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"[CHROMADCT DEBUG] DCT model operates directly in pixel space, no VAE encoding needed")
        return x

    @torch.inference_mode()
    def decode_first_stage(self, x):
        # DCT model outputs directly in pixel space
        # No decoding needed, just ensure proper range
        # Output should be in [-1, 1] range for UI compatibility
        
        print(f"[CHROMADCT DEBUG] Decode first stage - input shape: {x.shape}, dtype: {x.dtype}, range: [{x.min():.3f}, {x.max():.3f}]")
        result = x.clamp(-1.0, 1.0)
        print(f"[CHROMADCT DEBUG] DCT model outputs directly in pixel space, clamped to range: [{result.min():.3f}, {result.max():.3f}]")
        return result