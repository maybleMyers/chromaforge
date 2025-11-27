import torch

from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
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

        # Wrap the transformer to adapt parameter names
        class ZImageTransformerWrapper(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer

            def forward(self, x, timestep, context=None, **kwargs):
                # Z-Image list format is for multi-resolution, not batching
                # Keep batch dimension intact, just wrap in single-item list
                import torch

                if not isinstance(x, list):
                    x = [x]  # Wrap batched tensor in list for multi-res format

                if context is not None:
                    # Split context into list of 2D tensors per batch item
                    if not isinstance(context, list):
                        if len(context.shape) == 3:  # [batch, seq, features]
                            context = [context[i] for i in range(context.shape[0])]
                        else:
                            context = [context]

                # Debug: Check available embedders
                print(f"DEBUG: all_x_embedder keys: {list(self.transformer.all_x_embedder.keys())}")
                print(f"DEBUG: config type: {type(self.transformer.config)}")
                if hasattr(self.transformer.config, 'all_patch_size'):
                    print(f"DEBUG: all_patch_size: {self.transformer.config.all_patch_size}")
                    print(f"DEBUG: in_channels: {self.transformer.config.in_channels}")
                elif isinstance(self.transformer.config, dict):
                    print(f"DEBUG: all_patch_size: {self.transformer.config.get('all_patch_size')}")
                    print(f"DEBUG: in_channels: {self.transformer.config.get('in_channels')}")

                # Explicitly pass patch_size to ensure correct patchification
                return self.transformer(x=x, t=timestep, cap_feats=context, patch_size=2, f_patch_size=1)

            def __getattr__(self, name):
                # Pass through all other attributes to the underlying transformer
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.transformer, name)

        wrapped_transformer = ZImageTransformerWrapper(components_dict['transformer'])

        # Flow matching with mu=1.0 (same as Chroma)
        k_predictor = PredictionFlux(
            mu=1.0
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
        # Get embeddings with attention mask
        embeddings, attention_mask = self.text_processing_engine_qwen(prompt, return_attention_mask=True)
        # Store attention mask in a dict along with embeddings
        # Use 'crossattn' key to match the conditioning system's expectations
        return {'crossattn': embeddings, 'attention_mask': attention_mask}

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
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
