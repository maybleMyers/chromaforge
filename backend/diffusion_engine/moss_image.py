import json
import os

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.nn.moss_image import (
    MOSS_LATENT_CHANNELS,
    MOSS_VAE_SCALE,
    MOSSImageTransformer,
    MossFlux2VAEAdapter,
    PredictionMoss,
    _RMSNorm,
    parse_skip_layers,
)
from backend.operations import using_forge_operations
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.state_dict import load_state_dict

MAX_SEQUENCE_LENGTH = 4096


class MossLatentFormat:
    """MOSS samples in the batch-norm normalized, packed latent space; the
    normalization happens inside MossFlux2VAEAdapter."""

    latent_rgb_factors = None
    latent_rgb_factors_bias = None

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class MossTransformerWrapper(torch.nn.Module):
    """Adapts Forge's (x, timestep, context) UNet interface to the MOSS DiT.

    Forge passes sigma; the predictor maps it to MOSS' 0-1000 timestep and treats
    the model output as flow velocity. The negative (uncond) branch optionally
    skips transformer blocks (default layer 22), like the reference pipeline's
    uncond_skip_layers.
    """

    def __init__(self, transformer, depth):
        super().__init__()
        self.transformer = transformer
        self.depth = depth
        self._skip_cache = None
        self._skip_source = None

    def get_skip_layers(self):
        from modules import shared
        raw = getattr(shared.opts, 'moss_skip_layers', '22')
        if self._skip_cache is None or self._skip_source != raw:
            self._skip_cache = parse_skip_layers(raw, self.depth)
            self._skip_source = raw
        return self._skip_cache

    def forward(self, x, timestep, context=None, transformer_options=None, **kwargs):
        b = x.shape[0]

        mask = None
        if transformer_options is not None:
            mask = transformer_options.get('attention_mask', None)
        if mask is None:
            mask = torch.ones(context.shape[:2], device=x.device, dtype=torch.bool)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)

        t = timestep.reshape(-1)
        if t.shape[0] != b:
            t = t.expand(b)
        t = t.to(device=x.device)

        cond_mark = None
        if transformer_options is not None:
            cond_mark = transformer_options.get('cond_mark', None)

        skip = tuple(self.get_skip_layers())
        if skip and cond_mark is not None:
            uncond_mask = (cond_mark.reshape(-1) > 0.5).to(device=x.device, dtype=torch.bool)
            has_uncond = bool(uncond_mask.any())
            has_cond = bool((~uncond_mask).any())
            if not has_uncond:
                return self.transformer(x, t, context, mask)
            if not has_cond:
                return self.transformer(x, t, context, mask, skip_layers=skip)
            cond_idx = (~uncond_mask).nonzero(as_tuple=True)[0]
            uncond_idx = uncond_mask.nonzero(as_tuple=True)[0]
            cond_out = self.transformer(x[cond_idx], t[cond_idx], context[cond_idx], mask[cond_idx])
            uncond_out = self.transformer(x[uncond_idx], t[uncond_idx], context[uncond_idx], mask[uncond_idx], skip_layers=skip)
            out = torch.empty((b,) + tuple(cond_out.shape[1:]), dtype=cond_out.dtype, device=cond_out.device)
            out.index_copy_(0, cond_idx, cond_out)
            out.index_copy_(0, uncond_idx, uncond_out)
            return out

        return self.transformer(x, t, context, mask)


class MossImage(ForgeDiffusionEngine):
    def __init__(self, bundle_path):
        class MossEngineConfig:
            huggingface_repo = 'MOSS-Image'
            is_moss = True
            latent_format = MossLatentFormat()
            unet_config = {'in_channels': MOSS_LATENT_CHANNELS}

            def inpaint_model(self):
                return False

        super().__init__(MossEngineConfig(), {})
        self.is_inpaint = False
        self.bundle_path = bundle_path

        unet = self._load_transformer(bundle_path)
        clip = self._load_text_encoder(bundle_path)
        vae = self._load_vae(bundle_path)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    @staticmethod
    def _load_sharded_state_dict(directory):
        from safetensors.torch import load_file

        index_path = os.path.join(directory, 'model.safetensors.index.json')
        shard_path = os.path.join(directory, 'diffusion_pytorch_model.safetensors')
        if os.path.isfile(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                weight_map = json.load(f)['weight_map']
            sd = {}
            for shard in sorted(set(weight_map.values())):
                sd.update(load_file(os.path.join(directory, shard), device='cpu'))
            return sd
        if os.path.isfile(shard_path):
            return load_file(shard_path, device='cpu')
        shards = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.safetensors')]
        assert len(shards) > 0, f'No safetensors found in {directory}'
        sd = {}
        for shard in shards:
            sd.update(load_file(shard, device='cpu'))
        return sd

    @staticmethod
    def _load_transformer(bundle_path):
        config = json.load(open(os.path.join(bundle_path, 'transformer', 'config.json'), 'r', encoding='utf-8'))
        config = {k: v for k, v in config.items() if not k.startswith('_')}

        supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        state_dict = MossImage._load_sharded_state_dict(os.path.join(bundle_path, 'transformer'))
        state_dict_parameters = memory_management.state_dict_parameters(state_dict)
        state_dict_dtype = memory_management.state_dict_dtype(state_dict)

        storage_dtype = memory_management.unet_dtype(model_params=state_dict_parameters, supported_dtypes=supported_dtypes)
        unet_storage_dtype_overwrite = dynamic_args.get('forge_unet_storage_dtype')
        if unet_storage_dtype_overwrite is not None:
            storage_dtype = unet_storage_dtype_overwrite
        elif state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, 'nf4', 'fp4', 'gguf']:
            print(f'MOSS-Image: Using detected transformer dtype: {state_dict_dtype}')
            storage_dtype = state_dict_dtype

        load_device = memory_management.get_torch_device()
        computation_dtype = memory_management.get_computation_dtype(load_device, parameters=state_dict_parameters, supported_dtypes=supported_dtypes)
        offload_device = memory_management.unet_offload_device()

        if storage_dtype in ['nf4', 'fp4', 'gguf']:
            initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
            with using_forge_operations(device=initial_device, dtype=computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                transformer = MOSSImageTransformer(**config)
        else:
            initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
            need_manual_cast = storage_dtype != computation_dtype
            to_args = dict(device=initial_device, dtype=storage_dtype)
            with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast):
                transformer = MOSSImageTransformer(**config).to(**to_args)

        # Custom RMSNorms are not forge operations; keeping their weights in the
        # storage dtype (e.g. fp8) would lose precision on load, so restore them
        # to the computation dtype before copying the checkpoint in.
        if storage_dtype not in ['nf4', 'fp4', 'gguf'] and storage_dtype != computation_dtype:
            for module in transformer.modules():
                if isinstance(module, _RMSNorm) and module.weight.dtype != computation_dtype:
                    module.weight.data = torch.nn.Parameter(torch.ones_like(module.weight, dtype=computation_dtype))

        load_state_dict(transformer, state_dict, log_name='MOSSImageTransformer')
        transformer.eval()

        wrapped = MossTransformerWrapper(transformer, depth=config['num_layers'])
        wrapped.storage_dtype = storage_dtype
        wrapped.computation_dtype = computation_dtype
        wrapped.load_device = load_device
        wrapped.initial_device = initial_device
        wrapped.offload_device = offload_device

        class MossModelConfig:
            is_moss = True
            huggingface_repo = 'MOSS-Image'

        return UnetPatcher.from_model(
            model=wrapped,
            diffusers_scheduler=None,
            k_predictor=PredictionMoss(),
            config=MossModelConfig(),
        )

    @staticmethod
    def _text_encoder_storage_dtype():
        from modules import shared
        choice = getattr(shared.opts, 'moss_text_encoder_precision', 'Automatic')
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
            'float8_e4m3fn': torch.float8_e4m3fn,
            'float8_e5m2': torch.float8_e5m2,
            'nf4': 'nf4',
            'fp4': 'fp4',
        }
        if choice in dtype_map:
            return dtype_map[choice]
        if memory_management.total_vram >= 49152:
            return torch.bfloat16
        if memory_management.total_vram >= 28672:
            return torch.float8_e4m3fn
        return 'nf4'

    @staticmethod
    def _load_text_encoder(bundle_path):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.initialization import no_init_weights

        te_path = os.path.join(bundle_path, 'text_encoder')
        config = AutoConfig.from_pretrained(te_path)
        storage_dtype = MossImage._text_encoder_storage_dtype()
        computation_dtype = memory_management.text_encoder_dtype()
        print(f'MOSS-Image: Loading text encoder with storage dtype: {storage_dtype}')

        if storage_dtype in ['nf4', 'fp4', 'gguf']:
            # bnb4 forge ops compute in bf16; create all non-quantized weights in bf16 too
            with no_init_weights():
                with using_forge_operations(device=memory_management.cpu, dtype=torch.bfloat16, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                    model = AutoModel.from_config(config)
        else:
            with no_init_weights():
                with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                    model = AutoModel.from_config(config)

        state_dict = MossImage._load_sharded_state_dict(te_path)
        load_state_dict(model, state_dict, log_name='MOSSImageTextEncoder', ignore_errors=['lm_head.weight'])
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(bundle_path, 'tokenizer'))

        return CLIP(model_dict={'moss': model}, tokenizer_dict={'moss': tokenizer})

    @staticmethod
    def _load_vae(bundle_path):
        from diffusers import AutoencoderKLFlux2

        vae_dir = os.path.join(bundle_path, 'vae')
        vae_dtype = memory_management.vae_dtype()
        vae = AutoencoderKLFlux2.from_pretrained(vae_dir, torch_dtype=vae_dtype).eval()

        adapter = MossFlux2VAEAdapter(vae)
        vae_patch = VAE(model=adapter)
        # The FLUX.2 AE is 32-channel at f8, but MOSS packs latents 2x2 into
        # 128 channels, giving an effective f16 latent scale.
        vae_patch.downscale_ratio = MOSS_VAE_SCALE
        vae_patch.latent_channels = MOSS_LATENT_CHANNELS
        return vae_patch

    def set_clip_skip(self, clip_skip):
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        # Quantizing/loading the text encoder needs a large transient; demand
        # enough memory that other resident models (the DiT) get evicted first.
        memory_management.load_models_gpu([self.forge_objects.clip.patcher], memory_required=int(8 * 1024 ** 3))

        model = self.forge_objects.clip.cond_stage_model.moss
        tokenizer = self.forge_objects.clip.tokenizer.moss
        device = next(model.parameters()).device

        rows = []
        # Unpadded individual forwards also avoid padding effects in Qwen GDN.
        for value in prompt:
            formatted = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': value}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True)
            inputs = tokenizer(formatted, return_tensors='pt', truncation=True,
                               max_length=MAX_SEQUENCE_LENGTH).to(device)
            output = model(**inputs, use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])

        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
        for index, row in enumerate(rows):
            embeds[index, :row.shape[0]] = row
            mask[index, :row.shape[0]] = True

        return {'crossattn': embeds, 'attention_mask': mask}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        tokenizer = self.forge_objects.clip.tokenizer.moss
        token_count = len(tokenizer(prompt)['input_ids'])
        return token_count, MAX_SEQUENCE_LENGTH

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        decoded = self.forge_objects.vae.decode(x)
        return decoded.movedim(-1, 1) * 2.0 - 1.0
