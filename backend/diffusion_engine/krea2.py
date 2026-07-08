import json
import os

import torch
from einops import rearrange

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionFlux
from backend.nn.krea2 import (
    KREA2_CONFIG,
    Krea2QwenVAE,
    SingleStreamDiT,
    convert_diffusers_krea2_state_dict,
    is_diffusers_krea2_state_dict,
    prepare_image_tokens,
)
from backend.operations import using_forge_operations
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.state_dict import load_state_dict

local_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'huggingface', 'Krea2')

# Prompt template used to condition the Qwen3-VL encoder, from the reference
# implementation (encoder.py). The first PROMPT_PREFIX_INDEX tokens (the system
# prompt) are cut from the hidden states after encoding.
PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_PREFIX_INDEX = 34
PROMPT_SUFFIX_START_INDEX = 5
PROMPT_MAX_LENGTH = 512

# Text-encoder hidden-state layers stacked as conditioning (model_index.json).
TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class Krea2LatentFormat:
    """Krea 2 samples in the normalized Qwen-Image (Wan 2.1 f8) latent space; the
    normalization itself happens inside Krea2QwenVAE. These RGB factors are only
    used for cheap live-preview approximation."""

    latent_rgb_factors = [
        [-0.1299, -0.1692, 0.2932],
        [0.0671, 0.0406, 0.0442],
        [0.3568, 0.2548, 0.1747],
        [0.0372, 0.2344, 0.1420],
        [0.0313, 0.0189, -0.0328],
        [0.0296, -0.0956, -0.0665],
        [-0.3477, -0.4059, -0.2925],
        [0.0166, 0.1902, 0.1975],
        [-0.0412, 0.0267, -0.1364],
        [-0.1293, 0.0740, 0.1636],
        [0.0680, 0.3019, 0.1128],
        [0.0032, 0.0581, 0.0639],
        [-0.1251, 0.0927, 0.1699],
        [0.0060, -0.0633, 0.0005],
        [0.3477, 0.2275, 0.2950],
        [0.1984, 0.0913, 0.1861],
    ]

    def __init__(self):
        self.scale_factor = 1.0
        self.shift_factor = 0.0

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class Krea2TransformerWrapper(torch.nn.Module):
    """Adapts Forge's (x, timestep, context) UNet interface to the Krea 2 MMDiT.

    Forge passes sigma in [1, 0] which is exactly the Krea 2 flow time, and the
    model predicts the flow velocity v = noise - x0, matching PredictionFlux's
    denoised = x - sigma * v, so both timestep and output pass through unchanged.
    """

    def __init__(self, mmdit):
        super().__init__()
        self.mmdit = mmdit

    def forward(self, x, timestep, context=None, transformer_options=None, **kwargs):
        b, c, h, w = x.shape
        patch = self.mmdit.config.patch

        txtmask = None
        if transformer_options is not None:
            txtmask = transformer_options.get('attention_mask', None)
        if txtmask is None:
            txtmask = torch.ones(context.shape[:2], device=x.device, dtype=torch.bool)
        else:
            txtmask = txtmask.to(device=x.device, dtype=torch.bool)

        img_tokens, pos, mask = prepare_image_tokens(x, context.shape[1], patch, txtmask)

        t = timestep.reshape(-1)
        if t.shape[0] != b:
            t = t.expand(b)
        t = t.to(dtype=x.dtype, device=x.device)

        output = self.mmdit(img=img_tokens, context=context, t=t, pos=pos, mask=mask)

        return rearrange(
            output,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            ph=patch,
            pw=patch,
            h=h // patch,
            w=w // patch,
        )


class Krea2(ForgeDiffusionEngine):
    def __init__(self, state_dicts):
        class Krea2EngineConfig:
            huggingface_repo = 'Krea2'
            is_krea2 = True
            latent_format = Krea2LatentFormat()
            unet_config = {'in_channels': KREA2_CONFIG.channels}

            def inpaint_model(self):
                return False

        super().__init__(Krea2EngineConfig(), {})
        self.is_inpaint = False

        unet = self._load_transformer(state_dicts['transformer'])
        clip = self._load_text_encoder(state_dicts.get('text_encoder', None))
        vae = self._load_vae(state_dicts.get('vae', None))

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    @staticmethod
    def _load_transformer(state_dict):
        # ComfyUI-saved checkpoints bundle the transformer under a prefix.
        for prefix in ['model.diffusion_model.', 'diffusion_model.']:
            if any(k.startswith(prefix) for k in state_dict):
                print(f'Krea2: Stripping "{prefix}" prefix from transformer state dict')
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                break

        if is_diffusers_krea2_state_dict(state_dict):
            print('Krea2: Converting Diffusers-format transformer state dict to reference format')
            state_dict = convert_diffusers_krea2_state_dict(state_dict)

        supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        state_dict_parameters = memory_management.state_dict_parameters(state_dict)
        state_dict_dtype = memory_management.state_dict_dtype(state_dict)

        storage_dtype = memory_management.unet_dtype(model_params=state_dict_parameters, supported_dtypes=supported_dtypes)
        unet_storage_dtype_overwrite = dynamic_args.get('forge_unet_storage_dtype')
        if unet_storage_dtype_overwrite is not None:
            storage_dtype = unet_storage_dtype_overwrite
        elif state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, 'nf4', 'fp4', 'gguf']:
            print(f'Krea2: Using detected transformer dtype: {state_dict_dtype}')
            storage_dtype = state_dict_dtype

        load_device = memory_management.get_torch_device()
        computation_dtype = memory_management.get_computation_dtype(load_device, parameters=state_dict_parameters, supported_dtypes=supported_dtypes)
        offload_device = memory_management.unet_offload_device()

        if storage_dtype in ['nf4', 'fp4', 'gguf']:
            initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
            with using_forge_operations(device=initial_device, dtype=computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                mmdit = SingleStreamDiT(KREA2_CONFIG)
        else:
            initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
            need_manual_cast = storage_dtype != computation_dtype
            to_args = dict(device=initial_device, dtype=storage_dtype)
            with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast):
                mmdit = SingleStreamDiT(KREA2_CONFIG).to(**to_args)

        load_state_dict(mmdit, state_dict, log_name='Krea2Transformer')

        # RMSNorm scales and modulation tables are tiny but precision-sensitive
        # (used as float32 inside the model) — keep them out of FP8 storage.
        if storage_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            fixed = 0
            for name, p in mmdit.named_parameters():
                if p.dtype == storage_dtype and (name.endswith('.scale') or name.endswith('.lin')):
                    p.data = p.data.to(computation_dtype)
                    fixed += 1
            if fixed:
                print(f'Krea2: Kept {fixed} norm/modulation parameters in {computation_dtype}')

        wrapped = Krea2TransformerWrapper(mmdit)
        wrapped.storage_dtype = storage_dtype
        wrapped.computation_dtype = computation_dtype
        wrapped.load_device = load_device
        wrapped.initial_device = initial_device
        wrapped.offload_device = offload_device

        # Turbo runs with a constant mu=1.15; sampling_prepare re-applies the shift
        # per generation (manual override or resolution-derived for the Raw model).
        k_predictor = PredictionFlux(shift_override=1.15)

        class Krea2ModelConfig:
            is_krea2 = True
            huggingface_repo = 'Krea2'

        return UnetPatcher.from_model(
            model=wrapped,
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=Krea2ModelConfig(),
        )

    @staticmethod
    def _load_text_encoder(state_dict):
        assert state_dict is not None, (
            'Krea2 needs the Qwen3-VL text encoder! '
            'Select the repackaged text encoder file (see krea2_repackage.py) in the "VAE / Text Encoder" dropdown.'
        )

        # Normalize Qwen3VLForConditionalGeneration-format keys ("model." prefix, lm_head).
        if any(k.startswith('model.') for k in state_dict):
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}

        from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel, modeling_utils

        text_encoder_dtype = memory_management.text_encoder_dtype()
        config = AutoConfig.from_pretrained(os.path.join(local_config_path, 'text_encoder'))

        with modeling_utils.no_init_weights():
            with using_forge_operations(device=memory_management.cpu, dtype=text_encoder_dtype, manual_cast_enabled=True):
                model = Qwen3VLModel(config)

        load_state_dict(model, state_dict, log_name='Krea2TextEncoder', ignore_errors=['lm_head.weight'])

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_config_path, 'tokenizer'))

        return CLIP(model_dict={'qwen': model}, tokenizer_dict={'qwen': tokenizer})

    @staticmethod
    def _load_vae(state_dict):
        assert state_dict is not None, (
            'Krea2 needs the Qwen-Image VAE! '
            'Select the repackaged VAE file (see krea2_repackage.py) in the "VAE / Text Encoder" dropdown.'
        )

        with open(os.path.join(local_config_path, 'vae', 'config.json'), 'rt', encoding='utf-8') as f:
            config = json.load(f)

        vae_dtype = memory_management.vae_dtype()
        with using_forge_operations(device=memory_management.cpu, dtype=vae_dtype, manual_cast_enabled=True):
            model = Krea2QwenVAE(config)

        load_state_dict(model.ae, state_dict, log_name='Krea2VAE')

        return VAE(model=model)

    def set_clip_skip(self, clip_skip):
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        qwen = self.forge_objects.clip.cond_stage_model.qwen
        tokenizer = self.forge_objects.clip.tokenizer.qwen
        device = qwen.language_model.embed_tokens.weight.device

        texts = [PROMPT_PREFIX + p for p in prompt]
        inputs = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=PROMPT_MAX_LENGTH + PROMPT_PREFIX_INDEX - PROMPT_SUFFIX_START_INDEX,
            return_tensors='pt',
        )
        suffix_inputs = tokenizer([PROMPT_SUFFIX] * len(texts), return_tensors='pt')

        input_ids = torch.cat([inputs['input_ids'], suffix_inputs['input_ids']], dim=1).to(device)
        mask = torch.cat(
            [inputs['attention_mask'].bool(), suffix_inputs['attention_mask'].bool()], dim=1
        ).to(device)

        outputs = qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

        hiddens = torch.stack(
            [outputs.hidden_states[i] for i in TEXT_ENCODER_SELECT_LAYERS], dim=2
        )
        hiddens = hiddens[:, PROMPT_PREFIX_INDEX:]
        mask = mask[:, PROMPT_PREFIX_INDEX:]

        return {'crossattn': hiddens, 'attention_mask': mask}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        tokenizer = self.forge_objects.clip.tokenizer.qwen
        token_count = len(tokenizer(prompt)['input_ids'])
        return token_count, max(PROMPT_MAX_LENGTH, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        # x is (B, 3, H, W) in [-1, 1]; VAE.encode expects channels-last in [0, 1].
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        # VAE.decode returns channels-last in [0, 1]; webui expects (B, 3, H, W) in [-1, 1].
        decoded = self.forge_objects.vae.decode(x)
        result = decoded.movedim(-1, 1) * 2.0 - 1.0
        return result.to(x)
