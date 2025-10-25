import torch
import os
from backend.utils import read_arbitrary_config

class ModelList:
    pass

model_list = ModelList()

class DiffusersConvert:
    def convert_vae_state_dict(self, sd):
        new_sd = {}
        for k, v in sd.items():
            if 'encoder.down_blocks' in k: k = k.replace('down_blocks', 'down')
            if 'encoder.up_blocks' in k: k = k.replace('up_blocks', 'up')
            if 'decoder.down_blocks' in k: k = k.replace('down_blocks', 'down')
            if 'decoder.up_blocks' in k: k = k.replace('up_blocks', 'up')
            new_sd[k] = v
        return new_sd

diffusers_convert = DiffusersConvert()

class GuessBase:
    def __init__(self):
        self.unet_config = self.get_unet_config()
        self.ztsnr = False
        self.supported_inference_dtypes = ['fp32', 'fp16', 'bf16']

    def get_unet_config(self):
        config_directory_path = os.path.join(os.path.dirname(__file__), self.huggingface_repo, self.unet_name)
        config = read_arbitrary_config(config_directory_path)
        if hasattr(self, 'unet_extra_config'): config.update(self.unet_extra_config)
        if hasattr(self, 'unet_remove_config'):
            for x in self.unet_remove_config:
                if x in config: del config[x]
        return config

    def process_vae_state_dict(self, sd): return sd
    def process_clip_state_dict(self, sd): return sd
    def model_type(self, state_dict={}): return 'epsilon'
    def inpaint_model(self): return False

def add_supported_inference_dtypes(cls): pass

class GuessAuraFlow(GuessBase):
    huggingface_repo = 'AuraFlow'
    unet_name = 'transformer'
    unet_target = 'transformer'
    unet_key_prefix = []
    unet_remove_config = []
    unet_extra_config = {"in_channels": 4}
    vae_key_prefix = ['first_stage_model.', 'vae.']
    text_encoder_key_prefix = ['pile_t5xl.transformer.']
    vae_target = 'vae'
    scheduler_name = 'FlowMatchEulerDiscreteScheduler'
    model_class = None
    is_xl = True
    supported_inference_dtypes = ['fp16', 'bf16']

    def unet_instance(self, state_dict={}):
        if any(k.startswith("model.double_layers") for k in state_dict.keys()): return self
        return None

    def clip_target(self, state_dict={}):
        result = {}
        pref = self.text_encoder_key_prefix[0]
        if "{}encoder.final_layer_norm.weight".format(pref) in state_dict:
            result['pile_t5xl'] = 'text_encoder'
        return result

model_list.AuraFlow = GuessAuraFlow()
add_supported_inference_dtypes(model_list.AuraFlow)

def guess(sd):
    for name in dir(model_list):
        if name.startswith('__'): continue
        guesser_instance = getattr(model_list, name)
        if hasattr(guesser_instance, 'unet_instance') and callable(guesser_instance.unet_instance):
            result = guesser_instance.unet_instance(sd)
            if result is not None:
                print(f"[huggingface_guess] Model recognized as {name}.")
                return result
    print("[huggingface_guess] Model not recognized.")
    return None