## Fork of forge to use Chroma!

This is a fork with the patch from https://github.com/croquelois/forgeChroma preinstalled thanks @croquelois && @lllyasviel !  
I am going to keep updating it with new features. Suggestions are welcome.  

To use:  
Download one of the t5xxl text encoders from https://huggingface.co/silveroxides/t5xxl_flan_enc and place in models/text_encoder.  
Download the vae from https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main/vae and put it in models/vae.  
Download a checkpoint from https://huggingface.co/lodestones/Chroma/tree/main ie chroma-unlocked-v29.5.safetensors and place in models/Stable-diffusion.  

run webui-user.bat or install manually for linux.  

in Forge, on the top left select all and not flux  
select the checkpoint and then in the next field select the text encoder and vae.  
Use euler simple scheduler.  
set the distilled config scale to 1, and the normal config scale to something like 3.5  
use a very long positive prompt and a very long negative prompt.  
forge doesn't seem to work with all quantized model, Q4_K_S fail, but Q4_1 work  

## Changlog  

5/18/2025  
    Update to latest version of forge. Fix broken queuing.  