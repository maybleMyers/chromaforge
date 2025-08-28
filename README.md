![GUI Screenshot](images/screenshot.png)

## Fork of forge to use Chroma!

This is a fork with the patch from https://github.com/croquelois/forgeChroma preinstalled thanks Thanks to [@croquelois](https://github.com/croquelois) && [@lllyasviel](https://github.com/lllyasviel)!   
I am going to keep updating it with new features. Suggestions are welcome.  

To use:  
git clone https://github.com/maybleMyers/chromaforge  
Download one of the t5xxl text encoders from https://huggingface.co/silveroxides/t5xxl_flan_enc and place in models/text_encoder.  
Download the vae from https://huggingface.co/lodestones/Chroma/tree/main ae.safetenstors and put it in models/vae.  
Download a checkpoint from https://huggingface.co/lodestones/Chroma/tree/main ie chroma-unlocked-v29.5.safetensors and place in models/Stable-diffusion.  

You will need python3.10 installed.  
Run webui-user.bat for windows installation.  Run webui.sh for linux installation (./webui.sh) from the chromaforge root directory.  
After you first run webui-user.bat it might error, if so, close it and re run it.  
To update to the latest version navigate to your root directory in a terminal and type "git pull"  

in Forge, on the top left select all and not flux  
select the checkpoint and then in the next field select the text encoder and vae.  
Use euler simple scheduler.  
set the distilled config scale to 1, and the normal config scale to something like 3.5-7  
use a very long positive prompt and a very long negative prompt.  
forge doesn't seem to work with all quantized model, Q4_K_S fail, but Q4_1 work  
refer to screenshot for working settings.  

To update to torch 2.7.0 with cuda 12.8 on windows and install sage attention, navigate to your root directory after initial installation ie c:/chromaforge/ and run these commands:  
venv/scripts/activate  
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128  
pip install -U "triton-windows<3.4"  
pip install .\sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl  

there's a bunch of extra samplers/schedulers at these places:  
https://github.com/DenOfEquity/webUI_ExtraSchedulers
https://github.com/MisterChief95/sd-forge-extra-samplers

Here are experimental high resolution checkpoints for chroma:  
https://huggingface.co/lodestones/chroma-debug-development-only/tree/main/staging_large_3  
https://huggingface.co/lodestones/chroma-debug-development-only/tree/main/staging_large_4  
They can be converted to safetensors with convertpth.py  
for example modify the source.pth and output.safetensors on the last line in the script: convert_pth_to_safetensors("source.pth", "output.safetensors") to what you want them to be, then run the script with python convertpth.py.  

## Radiance Model

Download a checkpoint from here https://huggingface.co/lodestones/chroma-debug-development-only/tree/main/radiance then convert it with convertpth.py. Put it in your stable diffusion subfolder. Use the radiance model in the chroma tab of the gui without a vae. It will be automatically detected. Use the same T5 you normally use.  

Training: https://github.com/lodestone-rock/flow/  https://github.com/kohya-ss/sd-scripts/tree/sd3  
Donate to Lodestone (training is bookoo expensive and crowdfunded): https://ko-fi.com/lodestonerock  
Discord: http://discord.gg/SQVcWVbqKx  

## Changlog  
8/28/2025  
    Added initial support for the radiance model class.  
7/13/2025  
    Increase max res for chroma to 3072, add dedicated chroma button.  
6/11/2025  
    Add links, Sigmoid Offset scheduler (thanks to croq and silveroxides https://github.com/silveroxides/ComfyUI_SigmoidOffsetScheduler), fix euler a simple sampler via croq's PR in forge https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2915.  
5/27/2025  
    Fixed the sage attention implementation to work with chroma.  
5/25/2025  
    Add support for sage and flash attention from this pr: https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2881  from @spawner1145  
    use the methods by adding --use-sage-attention or --use-flash-attention  ... upon testing by a few people does not seem to have an increase on speed at all.  
5/18/2025  
    Update to latest version of forge. Fix broken queuing.  