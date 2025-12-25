# what is this?
An image generation model using a modified pruned Flux.1 architecture, from 12B down to 8.9B, de-distilled and finetuned from Flux.1 Schnell.
# Some background about the pre-training.
The pre-trained versions of Chroma, known as chroma-unlocked/chroma-unlocked-detail-calibrated, was done using multiple training nodes that were merged together to produce the final model.
These staging nodes are known as base, fast and large, the latter being an off shoot trained at 1024x. The primary pretraining consists of base and fast while the detail-calibrated one consists of base, fast and large.
# Software
### Generation/Inference
- [ComfyUI](https://www.comfy.org/) is the only officially supported software for inferencing Chroma. A basic workflow can be found in the templates.
- [SD webui Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) also supports Chroma, with the [chromaforge](https://github.com/maybleMyers/chromaforge) fork to support more features and Radiance.
- [SD.Next](https://github.com/vladmandic/sdnext) supports Chroma as well.
### Training
- [OneTrainer](https://github.com/Nerogar/OneTrainer) is the recommended trainer, it has a powerful GUI and lots of settings.
- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) is the trainer that has had Chroma support the longest.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) supports Chroma using the `sd3` branch.
- [AI-toolkit](https://github.com/ostris/ai-toolkit) is not recommended due to having potential issues and lacking optimization. 
# Final model versions
Here's a handy chart to show what the chroma version tree looks like:
![[dsaffsda-1.png]]
- Chroma1-Base is the 512x raw pretrained checkpoint, this one is recommended if you're doing a long full fine-tune. It is also known as chroma-unlocked-v48 in the old repository.
- Chroma1-HD is a mixed res (512, 768, 1024, 1152x) finetune of chroma-unlocked-v48. It is the primary on recommended for inference.
- chroma-unlocked-v48-detail-calibrated is the final pre-trained release of the detail-calibrated line up. It may be better at text than Chroma1-HD.
- 2k-test is an experimental finetune of chroma-unlocked-v48 using 2048x images, it works best at high resolutions but the results from the experimental finetune full checkpoint is inconsistent.
- DC-2k is a merge of v48-detail-calibrated and 2k-test. It is recommended for generating at >1024x resoltions.
# Additional model versions
- Chroma1-Radiance is still being trained and releases are in alpha testing. This is an experimental VAE-less PixelNerd based modification of chroma's architecture, it generates directly in pixel space rather than latent space, is currently trained at 512x.
- chroma-unlocked-v49/chroma-unlocked-v50 are borked (overfit on 1024x), not recommended, may give good results at 1152x.
- Chroma1-Flash is a fast checkpoint finetuned from chroma-unlocked-v48 at 512x. It can produce complete images at 8 steps using the samplers heun, dpmpp_sde/dpmpp_sde_gpu and dpmpp_2s_ancestral and at 16 steps with most other samplers.
- chroma-unlocked-v47-flash-heun is a low-step finetune of chroma-unlocked-v47, originally meant to be used with heun sampler at 8 steps with cfg set to 1(disables cfg and negative prompt), recommended due to being the best most stable version of flash.
# official LoRas
- chroma-unlocked-v47-flash-heun has also been converted to LoRA format spanning a large array of ranks(how much influence it has at inference) which might be the preferencial way to use for those who do not want the full strength of the full model or use with the final released Chroma models.
- It is currently the best option for flash lora, it is used to turn any checkpoint into a flash one, the higher the rank and weight the stronger its effect towards low step generation and low cfg. Rank 32 is the lower rank recommended for using with cfg set to one. Ranks 64/72/80/88/96 is generally recommended though. Rank 1 can be used to stabilize the model without introducing much bias.
# quantization
Chroma is sensitive to quantization, therefore there are two main recommended methods
1. GGUF Q8 - nearly indistinguishable from bf16, low likelihood of artifacts, inference speed is considerably affected by LoRas.
2. [Clybius' learned fp8 scaled,](https://github.com/Clybius/Learned-Rounding/) slightly worse quality than GGUF Q8 with a slightly smaller filesize and fastest inference speed, not affected by LoRas as much.
For Radiance, due to certain layers and the ComfyUI implementation, a good quality fp8 quant is currently infeasible, GGUF Q8 is preferable.
# text encoders
Chroma was trained with an fp8 t5xxl 1.1.
The recommended encoders are t5xxl v1.1 of flan t5, either at fp16/bf16, GGUF Q8, or fp8 scaled learned.
# prompting
Chroma's dataset was captioned using Gemini 1.5 flash, so Gemini is the recommended tool for captioning.
The following system prompt for Gemini produces good captions:
https://discord.com/channels/1101419796744114209/1319442661572345909/1390241273880907858
The max length of any prompt(without using concatenation) is 512 tokens, can be checked using the [sd tokenizer online tool](https://sd-tokenizer.rocker.boo/), set to Pixart.
T5 padding is always meant to equal 1, due to how the model was trained.
Using a short negative prompt (<70 tokens) is not recommended in any case, unless it is not used.
Tag-based prompting as well as shorter prompts in general work best with Flash models. 
Tags can be separated with commas, however that influences the style into cartoon/anime, so for realistic images they should be separated using periods.
General prompt format is: natural language sentences + period separated tags at the end.
### special tags and phrases
`aesthetic 0-10` are general aesthetic score tags.
`aesthetic 11` is for aesthetically curated AI generated images, may cause prompt bleeding.
`masterpiece; best quality; amazing quality; worst quality; low quality` - are the typical quality tags, may or may not work, other similar tags may be used.
`The photograph is a . `(x10) due to a Gemini quirk, repeating this sentence a few times at the end of the prompt may improve photorealism.
`A casual snapshot of ...` is an easy way to prompt for an amateur quality/phone photo style.
`...cosplayer dressed as XYZ character...` is a way to prompt for realistic images of fictional characters.
### artist styles and characters by name
Due to a captioning mistake, Chroma ended up not learning some obscure characters and most artist styles. It is recommended to caption every feature of a character.
### lora caveats
T5 is not great at learning out of context trigger words/tags, so it's recommended to add their domain, e.g. artist:artist name or character:character name.
Typically it is best to describe everything that can be seen in an image, for characters you can give them a non-short name.
# sampling
Ancestral samplers may sometimes improve results, feel free to use them or not.
Flash models produce better results with higher order (s)ubstep samplers.
### built-in:
- `euler` - the most basic sampler and also the fastest per step, works fine but there are better options.
- `res_multistep` - nearly always better than euler at a similar speed.
- `dpmpp_2m` - decent alternative to res multistep.
- `gradient_estimation` - trades a bit of aesthetics for better coherency.
- `heun; deis` - recommended for flash models.
### [RES4LYF:](https://github.com/ClownsharkBatwing/RES4LYF)
- `res_2m(_ode)` - potentially better than res multistep.
- `res_2s(_ode)` - potentially better alternative to Heun.
# scheduling
The default settings for chroma are: 26 steps, shift = 1, beta scheduler at 0.6 0.6.
The typical settings for Flash models are: 8 steps at cfg 1-1.5 using a second order substep sampler, or 16 steps using a multistep sampler.
### timestep shifting
Chroma had been trained without timestep shifting.
It is either recommended to use a shift = 1 or flux shift, although any custom value can be used, shift < 1 is not recommended.
Empirically, flux shift produces good results, but further testing may be needed.
### schedulers
- `beta` is recommended, if using shift = 1 then settings of 0.4 0.4 can improve results
- `sigmoid_offset` is a customizable scheduler made specifically for use with chroma, with shift = 1
- `bong_tangent` from RES4LYF is great, although ignores all shifting and is not customisable.
