import json
from contextlib import closing

import modules.scripts
from modules import processing, infotext_utils
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
from PIL import Image
import gradio as gr
from modules_forge import main_thread


def txt2img_create_processing(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_additional_modules: list, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, hr_cfg: float, hr_distilled_cfg: float, override_settings_texts, *args, force_enable_hr=False):
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    p = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_additional_modules=hr_additional_modules,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == 'Use same scheduler' else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        hr_cfg=hr_cfg,
        hr_distilled_cfg=hr_distilled_cfg,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    p.user = request.username if request else None

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)
    
    return p


def txt2img_function_wrapper(id_task_gradio_progress: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_additional_modules: list, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt_str, hr_cfg: float, hr_distilled_cfg_float, override_settings_texts, *script_args):
    p = txt2img_create_processing(
        id_task_gradio_progress, request, prompt, negative_prompt, prompt_styles,
        n_iter, batch_size, cfg_scale, distilled_cfg_scale, height, width, enable_hr,
        denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps,
        hr_resize_x, hr_resize_y, hr_checkpoint_name, hr_additional_modules,
        hr_sampler_name, hr_scheduler, hr_prompt, hr_negative_prompt_str, hr_cfg, hr_distilled_cfg_float,
        override_settings_texts, *script_args
    )

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)
        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")

def txt2img_upscale_function_wrapper(id_task_gradio_progress: str, request: gr.Request, gallery, gallery_index, generation_info, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_additional_modules: list, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt_str, hr_cfg: float, hr_distilled_cfg_float, override_settings_texts, *script_args):
    assert len(gallery) > 0, 'No image to upscale'

    if gallery_index < 0 or gallery_index >= len(gallery):
        error_html = plaintext_to_html(f'Bad image index: {gallery_index}', classname="error")
        return gallery, generation_info, error_html, ""
    
    p = txt2img_create_processing(
        id_task_gradio_progress, request, prompt, negative_prompt, prompt_styles,
        n_iter, batch_size, cfg_scale, distilled_cfg_scale, height, width, True, 
        denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps,
        hr_resize_x, hr_resize_y, hr_checkpoint_name, hr_additional_modules,
        hr_sampler_name, hr_scheduler, hr_prompt, hr_negative_prompt_str, hr_cfg, hr_distilled_cfg_float,
        override_settings_texts, *script_args
    )
    p.batch_size = 1
    p.n_iter = 1
    p.txt2img_upscale = True

    geninfo = json.loads(generation_info)
    image_info = gallery[gallery_index]
    
    image_path_or_url = None
    if isinstance(image_info, dict) and "name" in image_info:
        image_path_or_url = image_info["name"]
    elif isinstance(image_info, list) and image_info and isinstance(image_info[0], dict) and "name" in image_info[0]:
        image_path_or_url = image_info[0]["name"]
    else:
        image_path_or_url = image_info

    p.firstpass_image = infotext_utils.image_from_url_text(image_path_or_url)

    infotexts_array = geninfo.get('infotexts', [])
    if not infotexts_array or gallery_index >= len(infotexts_array):
         error_html = plaintext_to_html(f'Infotext not found for image index: {gallery_index}', classname="error")
         return gallery, generation_info, error_html, ""

    parameters = parse_generation_parameters(infotexts_array[gallery_index], [])
    p.seed = parameters.get('Seed', -1)
    p.subseed = parameters.get('Variation seed', -1)

    p.width = p.firstpass_image.width
    p.height = p.firstpass_image.height
    p.extra_generation_params['Original Size'] = f'{width}x{height}'

    p.override_settings['save_images_before_highres_fix'] = False

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)
        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    insert = getattr(shared.opts, 'hires_button_gallery_insert', False)
    new_gallery_output = []
    
    original_gallery_filepaths = [g_item[0].get("name") if isinstance(g_item, list) and g_item and isinstance(g_item[0], dict) else g_item.get("name") if isinstance(g_item, dict) else g_item for g_item in gallery]

    for i, img_path_or_pil in enumerate(original_gallery_filepaths):
        if insert or i != gallery_index:
            current_gallery_item = gallery[i]
            if isinstance(current_gallery_item, str):
                 new_gallery_output.append(current_gallery_item)
            elif isinstance(current_gallery_item, list) and current_gallery_item and isinstance(current_gallery_item[0], dict) and "name" in current_gallery_item[0]:
                 new_gallery_output.append(current_gallery_item)
            elif isinstance(current_gallery_item, dict) and "name" in current_gallery_item:
                 new_gallery_output.append(current_gallery_item)
            else: 
                if hasattr(img_path_or_pil, 'filename'):
                    dummy_pil = Image.new("RGB", (1,1)) 
                    dummy_pil.already_saved_as = img_path_or_pil.filename.rsplit('?',1)[0]
                    new_gallery_output.append(dummy_pil)
                else: 
                    new_gallery_output.append(img_path_or_pil)

        if i == gallery_index:
            new_gallery_output.extend(processed.images) 
    
    new_geninfo_infotexts = list(geninfo["infotexts"])
    new_active_index = gallery_index
    if insert:
        new_active_index += 1
        new_geninfo_infotexts.insert(new_active_index, processed.info)
    else:
        if gallery_index < len(new_geninfo_infotexts):
             new_geninfo_infotexts[gallery_index] = processed.info
        else:
             new_geninfo_infotexts.append(processed.info)

    geninfo["infotexts"] = new_geninfo_infotexts
    
    return new_gallery_output, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img(id_task: str, request: gr.Request, *args):
    return main_thread.run_and_wait_result(txt2img_function_wrapper, 
                                           id_task, 
                                           request, 
                                           *args, 
                                           gradio_progress_id=id_task)


def txt2img_upscale(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    return main_thread.run_and_wait_result(txt2img_upscale_function_wrapper,
                                           id_task, 
                                           request, gallery, gallery_index, generation_info, 
                                           *args, 
                                           gradio_progress_id=id_task)