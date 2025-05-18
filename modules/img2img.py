import os
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
import gradio as gr

from modules import images
from modules.infotext_utils import create_override_settings_dict
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
from modules.sd_models import get_closet_checkpoint_match
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.scripts
from modules_forge import main_thread


def process_batch(p, input_source, output_dir, inpaint_mask_dir, script_args_tuple, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    output_dir = output_dir.strip()
    processing.fix_seed(p)

    if isinstance(input_source, str):
        batch_images_paths = list(shared.walk_files(input_source, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".avif")))
    else:
        batch_images_paths = [os.path.abspath(x.name) for x in input_source]

    is_inpaint_batch = False
    inpaint_masks_paths = []
    if inpaint_mask_dir:
        inpaint_masks_paths = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks_paths)
        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks_paths)} masks found.")

    print(f"Will process {len(batch_images_paths)} images, creating {p.n_iter * p.batch_size} new images for each.")
    state.job_count = len(batch_images_paths) * p.n_iter

    original_prompt = p.prompt
    original_negative_prompt = p.negative_prompt
    original_seed = p.seed
    
    all_processed_results = None

    for i, image_file_path in enumerate(batch_images_paths):
        state.job = f"{i+1} out of {len(batch_images_paths)}"
        if state.skipped: state.skipped = False
        if state.interrupted or state.stopping_generation: break

        try:
            img = images.read(image_file_path)
        except UnidentifiedImageError as e:
            print(f"Could not read image {image_file_path}: {e}")
            continue
        img = ImageOps.exif_transpose(img)

        p.init_images = [img] * p.batch_size

        current_image_path_obj = Path(image_file_path)
        if is_inpaint_batch:
            corresponding_mask_path = None
            if len(inpaint_masks_paths) == 1:
                corresponding_mask_path = inpaint_masks_paths[0]
            else:
                mask_dir_path = Path(inpaint_mask_dir)
                found_masks = list(mask_dir_path.glob(f"{current_image_path_obj.stem}.*"))
                if not found_masks:
                    print(f"Warning: No mask found for {current_image_path_obj}. Skipping.")
                    continue
                corresponding_mask_path = found_masks[0]
            
            p.image_mask = images.read(corresponding_mask_path)

        if use_png_info:
            p.prompt = original_prompt
            p.negative_prompt = original_negative_prompt
            p.seed = original_seed
            # Placeholder for PNG info logic from original
            from modules.infotext_utils import parse_generation_parameters # Local import
            try:
                info_img_to_read = img
                if png_info_dir:
                    info_img_path_to_read = os.path.join(png_info_dir, os.path.basename(image_file_path))
                    info_img_to_read = images.read(info_img_path_to_read)
                geninfo_text, _ = images.read_info_from_image(info_img_to_read)
                parsed_png_parameters = parse_generation_parameters(geninfo_text)
                
                if "Prompt" in parsed_png_parameters and "Prompt" in (png_info_props or {}):
                    p.prompt += (" " + parsed_png_parameters["Prompt"])
                if "Negative prompt" in parsed_png_parameters and "Negative prompt" in (png_info_props or {}):
                    p.negative_prompt += (" " + parsed_png_parameters["Negative prompt"])
                if "Seed" in parsed_png_parameters and "Seed" in (png_info_props or {}):
                    p.seed = int(parsed_png_parameters["Seed"])
                # ... Add other PNG parameters as needed ...
            except Exception as e:
                print(f"Error reading PNG info for {image_file_path}: {e}")


        if output_dir:
            p.outpath_samples = output_dir
            p.override_settings['save_to_dirs'] = False
        
        if opts.img2img_batch_use_original_name:
            filename_pattern = f'{current_image_path_obj.stem}-[generation_number]' if p.n_iter > 1 or p.batch_size > 1 else f'{current_image_path_obj.stem}'
            p.override_settings['samples_filename_pattern'] = filename_pattern
        
        current_image_processed = modules.scripts.scripts_img2img.run(p, *script_args_tuple)
        if current_image_processed is None:
            current_image_processed = process_images(p)

        if current_image_processed:
            if all_processed_results is None:
                all_processed_results = current_image_processed
            else:
                all_processed_results.images.extend(current_image_processed.images)
                all_processed_results.infotexts.extend(current_image_processed.infotexts)
            
            limit = shared.opts.img2img_batch_show_results_limit
            if 0 <= limit < len(all_processed_results.images):
                all_processed_results.images = all_processed_results.images[:int(limit)]
                all_processed_results.infotexts = all_processed_results.infotexts[:int(limit)]

    return all_processed_results


def img2img_function_wrapper(id_task_gradio_progress: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img_bg, sketch_bg, sketch_fg, init_img_with_mask_bg, init_img_with_mask_fg, inpaint_color_sketch_bg, inpaint_color_sketch_fg, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, img2img_batch_source_type: str, img2img_batch_upload: list, *script_args):
    override_settings = create_override_settings_dict(override_settings_texts)
    is_batch = mode == 5
    height, width = int(height), int(width)

    image = None
    mask = None

    if mode == 0:
        image = init_img_bg
    elif mode == 1:
        if sketch_bg and sketch_fg: image = Image.alpha_composite(sketch_bg, sketch_fg)
        else: image = sketch_bg or sketch_fg
    elif mode == 2:
        image = init_img_with_mask_bg
        if init_img_with_mask_fg: 
            mask_channel = init_img_with_mask_fg.getchannel('A').convert('L')
            mask = Image.merge('RGBA', (mask_channel, mask_channel, mask_channel, Image.new('L', mask_channel.size, 255)))
    elif mode == 3:
        if inpaint_color_sketch_bg and inpaint_color_sketch_fg:
            image = Image.alpha_composite(inpaint_color_sketch_bg, inpaint_color_sketch_fg)
            mask_channel = inpaint_color_sketch_fg.getchannel('A').convert('L')
            short_side = min(mask_channel.size)
            dilation_size = int(0.015 * short_side) * 2 + 1
            if dilation_size > 0 : mask_channel = mask_channel.filter(ImageFilter.MaxFilter(dilation_size))
            mask = Image.merge('RGBA', (mask_channel, mask_channel, mask_channel, Image.new('L', mask_channel.size, 255)))
        else:
            image = inpaint_color_sketch_bg or inpaint_color_sketch_fg
    elif mode == 4:
        image = init_img_inpaint
        mask = init_mask_inpaint
    
    if image is None and not is_batch:
        error_html = plaintext_to_html("Input image missing for the selected mode.", classname="error")
        return [], "{}", error_html, ""

    if mask and isinstance(mask, Image.Image):
        mask = mask.point(lambda v: 255 if v > 128 else 0)

    image = images.fix_image(image)
    mask = images.fix_image(mask)

    if selected_scale_tab == 1 and not is_batch and image:
        width = int(image.width * scale_by)
        width -= width % 8
        height = int(image.height * scale_by)
        height -= height % 8

    assert 0. <= denoising_strength <= 1., 'Denoising strength must be between 0 and 1.'

    p = StableDiffusionProcessingImg2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt, negative_prompt=negative_prompt, styles=prompt_styles,
        batch_size=batch_size, n_iter=n_iter, cfg_scale=cfg_scale,
        width=width, height=height, init_images=[image] if image else None,
        mask=mask, mask_blur=mask_blur, inpainting_fill=inpainting_fill,
        resize_mode=resize_mode, denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale, inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings, distilled_cfg_scale=distilled_cfg_scale
    )
    p.scripts = modules.scripts.scripts_img2img
    p.script_args = script_args
    p.user = request.username if request else None

    if shared.opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    processed_result = None
    with closing(p):
        if is_batch:
            input_data_for_batch = img2img_batch_upload if img2img_batch_source_type == "upload" else img2img_batch_input_dir
            if not input_data_for_batch:
                 error_html = plaintext_to_html("Batch input missing.", classname="error")
                 return [], "{}", error_html, ""

            output_dir_for_batch = "" if img2img_batch_source_type == "upload" else img2img_batch_output_dir
            mask_dir_for_batch = "" if img2img_batch_source_type == "upload" else img2img_batch_inpaint_mask_dir
            
            processed_result = process_batch(
                p, input_data_for_batch, output_dir_for_batch, mask_dir_for_batch, script_args,
                to_scale=(selected_scale_tab == 1), scale_by=scale_by,
                use_png_info=img2img_batch_use_png_info,
                png_info_props=img2img_batch_png_info_props,
                png_info_dir=img2img_batch_png_info_dir if not shared.cmd_opts.hide_ui_dir_config else ""
            )
            if processed_result is None:
                processed_result = Processed(p, [], p.seed, "")
        else:
            if p.init_images is None or p.init_images[0] is None:
                error_html = plaintext_to_html("Input image required for img2img mode.", classname="error")
                return [], "{}", error_html, ""
            processed_result = modules.scripts.scripts_img2img.run(p, *script_args)
            if processed_result is None:
                processed_result = process_images(p)

    shared.total_tqdm.clear()
    generation_info_js = processed_result.js()
    if opts.samples_log_stdout: print(generation_info_js)
    if opts.do_not_show_images: processed_result.images = []

    return processed_result.images + processed_result.extra_images, generation_info_js, plaintext_to_html(processed_result.info), plaintext_to_html(processed_result.comments, classname="comments")

def img2img(id_task: str, request: gr.Request, *args):
    return main_thread.run_and_wait_result(img2img_function_wrapper,
                                           id_task,
                                           request,
                                           *args,
                                           gradio_progress_id=id_task)