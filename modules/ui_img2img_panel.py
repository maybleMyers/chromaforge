"""
The img2img generation panel, factored out of modules.ui so more than one tab can own a
full copy of it.

`modules.ui.create_ui()` builds the Img2img tab from here, and `modules.ui_llm2img` builds
the LLM2img tab from the same code with a different tabname and its own ScriptRunner. The
only differences between the two are the elem_id prefix, which ScriptRunner backs the
script accordions, and the submit wiring - which stays with the caller, since the two tabs
submit to different functions.

Everything here was previously inline in modules/ui.py; the parameterisation is mechanical
(hardcoded "img2img_" elem_ids became f"{tabname}_", `scripts.scripts_img2img` became the
passed-in runner, and the img2img-specific `_js` helpers became their tabname-aware
`...Mode...` equivalents in javascript/ui.js).
"""

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, List

import gradio as gr
from PIL import Image

from modules import progress, shared
from modules.call_queue import wrap_queued_call
from modules.shared import opts
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML, ResizeHandleRow
from modules_forge import main_entry
from modules_forge.forge_canvas.canvas import ForgeCanvas

import modules.infotext_utils as parameters_copypaste


#   Index of each mode inside the "mode_<tabname>" Tabs, used by the copy-image buttons to
#   switch the destination tab from javascript.
MODE_TAB_INDEX = {
    'img2img': 0,
    'ref_edit': 1,
    'sketch': 2,
    'inpaint': 3,
    'inpaint_sketch': 4,
}

#   A handful of the original img2img elem_ids predate the "img2img_" convention and are
#   hardcoded in javascript/aspectRatioOverlay.js, javascript/imageMaskFix.js and CSS.
#   Renaming them to fit the f"{tabname}_..." pattern would silently break those, so the
#   img2img tab keeps its historical ids and only clones get the generated ones.
LEGACY_ELEM_IDS = {
    'img2img': {
        'maskimg': 'img2maskimg',
        'inpaint_sketch_canvas': 'inpaint_sketch',
        'inpaint_upload_base': 'img_inpaint_base',
        'inpaint_upload_mask': 'img_inpaint_mask',
        'resize_mode': 'resize_mode',
        'inpaint_controls': 'inpaint_controls',
    },
}


def _eid(tabname, key):
    return LEGACY_ELEM_IDS.get(tabname, {}).get(key, f"{tabname}_{key}")


#   Names of the values in Img2imgPanel.submit_inputs, after the leading id_task slot and
#   before the variable-length script-args tail. Mirrors modules.img2img.img2img_function's
#   parameter list exactly - modules.llm2img indexes into the argument tuple by name rather
#   than by a magic number, and create_img2img_panel() asserts the two stay in step.
IMG2IMG_ARG_NAMES = [
    'mode', 'prompt', 'negative_prompt', 'prompt_styles', 'init_img', 'sketch', 'sketch_fg',
    'init_img_with_mask', 'init_img_with_mask_fg', 'inpaint_color_sketch',
    'inpaint_color_sketch_fg', 'init_img_inpaint', 'init_mask_inpaint', 'ref_edit_main_img',
    'ref_edit_img1', 'ref_edit_img2', 'ref_edit_img3', 'ref_edit_img4', 'ref_edit_img5',
    'ref_edit_img6', 'ref_edit_lora_strength', 'mask_blur', 'mask_alpha', 'inpainting_fill',
    'n_iter', 'batch_size', 'cfg_scale', 'distilled_cfg_scale', 'zimage_shift',
    'sigma_rescale_start', 'sigma_rescale_end', 'apg_enabled', 'apg_eta', 'apg_momentum',
    'apg_threshold', 'image_cfg_scale', 'denoising_strength', 'selected_scale_tab', 'height',
    'width', 'scale_by', 'resize_mode', 'fill_left', 'fill_right', 'fill_up', 'fill_down',
    'fill_mask_mode', 'inpaint_full_res', 'inpaint_full_res_padding', 'inpainting_mask_invert',
    'img2img_batch_input_dir', 'img2img_batch_output_dir', 'img2img_batch_inpaint_mask_dir',
    'override_settings_texts', 'img2img_batch_use_png_info', 'img2img_batch_png_info_props',
    'img2img_batch_png_info_dir', 'img2img_batch_source_type', 'img2img_batch_upload',
    'checkpoint_name', 'vae_te_modules', 'forge_preset_name',
]


def arg_index(name):
    """Position of a named img2img_function argument within the tuple llm2img receives."""
    return IMG2IMG_ARG_NAMES.index(name)


@dataclass
class Img2imgPanel:
    """Every component the callers need to wire submit, extra networks and paste buttons."""

    tabname: str
    toprow: Any
    generation_tab: Any
    output_panel: Any
    dummy_component: Any
    #   Ordered exactly as modules.img2img.img2img_function expects them, minus id_task
    #   and request. Both callers must forward this list verbatim.
    submit_inputs: List[Any] = field(default_factory=list)
    custom_inputs: List[Any] = field(default_factory=list)
    paste_fields: List[Any] = field(default_factory=list)
    #   Individually named components the callers need to reach outside the panel.
    components: dict = field(default_factory=dict)


def create_img2img_panel(tabname, script_runner, toprow, dummy_component=None, extra_ui=None):
    """Build the img2img generation panel under `tabname` and return its components.

    Must be called inside the tab's gr.Blocks context. `script_runner` must already have
    had initialize_scripts(is_img2img=True) called on it.

    `extra_ui`, if given, is called with no arguments while the Generation tab is still the
    open gradio context, so a caller can append its own controls below the panel.
    """

    #   Imported lazily: both of these import modules.ui, which imports this module, so a
    #   module-level import would be circular. By the time this runs, modules.ui is loaded.
    from modules import ui as ui_module
    import modules.img2img

    create_output_panel = ui_module.create_output_panel
    ordered_ui_categories = ui_module.ordered_ui_categories
    resize_from_to_html = ui_module.resize_from_to_html
    create_override_settings_dropdown = ui_module.create_override_settings_dropdown
    process_interrogate = ui_module.process_interrogate
    interrogate = ui_module.interrogate
    interrogate_deepbooru = ui_module.interrogate_deepbooru
    expand_prompt_with_llm = ui_module.expand_prompt_with_llm
    update_token_counter = ui_module.update_token_counter
    update_negative_prompt_token_counter = ui_module.update_negative_prompt_token_counter
    switch_values_symbol = ui_module.switch_values_symbol
    detect_image_size_symbol = ui_module.detect_image_size_symbol
    reset_symbol = ui_module.reset_symbol

    mode_elem_id = f"mode_{tabname}"

    if dummy_component is None:
        dummy_component = gr.Textbox(visible=False)

    #   The settings/results split is entered through a stack rather than the `with` header
    #   so it can be closed early, letting extra_ui() render as a full-width block under the
    #   two columns instead of becoming a third column inside the row.
    with gr.Tab("Generation", id=f"{tabname}_generation") as generation_tab, ExitStack() as tab_stack:
        tab_stack.enter_context(ResizeHandleRow(equal_height=False))

        with ExitStack() as stack:
            if shared.opts.img2img_settings_accordion:
                stack.enter_context(gr.Accordion("Open for Settings", open=False))
            stack.enter_context(gr.Column(variant='compact', elem_id=f"{tabname}_settings"))

            copy_image_buttons = []
            copy_image_destinations = {}

            def add_copy_image_controls(tab_name, elem):
                with gr.Row(variant="compact", elem_id=f"{tabname}_copy_to_{tab_name}"):
                    for title, name in zip(['to img2img', 'to sketch', 'to inpaint', 'to inpaint sketch'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                        if name == tab_name:
                            gr.Button(title, interactive=False)
                            copy_image_destinations[name] = elem
                            continue

                        button = gr.Button(title)
                        copy_image_buttons.append((button, name, elem))

            script_runner.prepare_ui()

            for category in ordered_ui_categories():
                if category == "prompt":
                    toprow.create_inline_toprow_prompts()

                if category == "image":
                    with gr.Tabs(elem_id=mode_elem_id):
                        selected_tab = gr.Number(value=0, visible=False)

                        with gr.TabItem('img2img', id='img2img', elem_id=f"{tabname}_img2img_tab") as tab_img2img:
                            init_img = ForgeCanvas(elem_id=f"{tabname}_image", height=512, no_scribbles=True)
                            add_copy_image_controls('img2img', init_img)

                        with gr.TabItem('Reference Edit', id='img2img_ref_edit', elem_id=f"{tabname}_ref_edit_tab") as tab_ref_edit:
                            gr.HTML("<p class='text-gray-500' style='margin-bottom: 0.5em;'>FLUX.2/Chroma2 reference-based generation. Starts from pure noise - all images are used as references to guide generation.</p>")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    ref_edit_main_img = gr.Image(label="Main Image", source="upload", interactive=True, type="pil", elem_id=f"{tabname}_ref_edit_main")
                                with gr.Column(scale=2):
                                    gr.HTML("<p class='text-gray-500' style='margin-bottom: 0.3em;'>Reference Images (up to 6)</p>")
                                    with gr.Row():
                                        ref_edit_img1 = gr.Image(label="Ref 1", source="upload", interactive=True, type="pil", elem_id=f"{tabname}_ref_edit_1", height=150)
                                        ref_edit_img2 = gr.Image(label="Ref 2", source="upload", interactive=True, type="pil", elem_id=f"{tabname}_ref_edit_2", height=150)
                                        ref_edit_img3 = gr.Image(label="Ref 3", source="upload", interactive=True, type="pil", elem_id=f"{tabname}_ref_edit_3", height=150)
                                    with gr.Row():
                                        ref_edit_img4 = gr.Image(label="Ref 4", source="upload", interactive=True, type="pil", elem_id=f"{tabname}_ref_edit_4", height=150)
                                        ref_edit_img5 = gr.Image(label="Ref 5", source="upload", interactive=True, type="pil", elem_id=f"{tabname}_ref_edit_5", height=150)
                                        ref_edit_img6 = gr.Image(label="Ref 6", source="upload", interactive=True, type="pil", elem_id=f"{tabname}_ref_edit_6", height=150)
                            with gr.Row():
                                ref_edit_lora_strength = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.05, label="LoRA Strength (Z-Image i2L)", elem_id=f"{tabname}_ref_edit_lora_strength")

                        with gr.TabItem('Sketch', id='img2img_sketch', elem_id=f"{tabname}_img2img_sketch_tab") as tab_sketch:
                            sketch = ForgeCanvas(elem_id=f"{tabname}_sketch", height=512, scribble_color=opts.img2img_sketch_default_brush_color)
                            add_copy_image_controls('sketch', sketch)

                        with gr.TabItem('Inpaint', id='inpaint', elem_id=f"{tabname}_inpaint_tab") as tab_inpaint:
                            init_img_with_mask = ForgeCanvas(elem_id=_eid(tabname, 'maskimg'), height=512, contrast_scribbles=opts.img2img_inpaint_mask_high_contrast, scribble_color=opts.img2img_inpaint_mask_brush_color, scribble_color_fixed=True, scribble_alpha=opts.img2img_inpaint_mask_scribble_alpha, scribble_alpha_fixed=True, scribble_softness_fixed=True)
                            add_copy_image_controls('inpaint', init_img_with_mask)

                        with gr.TabItem('Inpaint sketch', id='inpaint_sketch', elem_id=f"{tabname}_inpaint_sketch_tab") as tab_inpaint_color:
                            inpaint_color_sketch = ForgeCanvas(elem_id=_eid(tabname, 'inpaint_sketch_canvas'), height=512, scribble_color=opts.img2img_inpaint_sketch_default_brush_color)
                            add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                        with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id=f"{tabname}_inpaint_upload_tab") as tab_inpaint_upload:
                            init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id=_eid(tabname, 'inpaint_upload_base'))
                            init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", image_mode="RGBA", elem_id=_eid(tabname, 'inpaint_upload_mask'))

                        with gr.TabItem('Batch', id='batch', elem_id=f"{tabname}_batch_tab") as tab_batch:
                            with gr.Tabs(elem_id=f"{tabname}_batch_source"):
                                img2img_batch_source_type = gr.Textbox(visible=False, value="upload")
                                with gr.TabItem('Upload', id='batch_upload', elem_id=f"{tabname}_batch_upload_tab") as tab_batch_upload:
                                    img2img_batch_upload = gr.Files(label="Files", interactive=True, elem_id=f"{tabname}_batch_upload")
                                with gr.TabItem('From directory', id='batch_from_dir', elem_id=f"{tabname}_batch_from_dir_tab") as tab_batch_from_dir:
                                    hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                                    gr.HTML(
                                        "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running." +
                                        "<br>Use an empty output directory to save pictures normally instead of writing to the output directory." +
                                        f"<br>Add inpaint batch mask directory to enable inpaint batch processing."
                                        f"{hidden}</p>"
                                    )
                                    img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, elem_id=f"{tabname}_batch_input_dir")
                                    img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, elem_id=f"{tabname}_batch_output_dir")
                                    img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory (required for inpaint batch processing only)", **shared.hide_dirs, elem_id=f"{tabname}_batch_inpaint_mask_dir")
                            tab_batch_upload.select(fn=lambda: "upload", inputs=[], outputs=[img2img_batch_source_type])
                            tab_batch_from_dir.select(fn=lambda: "from dir", inputs=[], outputs=[img2img_batch_source_type])
                            with gr.Accordion("PNG info", open=False):
                                img2img_batch_use_png_info = gr.Checkbox(label="Append png info to prompts", elem_id=f"{tabname}_batch_use_png_info")
                                img2img_batch_png_info_dir = gr.Textbox(label="PNG info directory", **shared.hide_dirs, placeholder="Leave empty to use input directory", elem_id=f"{tabname}_batch_png_info_dir")
                                img2img_batch_png_info_props = gr.CheckboxGroup(["Prompt", "Negative prompt", "Seed", "CFG scale", "Sampler", "Steps", "Model hash"], label="Parameters to take from png info", info="Prompts from png info will be appended to prompts set in ui.")

                        mode_tabs = [tab_img2img, tab_ref_edit, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]

                        for i, tab in enumerate(mode_tabs):
                            tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[selected_tab])

                    def copyCanvas_img2img(background, foreground, source):
                        if source == 2 or source == 4:  # 2 is Sketch, 4 is Inpaint sketch
                            bg = Image.alpha_composite(background, foreground)
                            return bg, None
                        return background, None

                    for button, name, elem in copy_image_buttons:
                        button.click(
                            fn=copyCanvas_img2img,
                            inputs=[elem.background, elem.foreground, selected_tab],
                            outputs=[copy_image_destinations[name].background, copy_image_destinations[name].foreground],
                        )
                        button.click(
                            fn=None,
                            _js=f"() => switchToModeTab('{mode_elem_id}', {MODE_TAB_INDEX[name]})",
                            inputs=[],
                            outputs=[],
                        )

                    with FormRow():
                        resize_mode = gr.Radio(label="Resize mode", elem_id=_eid(tabname, 'resize_mode'), choices=["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"], type="index", value="Just resize")

                    with FormGroup(elem_id=f"{tabname}_fill_direction", visible=False) as fill_direction_controls:
                        with FormRow():
                            fill_left = gr.Slider(label='Expand left %', minimum=0, maximum=200, step=1, value=0, elem_id=f"{tabname}_fill_left")
                            fill_right = gr.Slider(label='Expand right %', minimum=0, maximum=200, step=1, value=0, elem_id=f"{tabname}_fill_right")
                        with FormRow():
                            fill_up = gr.Slider(label='Expand up %', minimum=0, maximum=200, step=1, value=0, elem_id=f"{tabname}_fill_up")
                            fill_down = gr.Slider(label='Expand down %', minimum=0, maximum=200, step=1, value=0, elem_id=f"{tabname}_fill_down")
                        with FormRow():
                            fill_mask_mode = gr.Radio(label='New area', choices=['Leave mask unchanged', 'Add new area to mask', 'Mask only the new area'], type="index", value='Add new area to mask', elem_id=f"{tabname}_fill_mask_mode",
                                                      info="Percentages grow the canvas outward and set the target size, overriding Width/Height. 'Leave mask unchanged' composites the old edge back over the new area, so it will not outpaint.")

                    resize_mode.change(fn=lambda m: gr.update(visible=(m == 2)), inputs=[resize_mode], outputs=[fill_direction_controls], queue=False, show_progress=False)

                elif category == "dimensions":
                    with FormRow():
                        with gr.Column(elem_id=f"{tabname}_column_size", scale=4):
                            selected_scale_tab = gr.Number(value=0, visible=False)

                            with gr.Tabs(elem_id=f"{tabname}_tabs_resize"):
                                with gr.Tab(label="Resize to", id="to", elem_id=f"{tabname}_tab_resize_to") as tab_scale_to:
                                    with FormRow():
                                        with gr.Column(elem_id=f"{tabname}_column_size", scale=4):
                                            width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id=f"{tabname}_width")
                                            height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id=f"{tabname}_height")
                                        with gr.Column(elem_id=f"{tabname}_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                            res_switch_btn = ToolButton(value=switch_values_symbol, elem_id=f"{tabname}_res_switch_btn", tooltip="Switch width/height")
                                            detect_image_size_btn = ToolButton(value=detect_image_size_symbol, elem_id=f"{tabname}_detect_image_size_btn", tooltip="Auto detect size from img2img")

                                with gr.Tab(label="Resize by", id="by", elem_id=f"{tabname}_tab_resize_by") as tab_scale_by:
                                    scale_by = gr.Slider(minimum=0.05, maximum=4.0, step=0.01, label="Scale", value=1.0, elem_id=f"{tabname}_scale")

                                    with FormRow():
                                        scale_by_html = FormHTML(resize_from_to_html(0, 0, 0.0), elem_id=f"{tabname}_scale_resolution_preview")
                                        gr.Slider(label="Unused", elem_id=f"{tabname}_unused_scale_by_slider")
                                        button_update_resize_to = gr.Button(visible=False, elem_id=f"{tabname}_update_resize_to")

                                on_change_args = dict(
                                    fn=resize_from_to_html,
                                    _js=f"(w, h, r) => currentModeSourceResolution('{mode_elem_id}', w, h, r)",
                                    inputs=[dummy_component, dummy_component, scale_by],
                                    outputs=scale_by_html,
                                    show_progress=False,
                                )

                                scale_by.change(**on_change_args)
                                button_update_resize_to.click(**on_change_args)

                                img_sources = [init_img.background, ref_edit_main_img, sketch.background, init_img_with_mask.background, inpaint_color_sketch.background, init_img_inpaint]
                                for i in img_sources:
                                    i.change(**on_change_args)

                        tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
                        tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])

                        if opts.dimensions_and_batch_together:
                            with gr.Column(elem_id=f"{tabname}_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{tabname}_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{tabname}_batch_size")

                elif category == "denoising":
                    with gr.Row():
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75, elem_id=f"{tabname}_denoising_strength")
                        reset_denoising_btn = ToolButton(value=reset_symbol, elem_id=f"{tabname}_reset_denoising_btn", tooltip="Reset denoising strength to 0.75")
                        reset_denoising_btn.click(fn=lambda: gr.update(value=0.75), inputs=[], outputs=[denoising_strength], show_progress=False)

                elif category == "cfg":
                    with gr.Row():
                        distilled_cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Distilled CFG Scale', value=3.5, elem_id=f"{tabname}_distilled_cfg_scale")
                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id=f"{tabname}_cfg_scale")
                        zimage_shift = gr.Slider(minimum=0.0, maximum=30.0, step=0.05, label='Z-Image Shift', value=0.0, elem_id=f"{tabname}_zimage_shift")
                        if tabname == "img2img":
                            #   main_entry toggles this slider's visibility per Forge preset; it
                            #   only tracks the two original tabs, so a clone must not claim it.
                            main_entry.ui_img2img_zimage_shift = zimage_shift
                        image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale', value=1.5, elem_id=f"{tabname}_image_cfg_scale", visible=False)
                        cfg_scale.change(lambda x: gr.update(interactive=(x != 1)), inputs=[cfg_scale], outputs=[toprow.negative_prompt], queue=False, show_progress=False)
                    with gr.Row():
                        sigma_rescale_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Sigma Rescale Start', value=1.0, elem_id=f"{tabname}_sigma_rescale_start")
                        sigma_rescale_end = gr.Slider(minimum=0.0, maximum=0.5, step=0.01, label='Sigma Rescale End', value=0.0, elem_id=f"{tabname}_sigma_rescale_end")
                    with gr.Row():
                        apg_enabled = gr.Checkbox(label='APG (Adaptive Projected Guidance)', value=False, elem_id=f"{tabname}_apg_enabled")
                        apg_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='APG Eta', value=1.0, elem_id=f"{tabname}_apg_eta")
                        apg_momentum = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, label='APG Momentum', value=-0.5, elem_id=f"{tabname}_apg_momentum")
                        apg_threshold = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, label='APG Threshold', value=0.0, elem_id=f"{tabname}_apg_threshold")

                elif category == "checkboxes":
                    with FormRow(elem_classes="checkboxes-row", variant="compact"):
                        pass

                elif category == "accordions":
                    with gr.Row(elem_id=f"{tabname}_accordions", elem_classes="accordions"):
                        script_runner.setup_ui_for_section(category)

                elif category == "batch":
                    if not opts.dimensions_and_batch_together:
                        with FormRow(elem_id=f"{tabname}_column_batch"):
                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{tabname}_batch_count")
                            batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{tabname}_batch_size")

                elif category == "override_settings":
                    with FormRow(elem_id=f"{tabname}_override_settings_row") as row:
                        override_settings = create_override_settings_dropdown(tabname, row)

                elif category == "scripts":
                    with FormGroup(elem_id=f"{tabname}_script_container"):
                        custom_inputs = script_runner.setup_ui()

                elif category == "inpaint":
                    with FormGroup(elem_id=_eid(tabname, 'inpaint_controls'), visible=False) as inpaint_controls:
                        with FormRow():
                            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id=f"{tabname}_mask_blur")
                            mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id=f"{tabname}_mask_alpha")

                        with FormRow():
                            inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id=f"{tabname}_mask_mode")

                        with FormRow():
                            inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index", elem_id=f"{tabname}_inpainting_fill")

                        with FormRow():
                            with gr.Column():
                                inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id=f"{tabname}_inpaint_full_res")

                            with gr.Column(scale=4):
                                inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32, elem_id=f"{tabname}_inpaint_full_res_padding")

                if category not in {"accordions"}:
                    script_runner.setup_ui_for_section(category)

        def expand_target_size(src_w, src_h, l, r, u, d, w, h):
            if not src_w or not src_h:
                return w, h

            if any(float(pct) > 0 for pct in (l, r, u, d)):
                return modules.img2img.compute_expansion(int(src_w), int(src_h), l, r, u, d)[4:]

            return (int(src_w), int(src_h)) if shared.opts.img2img_autosize else (w, h)

        def updateWH(img, w, h, l, r, u, d):
            if img is None:
                return w, h
            return expand_target_size(img.size[0], img.size[1], l, r, u, d, w, h)

        fill_sliders = [fill_left, fill_right, fill_up, fill_down]

        for i in img_sources:
            i.change(fn=updateWH, inputs=[i, width, height] + fill_sliders, outputs=[width, height], show_progress='hidden')

        expand_size_args = dict(
            fn=expand_target_size,
            _js=f"(...a) => currentModeExpandResolution('{mode_elem_id}', ...a)",
            inputs=[dummy_component, dummy_component] + fill_sliders + [width, height],
            outputs=[width, height],
            show_progress=False,
            queue=False,
        )

        for slider in fill_sliders:
            slider.change(**expand_size_args)

        resize_mode.change(**expand_size_args)

        def select_img2img_tab(tab):
            return gr.update(visible=tab in [3, 4, 5]), gr.update(visible=tab == 4),

        for i, elem in enumerate(mode_tabs):
            elem.select(
                fn=lambda tab=i: select_img2img_tab(tab),
                inputs=[],
                outputs=[inpaint_controls, mask_alpha],
            )

        output_panel = create_output_panel(tabname, opts.outdir_img2img_samples, toprow)

        submit_inputs = [
            dummy_component,
            selected_tab,
            toprow.prompt,
            toprow.negative_prompt,
            toprow.ui_styles.dropdown,
            init_img.background,
            sketch.background,
            sketch.foreground,
            init_img_with_mask.background,
            init_img_with_mask.foreground,
            inpaint_color_sketch.background,
            inpaint_color_sketch.foreground,
            init_img_inpaint,
            init_mask_inpaint,
            ref_edit_main_img,
            ref_edit_img1,
            ref_edit_img2,
            ref_edit_img3,
            ref_edit_img4,
            ref_edit_img5,
            ref_edit_img6,
            ref_edit_lora_strength,
            mask_blur,
            mask_alpha,
            inpainting_fill,
            batch_count,
            batch_size,
            cfg_scale,
            distilled_cfg_scale,
            zimage_shift,
            sigma_rescale_start,
            sigma_rescale_end,
            apg_enabled,
            apg_eta,
            apg_momentum,
            apg_threshold,
            image_cfg_scale,
            denoising_strength,
            selected_scale_tab,
            height,
            width,
            scale_by,
            resize_mode,
            fill_left,
            fill_right,
            fill_up,
            fill_down,
            fill_mask_mode,
            inpaint_full_res,
            inpaint_full_res_padding,
            inpainting_mask_invert,
            img2img_batch_input_dir,
            img2img_batch_output_dir,
            img2img_batch_inpaint_mask_dir,
            override_settings,
            img2img_batch_use_png_info,
            img2img_batch_png_info_props,
            img2img_batch_png_info_dir,
            img2img_batch_source_type,
            img2img_batch_upload,
            main_entry.ui_checkpoint,
            main_entry.ui_vae,
            main_entry.ui_forge_preset,
        ] + custom_inputs

        interrogate_args = dict(
            _js=f"(...a) => getModeTabIndexArgs('{mode_elem_id}', ...a)",
            inputs=[
                dummy_component,
                img2img_batch_input_dir,
                img2img_batch_output_dir,
                init_img.background,
                ref_edit_main_img,
                sketch.background,
                init_img_with_mask.background,
                inpaint_color_sketch.background,
                init_img_inpaint,
            ],
            outputs=[toprow.prompt, dummy_component],
        )

        res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

        detect_image_size_btn.click(
            fn=lambda w, h: (w or gr.update(), h or gr.update()),
            _js=f"(w, h) => currentModeSourceResolution('{mode_elem_id}', w, h)",
            inputs=[dummy_component, dummy_component],
            outputs=[width, height],
            show_progress=False,
        )

        toprow.restore_progress_button.click(
            fn=progress.restore_progress,
            _js=f"() => restoreProgressForTab('{tabname}')",
            inputs=[dummy_component],
            outputs=[
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
            ],
            show_progress=False,
        )

        toprow.button_interrogate.click(
            fn=lambda *args: process_interrogate(interrogate, *args),
            **interrogate_args,
        )

        toprow.button_deepbooru.click(
            fn=lambda *args: process_interrogate(interrogate_deepbooru, *args),
            **interrogate_args,
        )

        steps = script_runner.script('Sampler').steps

        toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
        toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
        toprow.token_button.click(fn=update_token_counter, inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
        toprow.negative_token_button.click(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])

        # Connect expand prompt buttons for Prompt Expansion accordion (img2img with image context)
        # Expand Positive Prompt button - uses image from img2img
        toprow.expand_positive_button.click(
            fn=expand_prompt_with_llm,
            inputs=[
                toprow.prompt,
                init_img.background,  # Image context from img2img
                toprow.llm_model_dropdown,
                toprow.positive_system_prompt,
                gr.State(False),  # is_negative=False
                gr.State(None),  # positive_prompt (not needed for positive expansion)
                toprow.user_prompt_input  # User input to append with <|user|> tag
            ],
            outputs=[toprow.prompt],
            show_progress=True,
        )

        # Expand Negative Prompt button - uses image from img2img and positive prompt as context
        toprow.expand_negative_button.click(
            fn=expand_prompt_with_llm,
            inputs=[
                toprow.negative_prompt,
                init_img.background,  # Image context from img2img
                toprow.llm_model_dropdown,
                toprow.negative_system_prompt,
                gr.State(True),  # is_negative=True
                toprow.prompt,  # Pass positive prompt as context
                toprow.user_prompt_input  # User input to append with <|user|> tag
            ],
            outputs=[toprow.negative_prompt],
            show_progress=True,
        )

        # Legacy expand prompt button (hidden, but kept for backward compatibility)
        toprow.expand_prompt_button.click(
            fn=expand_prompt_with_llm,
            inputs=[toprow.prompt, init_img.background],
            outputs=[toprow.prompt],
            show_progress=True,
        )

        paste_fields = [
            (toprow.prompt, "Prompt"),
            (toprow.negative_prompt, "Negative prompt"),
            (cfg_scale, "CFG scale"),
            (distilled_cfg_scale, "Distilled CFG Scale"),
            (image_cfg_scale, "Image CFG scale"),
            (width, "Size-1"),
            (height, "Size-2"),
            (batch_size, "Batch size"),
            (toprow.ui_styles.dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update()),
            (denoising_strength, "Denoising strength"),
            (mask_blur, "Mask blur"),
            (inpainting_mask_invert, 'Mask mode'),
            (inpainting_fill, 'Masked content'),
            (inpaint_full_res, 'Inpaint area'),
            (inpaint_full_res_padding, 'Masked area padding'),
            (zimage_shift, "Z-Image Shift"),
            (sigma_rescale_start, "Sigma Rescale Start"),
            (sigma_rescale_end, "Sigma Rescale End"),
            (apg_enabled, lambda d: "APG Eta" in d),
            (apg_eta, "APG Eta"),
            (apg_momentum, "APG Momentum"),
            (apg_threshold, "APG Threshold"),
            *script_runner.infotext_fields
        ]
        parameters_copypaste.add_paste_fields(tabname, init_img.background, paste_fields, override_settings)
        if tabname == "img2img":
            #   "inpaint" is a paste destination name baked into the send-to buttons; only
            #   the real img2img tab may claim it.
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask.background, paste_fields, override_settings)
        parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
            paste_button=toprow.paste, tabname=tabname, source_text_component=toprow.prompt, source_image_component=None,
        ))

        if extra_ui is not None:
            #   Leave the two-column row first, so the caller's controls span the tab.
            tab_stack.close()
            extra_ui()

    assert len(submit_inputs) == 1 + len(IMG2IMG_ARG_NAMES) + len(custom_inputs), (
        f"{tabname}: submit_inputs has {len(submit_inputs)} entries but IMG2IMG_ARG_NAMES "
        f"describes {1 + len(IMG2IMG_ARG_NAMES) + len(custom_inputs)}. Update both together "
        f"when img2img_function gains or loses an argument."
    )

    panel = Img2imgPanel(
        tabname=tabname,
        toprow=toprow,
        generation_tab=generation_tab,
        output_panel=output_panel,
        dummy_component=dummy_component,
        submit_inputs=submit_inputs,
        custom_inputs=custom_inputs,
        paste_fields=paste_fields,
    )
    panel.components = dict(
        selected_tab=selected_tab,
        init_img=init_img,
        width=width,
        height=height,
        batch_count=batch_count,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        image_cfg_scale=image_cfg_scale,
        denoising_strength=denoising_strength,
        override_settings=override_settings,
        resize_mode=resize_mode,
        inpaint_controls=inpaint_controls,
        mask_alpha=mask_alpha,
    )
    return panel
