import gradio as gr
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.ui_components import FormRow, FormGroup, ToolButton
from modules.ui_common import create_output_panel
from modules.ui_toprow import Toprow
from modules import scripts, processing, ui_common, timer
from modules.shared import opts, cmd_opts
import modules.infotext_utils as parameters_copypaste
from modules import prompt_parser
from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
from modules_forge import main_entry
import modules.processing_scripts.comments as comments
from modules.ui_gradio_extensions import reload_javascript
from contextlib import ExitStack
from modules import shared


def create_auraflow_interface():
    reload_javascript()

    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False, head=canvas_head) as auraflow_interface:
        toprow = Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box)

        dummy_component = gr.Textbox(visible=False)
        dummy_component_number = gr.Number(visible=False)

        extra_tabs = gr.Tabs(elem_id="auraflow_extra_tabs", elem_classes=["extra-networks"])
        extra_tabs.__enter__()

        with gr.Tab("Generation", id="auraflow_generation") as auraflow_generation_tab, modules.ui.ResizeHandleRow(equal_height=False):
            with ExitStack() as stack:
                if shared.opts.txt2img_settings_accordion:
                    stack.enter_context(gr.Accordion("Open for Settings", open=False))
                stack.enter_context(gr.Column(variant='compact', elem_id="auraflow_settings"))

                scripts.scripts_txt2img.prepare_ui()

                for category in modules.ui.ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="auraflow_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=1024, elem_id="auraflow_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=1024, elem_id="auraflow_height")

                            with gr.Column(elem_id="auraflow_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=modules.ui.switch_values_symbol, elem_id="auraflow_res_switch_btn", tooltip="Switch width/height")

                    elif category == "cfg":
                        with gr.Row():
                            distilled_cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Distilled CFG Scale', value=3.5, elem_id="auraflow_distilled_cfg_scale")
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='CFG Scale', value=7.0, elem_id="auraflow_cfg_scale")
                            cfg_scale.change(lambda x: gr.update(interactive=(x != 1)), inputs=[cfg_scale], outputs=[toprow.negative_prompt], queue=False, show_progress=False)

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "scripts":
                        with FormGroup(elem_id="auraflow_script_container"):
                            custom_inputs = scripts.scripts_txt2img.setup_ui()

                    if category not in {"accordions"}:
                        scripts.scripts_txt2img.setup_ui_for_section(category)

            output_panel = create_output_panel("auraflow", opts.outdir_txt2img_samples, toprow)

            auraflow_inputs = [
                dummy_component,
                toprow.prompt,
                toprow.negative_prompt,
                toprow.ui_styles.dropdown,
                gr.Number(value=1, visible=False),  # batch_count placeholder
                gr.Number(value=1, visible=False),  # batch_size placeholder
                cfg_scale,
                distilled_cfg_scale,
                height,
                width,
                gr.Checkbox(value=False, visible=False),  # enable_hr placeholder
                gr.Number(value=0.0, visible=False),  # denoising_strength placeholder
                gr.Number(value=1.0, visible=False),  # hr_scale placeholder
                gr.Textbox(value="", visible=False),  # hr_upscaler placeholder
                gr.Number(value=0, visible=False),  # hr_second_pass_steps placeholder
                gr.Number(value=0, visible=False),  # hr_resize_x placeholder
                gr.Number(value=0, visible=False),  # hr_resize_y placeholder
                gr.Textbox(value="Use same checkpoint", visible=False),  # hr_checkpoint_name placeholder
                gr.CheckboxGroup(value=["Use same choices"], visible=False),  # hr_additional_modules placeholder
                gr.Textbox(value="Use same sampler", visible=False),  # hr_sampler_name placeholder
                gr.Textbox(value="Use same scheduler", visible=False),  # hr_scheduler placeholder
                gr.Textbox(value="", visible=False),  # hr_prompt placeholder
                gr.Textbox(value="", visible=False),  # hr_negative_prompt placeholder
                gr.Number(value=7.0, visible=False),  # hr_cfg placeholder
                gr.Number(value=3.5, visible=False),  # hr_distilled_cfg placeholder
                gr.CheckboxGroup(value=[], visible=False),  # override_settings placeholder
            ] + custom_inputs

            auraflow_outputs = [
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
            ]

            auraflow_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=auraflow_inputs,
                outputs=auraflow_outputs,
                show_progress=False,
            )

            toprow.prompt.submit(**auraflow_args)
            toprow.submit.click(**auraflow_args)

            def select_gallery_image(index):
                index = int(index)
                if getattr(shared.opts, 'hires_button_gallery_insert', False):
                    index += 1
                return gr.update(selected_index=index)

            auraflow_upscale_inputs = auraflow_inputs[0:1] + [output_panel.gallery, dummy_component_number, output_panel.generation_info] + auraflow_inputs[1:]
            output_panel.button_upscale.click(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img_upscale, extra_outputs=[None, '', '']),
                _js="submit_auraflow_upscale",
                inputs=auraflow_upscale_inputs,
                outputs=auraflow_outputs,
                show_progress=False,
            ).then(fn=select_gallery_image, js="selected_gallery_index", inputs=[dummy_component], outputs=[output_panel.gallery])

            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressAuraflow",
                inputs=[dummy_component],
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            auraflow_paste_fields = [
                parameters_copypaste.PasteField(toprow.prompt, "Prompt", api="prompt"),
                parameters_copypaste.PasteField(toprow.negative_prompt, "Negative prompt", api="negative_prompt"),
                parameters_copypaste.PasteField(cfg_scale, "CFG scale", api="cfg_scale"),
                parameters_copypaste.PasteField(distilled_cfg_scale, "Distilled CFG Scale", api="distilled_cfg_scale"),
                parameters_copypaste.PasteField(width, "Size-1", api="width"),
                parameters_copypaste.PasteField(height, "Size-2", api="height"),
                parameters_copypaste.PasteField(toprow.ui_styles.dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update(), api="styles"),
                *scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("auraflow", None, auraflow_paste_fields, gr.CheckboxGroup(value=[], visible=False))
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=toprow.paste, tabname="auraflow", source_text_component=toprow.prompt, source_image_component=None,
            ))

            steps = scripts.scripts_txt2img.script('Sampler').steps

            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(modules.ui.update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(modules.ui.update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
            toprow.token_button.click(fn=wrap_queued_call(modules.ui.update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(modules.ui.update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])

        extra_networks_ui = ui_extra_networks.create_ui(auraflow_interface, [auraflow_generation_tab], 'auraflow')
        ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)

        extra_tabs.__exit__()

    scripts.scripts_current = None

    return auraflow_interface


# Add auraflow to shared tab names
if "AuraFlow" not in shared.tab_names:
    shared.tab_names.append("AuraFlow")