"""
The LLM2img tab: a full copy of the img2img option set, plus vlm.py's llama-server
settings, driving the prompt -> batch -> review loop in modules.llm2img.
"""

import json
import os

import gradio as gr

from modules import llm2img, scripts, shared, ui_extra_networks, ui_toprow
from modules.paths_internal import script_path
from modules.ui_components import FormRow
from modules.ui_img2img_panel import IMG2IMG_ARG_NAMES, create_img2img_panel
from modules_forge.forge_canvas.canvas import canvas_head

SETTINGS_FILE = os.path.join(script_path, "llm2img_settings.json")

#   Everything vlm.py keeps under Model Settings, minus its Backend dropdown: LLM2img is
#   llama-server only, because stopping a subprocess is the only reliable way to get the
#   VRAM back before the diffusion model loads.
DEFAULTS = {
    'n_gpu_layers': 99,
    'n_ctx': 32768,
    'tensor_split': "",
    'main_gpu': 0,
    'kv_cache_type': "f16",
    'flash_attn': True,
    'use_mmap': True,
    'use_mlock': False,
    'override_tensor': "",
    'extra_args': "",
    'server_port': 8080,
    'llama_server_path': "llama.cpp/build/bin/llama-server",

    #   A thinking model spends this budget on its reasoning before it writes a word of the
    #   answer, so 2048 routinely left the review cut off mid-critique with no prompt in it.
    'max_tokens': 8192,
    'temperature': 0.7,
    'top_p': 0.95,
    'repeat_penalty': 1.0,
    'seed': -1,
    'video_max_frames': 8,
    'every_other_frame': False,
    'show_thinking': False,
    'reasoning_level': "default",
    'thinking_mode': True,

    'idea': "",
    'writer_system_prompt': llm2img.DEFAULT_WRITER_SYSTEM_PROMPT,
    'reviewer_system_prompt': llm2img.DEFAULT_REVIEWER_SYSTEM_PROMPT,
    'rounds': 3,
    'max_review_images': 8,
    'feed_forward': True,
    'unload_diffusion': True,
    'keep_llm_loaded': False,

    'model_name': None,
}

#   Rough vision-token cost of one image for a Qwen3-VL-class model at default resolution.
#   Only used to warn before a review turn blows the context window.
TOKENS_PER_IMAGE_ESTIMATE = 1500

#   Per-model load profiles are shared with vlm.py (vlm_model_profiles.json). Its
#   backend_type key has no component here, so it is filtered out.
PROFILE_KEYS = ['n_gpu_layers', 'n_ctx', 'tensor_split', 'main_gpu', 'kv_cache_type',
                'flash_attn', 'use_mmap', 'use_mlock', 'override_tensor', 'extra_args',
                'server_port']

#   Image-generation settings the Save button persists, out of IMG2IMG_ARG_NAMES. Left out
#   on purpose:
#     - the canvases and file uploads (init_img, sketch, ref_edit_*, batch upload): not
#       JSON, and re-populating them on startup is not what anyone wants;
#     - mode and selected_scale_tab: driven by which sub-tab is open, so a restored value
#       would disagree with what the UI is actually showing;
#     - checkpoint_name / vae_te_modules / forge_preset_name: these are the global
#       quicksettings widgets shared with txt2img and img2img, not this tab's to own.
#   Script/extension args are left to ui-config.json, which already keys them per script.
SAVEABLE_IMG_ARGS = [
    'prompt', 'negative_prompt', 'prompt_styles',
    'ref_edit_lora_strength', 'mask_blur', 'mask_alpha', 'inpainting_fill',
    'n_iter', 'batch_size', 'cfg_scale', 'distilled_cfg_scale', 'zimage_shift',
    'sigma_rescale_start', 'sigma_rescale_end',
    'apg_enabled', 'apg_eta', 'apg_momentum', 'apg_threshold',
    'image_cfg_scale', 'denoising_strength', 'height', 'width', 'scale_by', 'resize_mode',
    'fill_left', 'fill_right', 'fill_up', 'fill_down', 'fill_mask_mode',
    'inpaint_full_res', 'inpaint_full_res_padding', 'inpainting_mask_invert',
    'img2img_batch_input_dir', 'img2img_batch_output_dir', 'img2img_batch_inpaint_mask_dir',
    'override_settings_texts', 'img2img_batch_use_png_info', 'img2img_batch_png_info_props',
    'img2img_batch_png_info_dir',
]


def load_settings():
    """Returns (llm settings, image settings), both already merged over the defaults."""
    settings = dict(DEFAULTS)
    image = {}
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
                stored_image = saved.get("image")
                if isinstance(stored_image, dict):
                    image = {k: v for k, v in stored_image.items() if k in SAVEABLE_IMG_ARGS}
            print(f"[llm2img] Settings loaded from {SETTINGS_FILE}")
    except Exception as e:
        print(f"[llm2img] Could not read {SETTINGS_FILE}: {e}; using defaults")
    return settings, image


def save_settings(*values):
    """Persist the whole page: the LLM args first, then SAVEABLE_IMG_ARGS."""
    n = len(llm2img.LLM_ARG_NAMES)
    settings = dict(zip(llm2img.LLM_ARG_NAMES, values[:n]))
    settings["image"] = dict(zip(SAVEABLE_IMG_ARGS, values[n:]))
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return f"Saved {len(values)} settings to {os.path.basename(SETTINGS_FILE)}"
    except Exception as e:
        return f"Error saving settings: {e}"


def _restorable(component, value):
    """Convert a value as a handler received it back into a valid `component.value`.

    Radios and Dropdowns declared with type="index" hand the callback an integer but must
    be given one of their choices to display, so the integer has to be mapped back.
    """
    if getattr(component, "type", "value") == "index":
        choices = [c[0] if isinstance(c, tuple) else c for c in (component.choices or [])]
        if isinstance(value, int) and 0 <= value < len(choices):
            return choices[value]
        return None
    return value


def _own_the_value(component):
    """Stop ui-config.json from overwriting this component's value at startup.

    UiLoadsave.add_component() does `setattr(obj, 'value', saved_value)` for every tracked
    component (modules/ui_loadsave.py:71), which would clobber what we just restored from
    llm2img_settings.json with whatever ui-config recorded on the very first run.
    """
    component.do_not_save_to_config = True


def create_llm2img_interface():
    """Build the LLM2img tab. Returns the gr.Blocks for modules.ui to register."""

    saved, saved_image = load_settings()

    scripts.scripts_current = scripts.scripts_llm2img
    scripts.scripts_llm2img.initialize_scripts(is_img2img=True)

    llm_components = {}

    with gr.Blocks(analytics_enabled=False, head=canvas_head) as llm2img_interface:
        toprow = ui_toprow.Toprow(is_img2img=True, is_compact=shared.opts.compact_prompt_box, id_part="llm2img")

        extra_tabs = gr.Tabs(elem_id="llm2img_extra_tabs", elem_classes=["extra-networks"])
        extra_tabs.__enter__()

        def build_llm_ui():
            """The LLM half of the tab, rendered below the img2img panel."""
            model_choices = llm2img.get_model_names()
            initial_model = saved.get('model_name')
            if initial_model not in model_choices:
                initial_model = model_choices[0] if model_choices else None

            gr.Markdown(
                "### LLM2img\n"
                "Each round: unload the diffusion model → start llama-server → the LLM "
                "writes or refines the prompt → stop llama-server → generate a batch with "
                "the settings above → restart llama-server → the VLM reviews the whole "
                "batch and returns the next round's prompt. Press **Interrupt** to stop at "
                "the next phase boundary."
            )

            with gr.Accordion("LLM Model Settings (llama-server)", open=True):
                with FormRow():
                    model_dropdown = gr.Dropdown(
                        label="GGUF Model", choices=model_choices, value=initial_model,
                        interactive=True, scale=4, elem_id="llm2img_llm_model",
                        info="Folders under models/LLM containing *.gguf. An mmproj-*.gguf "
                             "beside the weights enables vision, which the review needs.",
                    )
                    refresh_models_btn = gr.Button("Refresh", scale=1, min_width=80)

                with FormRow():
                    n_gpu_layers = gr.Slider(minimum=-1, maximum=100, step=1, label="GPU Layers (-1 = all)",
                                             value=saved['n_gpu_layers'], elem_id="llm2img_n_gpu_layers")
                    n_ctx = gr.Slider(minimum=512, maximum=262144, step=512, label="Context Length",
                                      value=saved['n_ctx'], elem_id="llm2img_n_ctx")
                    main_gpu = gr.Slider(minimum=0, maximum=7, step=1, label="Main GPU",
                                         value=saved['main_gpu'], elem_id="llm2img_main_gpu")
                    kv_cache_type = gr.Dropdown(label="KV Cache Type", choices=["f16", "q8_0", "q4_0"],
                                                value=saved['kv_cache_type'], elem_id="llm2img_kv_cache_type")

                with FormRow():
                    tensor_split = gr.Textbox(label="Tensor Split (Multi-GPU)", value=saved['tensor_split'],
                                              placeholder="e.g. 2,1", elem_id="llm2img_tensor_split",
                                              info="Comma-separated ratios for spreading layers across GPUs")
                    flash_attn = gr.Checkbox(label="Flash Attention", value=saved['flash_attn'],
                                             elem_id="llm2img_flash_attn")
                    use_mmap = gr.Checkbox(label="Use MMap", value=saved['use_mmap'],
                                           elem_id="llm2img_use_mmap",
                                           info="Uncheck to load fully into RAM")
                    use_mlock = gr.Checkbox(label="Lock in RAM (mlock)", value=saved['use_mlock'],
                                            elem_id="llm2img_use_mlock")

                with FormRow():
                    override_tensor = gr.Textbox(label="Override Tensor (-ot)", value=saved['override_tensor'],
                                                 placeholder=r"\.ffn_.*_exps\.weight=CPU", scale=3,
                                                 elem_id="llm2img_override_tensor",
                                                 info="MoE optimisation: keep expert FFN on CPU. ; separates patterns.")
                    extra_args = gr.Textbox(label="Extra Args", value=saved['extra_args'], scale=2,
                                            placeholder="--n-cpu-moe 82", elem_id="llm2img_extra_args")
                    server_port = gr.Number(label="Server Port", value=saved['server_port'], precision=0,
                                            elem_id="llm2img_server_port")
                    llama_server_path = gr.Textbox(label="llama-server Path", value=saved['llama_server_path'],
                                                   scale=2, elem_id="llm2img_llama_server_path")

                with FormRow():
                    load_btn = gr.Button("Load Model", variant="primary")
                    unload_btn = gr.Button("Unload")
                    status_display = gr.Textbox(label="LLM Status", value="No model loaded",
                                                interactive=False, scale=3)

            with gr.Accordion("LLM Generation Settings", open=False):
                with FormRow():
                    max_tokens = gr.Slider(minimum=64, maximum=262048, step=64, label="Max New Tokens",
                                           value=saved['max_tokens'], elem_id="llm2img_max_tokens",
                                           info="Shared by the reasoning and the answer — too low and the review is cut off before it reaches the revised prompt")
                    temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, label="Temperature",
                                            value=saved['temperature'], elem_id="llm2img_temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Top P",
                                      value=saved['top_p'], elem_id="llm2img_top_p")
                    repeat_penalty = gr.Slider(minimum=0.8, maximum=1.5, step=0.01, label="Repeat Penalty",
                                               value=saved['repeat_penalty'], elem_id="llm2img_repeat_penalty",
                                               info="1.0 = disabled")

                with FormRow():
                    llm_seed = gr.Number(label="LLM Seed", value=saved['seed'], precision=0,
                                         elem_id="llm2img_llm_seed", info="-1 = random")
                    show_thinking = gr.Checkbox(label="Show Thinking", value=saved['show_thinking'],
                                                elem_id="llm2img_show_thinking",
                                                info="Include the reasoning block in the round log")
                    reasoning_level = gr.Dropdown(label="Reasoning Level",
                                                  choices=["default", "low", "medium", "high", "xhigh"],
                                                  value=saved['reasoning_level'], elem_id="llm2img_reasoning_level",
                                                  info="Muse Glimmer strength / GPT-OSS effort")
                    thinking_mode = gr.Checkbox(label="Thinking Mode", value=saved['thinking_mode'],
                                                elem_id="llm2img_thinking_mode",
                                                info="Kimi K2.6: off = instant mode")

                with FormRow():
                    video_max_frames = gr.Slider(minimum=1, maximum=201, step=1, label="Max Video Frames",
                                                 value=saved['video_max_frames'], elem_id="llm2img_video_max_frames",
                                                 info="Carried over from vlm.py; unused when reviewing stills")
                    every_other_frame = gr.Checkbox(label="Every Other Frame", value=saved['every_other_frame'],
                                                    elem_id="llm2img_every_other_frame",
                                                    info="Carried over from vlm.py; unused when reviewing stills")

            with gr.Accordion("Loop", open=True):
                idea = gr.Textbox(label="Idea / concept", lines=2, value=saved['idea'],
                                  elem_id="llm2img_idea",
                                  placeholder="What you want to see. Leave empty to start from the prompt box above instead.")

                with FormRow():
                    rounds = gr.Slider(minimum=1, maximum=20, step=1, label="Rounds",
                                       value=saved['rounds'], elem_id="llm2img_rounds")
                    max_review_images = gr.Slider(minimum=1, maximum=llm2img.MAX_REVIEW_IMAGES, step=1,
                                                  label="Max review images", value=saved['max_review_images'],
                                                  elem_id="llm2img_max_review_images",
                                                  info=f"All go to the VLM in one turn (~{TOKENS_PER_IMAGE_ESTIMATE} vision tokens each)")
                    feed_forward = gr.Checkbox(label="Feed first result forward", value=saved['feed_forward'],
                                               elem_id="llm2img_feed_forward",
                                               info="Rounds after the first run img2img on the previous batch's first image")
                    unload_diffusion = gr.Checkbox(label="Unload diffusion before the LLM", value=saved['unload_diffusion'],
                                                   elem_id="llm2img_unload_diffusion",
                                                   info="Leave on unless both models fit in VRAM at once")
                    keep_llm_loaded = gr.Checkbox(label="Keep LLM loaded across rounds", value=saved['keep_llm_loaded'],
                                                  elem_id="llm2img_keep_llm_loaded",
                                                  info="Skips the reload, but then the LLM and the checkpoint share VRAM")

                with gr.Accordion("System prompts", open=False):
                    writer_system_prompt = gr.Textbox(label="Prompt writer (round 1)", lines=8,
                                                      value=saved['writer_system_prompt'],
                                                      elem_id="llm2img_writer_system_prompt")
                    reviewer_system_prompt = gr.Textbox(label="Reviewer (every round)", lines=12,
                                                        value=saved['reviewer_system_prompt'],
                                                        elem_id="llm2img_reviewer_system_prompt")

                with FormRow():
                    save_settings_btn = gr.Button("Save LLM2img Settings", variant="secondary")
                    save_status = gr.Textbox(label="", value="", interactive=False, show_label=False, scale=3)
                    gr.Markdown(
                        "Saves everything on this tab — LLM settings *and* the image "
                        "generation settings above — to `llm2img_settings.json`."
                    )

            current_prompt = gr.Textbox(label="Current prompt", lines=3, value="",
                                        elem_id="llm2img_current_prompt",
                                        info="The prompt the last round used, or the one the review produced. Edit it and press Generate to continue from here.")
            round_log = gr.Markdown(value="", elem_id="llm2img_round_log")

            llm_components.update(
                model_name=model_dropdown,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                tensor_split=tensor_split,
                main_gpu=main_gpu,
                kv_cache_type=kv_cache_type,
                flash_attn=flash_attn,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                override_tensor=override_tensor,
                extra_args=extra_args,
                server_port=server_port,
                llama_server_path=llama_server_path,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=llm_seed,
                video_max_frames=video_max_frames,
                every_other_frame=every_other_frame,
                show_thinking=show_thinking,
                reasoning_level=reasoning_level,
                thinking_mode=thinking_mode,
                idea=idea,
                writer_system_prompt=writer_system_prompt,
                reviewer_system_prompt=reviewer_system_prompt,
                rounds=rounds,
                max_review_images=max_review_images,
                feed_forward=feed_forward,
                unload_diffusion=unload_diffusion,
                keep_llm_loaded=keep_llm_loaded,
            )
            llm_components['_status'] = status_display
            llm_components['_round_log'] = round_log
            llm_components['_current_prompt'] = current_prompt
            #   Wired after create_img2img_panel() returns: the image settings are part of
            #   what Save persists, and their components do not exist yet.
            llm_components['_save_btn'] = save_settings_btn
            llm_components['_save_status'] = save_status

            # ---- LLM model lifecycle -------------------------------------------------
            refresh_models_btn.click(
                fn=lambda: gr.update(choices=llm2img.get_model_names()),
                outputs=[model_dropdown],
                show_progress=False,
            )

            model_dropdown.change(
                fn=_apply_model_profile,
                inputs=[model_dropdown],
                outputs=[llm_components[k] for k in PROFILE_KEYS],
                show_progress=False,
            )

            load_inputs = [llm_components[k] for k in
                           ['model_name'] + PROFILE_KEYS + ['llama_server_path']]
            load_btn.click(fn=_load_model, inputs=load_inputs, outputs=[status_display])
            unload_btn.click(fn=_unload_model, outputs=[status_display])

            #   Warn as soon as the two sliders make an over-long review turn likely,
            #   rather than after the model has already been loaded and fed.
            for component in (max_review_images, n_ctx):
                component.release(
                    fn=_warn_on_context_budget,
                    inputs=[max_review_images, n_ctx],
                    outputs=[],
                    show_progress=False,
                )

        panel = create_img2img_panel(
            "llm2img", scripts.scripts_llm2img, toprow, extra_ui=build_llm_ui
        )
        output_panel = panel.output_panel

        #   submit_inputs[0] is the id_task slot; the rest line up with IMG2IMG_ARG_NAMES.
        img_components = dict(zip(IMG2IMG_ARG_NAMES, panel.submit_inputs[1:]))

        #   Restore the image settings this tab saved last time, and take them out of
        #   ui-config's hands so they are not reset to the hardcoded defaults on restart.
        for name in SAVEABLE_IMG_ARGS:
            component = img_components[name]
            _own_the_value(component)
            if name in saved_image:
                restored = _restorable(component, saved_image[name])
                if restored is not None:
                    component.value = restored

        for name in llm2img.LLM_ARG_NAMES:
            _own_the_value(llm_components[name])

        llm_components['_save_btn'].click(
            fn=save_settings,
            inputs=(
                [llm_components[name] for name in llm2img.LLM_ARG_NAMES]
                + [img_components[name] for name in SAVEABLE_IMG_ARGS]
            ),
            outputs=[llm_components['_save_status']],
            show_progress=False,
        )

        def run(id_task, request: gr.Request, *args):
            yield from llm2img.run_with_job(id_task, request, scripts.scripts_llm2img, *args)

        run_args = dict(
            fn=run,
            _js="submit_llm2img",
            inputs=(
                [panel.dummy_component]
                + [llm_components[name] for name in llm2img.LLM_ARG_NAMES]
                + panel.submit_inputs[1:]
            ),
            #   Order matters: submit_llm2img() in javascript/ui.js identifies the echoed
            #   outputs by finding the gallery array six places from the end.
            outputs=[
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
                llm_components['_round_log'],
                llm_components['_current_prompt'],
            ],
            show_progress=False,
        )

        toprow.prompt.submit(**run_args)
        toprow.submit.click(**run_args)

        #   The prompt box is the loop's memory between Runs: whatever the last review
        #   produced becomes the starting prompt for the next one.
        llm_components['_current_prompt'].change(
            fn=lambda p: gr.update(value=p) if p else gr.update(),
            inputs=[llm_components['_current_prompt']],
            outputs=[toprow.prompt],
            show_progress=False,
        )

        try:
            extra_networks_ui = ui_extra_networks.create_ui(llm2img_interface, [panel.generation_tab], 'llm2img')
            ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)
        except Exception as e:
            #   The card UI is keyed by tabname and not every extra-networks page copes
            #   with a third one. LoRAs still work through <lora:...> prompt syntax.
            print(f"[llm2img] Extra networks UI unavailable on this tab: {e}")

        extra_tabs.__exit__()

    scripts.scripts_current = None
    return llm2img_interface


def _apply_model_profile(model_name):
    """Restore the load settings that last worked for this model, or the recommended ones."""
    import vlm_llamacpp

    profiles = vlm_llamacpp.load_model_profiles()
    profile = profiles.get(model_name) or vlm_llamacpp.get_recommended_profile(model_name) or {}
    return [gr.update(value=profile[k]) if k in profile else gr.update() for k in PROFILE_KEYS]


def _load_model(model_name, *values):
    """Start llama-server, then remember the settings that worked for this model."""
    import vlm_llamacpp

    cfg = dict(zip(PROFILE_KEYS, values[:len(PROFILE_KEYS)]))
    cfg['model_name'] = model_name
    cfg['llama_server_path'] = values[len(PROFILE_KEYS)]

    if not model_name or model_name.startswith("No GGUF models"):
        return "Select a GGUF model first"

    status = llm2img.start_server(cfg)

    if isinstance(status, str) and not status.startswith("Error"):
        profile = {k: cfg[k] for k in PROFILE_KEYS}
        profile['backend_type'] = "llama-server"
        vlm_llamacpp.save_model_profile(model_name, profile)
        vram = vlm_llamacpp.get_vram_info()
        if vram:
            status = f"{status} | VRAM: {vram}"

    return status


def _unload_model():
    status = llm2img.stop_server()
    import vlm_llamacpp
    vram = vlm_llamacpp.get_vram_info()
    return f"{status} | VRAM: {vram}" if vram else status


def _warn_on_context_budget(max_review_images, n_ctx):
    needed = int(max_review_images) * TOKENS_PER_IMAGE_ESTIMATE
    if needed > int(n_ctx):
        gr.Warning(
            f"{int(max_review_images)} images is roughly {needed:,} vision tokens, more than "
            f"the {int(n_ctx):,}-token context. Raise Context Length or lower Max review images."
        )
