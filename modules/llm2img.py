"""
LLM2img: prompt -> batch -> review loop.

One round is:

    unload diffusion -> start llama-server -> LLM writes / holds the prompt -> stop
    llama-server -> webui generates a batch -> restart llama-server -> the VLM reviews the
    whole batch and returns a critique plus the prompt for the next round

The LLM and the diffusion model never sit in VRAM at the same time, which is the whole
point of the tab: on a single GPU a 27B VLM and a diffusion checkpoint do not both fit.

The llama.cpp side is vlm_llamacpp.py at the repo root - a verbatim copy of vlm.py from the
`vlm` branch, so it can be re-synced with a plain `cp`. Only the llama-server backend is
used here; the in-process llama-cpp-python path is deliberately not wired up, because
killing a subprocess is the only way to be certain the VRAM came back.
"""

import json
import os
import re
import sys
import time
from contextlib import closing

import gradio as gr

from modules import call_queue, progress, shared
from modules.paths_internal import script_path
from modules.shared import opts
from modules.ui_img2img_panel import IMG2IMG_ARG_NAMES, arg_index
from modules_forge import main_thread

#   vlm_llamacpp.py lives at the repo root next to webui.py, which is already on sys.path
#   when the webui runs. Guarded so a missing/broken copy disables the tab instead of
#   taking the whole UI down.
try:
    if script_path not in sys.path:
        sys.path.insert(0, script_path)
    import vlm_llamacpp
    VLM_AVAILABLE = True
    VLM_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on the checkout
    vlm_llamacpp = None
    VLM_AVAILABLE = False
    VLM_IMPORT_ERROR = e


#   Order of the LLM-side values in the event's input list. modules.ui_llm2img builds its
#   component list from this same list, so the two cannot drift.
LLM_ARG_NAMES = [
    # llama-server load settings (mirrors vlm.py's Model Settings accordion)
    'model_name', 'n_gpu_layers', 'n_ctx', 'tensor_split', 'main_gpu', 'kv_cache_type',
    'flash_attn', 'use_mmap', 'use_mlock', 'override_tensor', 'extra_args', 'server_port',
    'llama_server_path',
    # sampling settings (mirrors vlm.py's Generation Settings accordion; vlm.py's single
    # System Prompt is split into the writer/reviewer pair below, since the two turns of a
    # round want different instructions)
    'max_tokens', 'temperature', 'top_p', 'repeat_penalty', 'seed',
    'video_max_frames', 'every_other_frame', 'show_thinking', 'reasoning_level',
    'thinking_mode',
    # the loop itself
    'idea', 'writer_system_prompt', 'reviewer_system_prompt', 'rounds',
    'max_review_images', 'feed_forward', 'unload_diffusion', 'keep_llm_loaded',
]

N_LLM_INPUTS = len(LLM_ARG_NAMES)

#   The review turn sends every image of the batch in one request. vlm.py's chat tab caps
#   this at 10 for interactive use; here the cap is the user's own "Max review images"
#   slider, bounded by this.
MAX_REVIEW_IMAGES = 100

#   Minimum wall-clock gap between pushes of a partial LLM reply to the browser. The model
#   produces ~60 tokens/s; a yield per token means a round trip per token, and the browser
#   cannot render that fast anyway.
STREAM_PUSH_INTERVAL = 0.15

#   Shown whenever a reply arrives without a usable prompt in it. On a thinking model the
#   reasoning and the answer come out of the same Max New Tokens budget, so a budget that
#   looks generous can still leave the answer cut off a few words in.
_TRUNCATION_HINT = (
    "If the reply in the console stops mid-sentence, raise **Max New Tokens** — on a "
    "thinking model the reasoning eats the same budget as the answer."
)


DEFAULT_WRITER_SYSTEM_PROMPT = """You write prompts for a text-to-image diffusion model.

Given the user's idea, produce ONE prompt that is vivid and concrete: subject, action, \
setting, composition, lighting, colour, lens/medium, and style. Prefer specific nouns and \
adjectives over abstractions. Do not include negative-prompt terms, weights, or LoRA tags.

Reply with the prompt and nothing else, wrapped in a fenced code block:

```
<the prompt>
```"""

DEFAULT_REVIEWER_SYSTEM_PROMPT = """You are reviewing a batch of images that a \
text-to-image model generated from a prompt, in order to improve that prompt.

Look at every image. Judge how well the batch realises the prompt's intent, and note what \
went wrong: missing or wrong subject matter, anatomy and structure errors, muddy \
composition, flat or inconsistent lighting, wrong style, artefacts, sameness across the \
batch.

Reply in exactly this shape, and always finish with the fenced block:

CRITIQUE:
<three to six short bullet points on what this batch got right and wrong>

```
<the full revised prompt, self-contained, ready to paste into the model>
```

Keep the critique brief — the fenced prompt is the part that matters, and a reply that \
runs out of room before reaching it is wasted. The revised prompt must be a complete \
prompt, not a diff or a list of changes, and it must still deliver the user's original \
idea. Keep what worked, fix what did not."""


def _manager():
    """The one LlamaCppVLM instance for this process, created on first use."""
    global _vlm_manager
    if _vlm_manager is None:
        if not VLM_AVAILABLE:
            raise RuntimeError(f"vlm_llamacpp.py could not be imported: {VLM_IMPORT_ERROR}")
        models_dir = os.path.join(script_path, "models", "LLM")
        _vlm_manager = vlm_llamacpp.LlamaCppVLM(models_dir)
    return _vlm_manager


_vlm_manager = None


def get_manager():
    """Public accessor; returns None instead of raising when llama.cpp support is absent."""
    if not VLM_AVAILABLE:
        return None
    return _manager()


def get_model_names():
    manager = get_manager()
    if manager is None:
        return ["vlm_llamacpp.py not available"]
    names = manager.get_model_names()
    return names or ["No GGUF models found in models/LLM"]


def server_status():
    """Status line for the tab, matching vlm.py's own status box."""
    manager = get_manager()
    if manager is None:
        return f"llama.cpp backend unavailable: {VLM_IMPORT_ERROR}"
    status = manager.get_status()
    vram = vlm_llamacpp.get_vram_info()
    return f"{status} | VRAM: {vram}" if vram else status


# ---------------------------------------------------------------------------- lifecycle


def unload_diffusion_models():
    """Free the diffusion model's VRAM so the LLM has somewhere to live.

    Same call modules/ui.py uses before the transformers prompt expander; the next
    generation reloads the checkpoint on demand. Runs on the main thread because it
    reaches into shared.sd_model and the forge object graph.
    """
    from backend import memory_management
    from modules import processing

    def work():
        memory_management.emergency_memory_cleanup()
        processing.need_global_unload = True

    main_thread.run_and_wait_result(work)


def start_server(cfg, progress_cb=gr.Progress()):
    """Start llama-server for the configured model. Returns the status string."""
    manager = _manager()

    if manager.server_process is not None:
        return manager.get_status()

    path = (cfg.get('llama_server_path') or "").strip()
    if path:
        manager.llama_server_path = path

    kv = cfg.get('kv_cache_type')
    #   f16 is llama.cpp's own default; passing it explicitly is harmless but noisy, and
    #   vlm.py's handlers treat it as "leave alone".
    type_k = type_v = kv if kv and kv != "f16" else None

    return manager.load_model_server(
        model_name=cfg['model_name'],
        n_gpu_layers=int(cfg['n_gpu_layers']),
        n_ctx=int(cfg['n_ctx']),
        tensor_split=cfg.get('tensor_split'),
        flash_attn=bool(cfg.get('flash_attn')),
        main_gpu=int(cfg.get('main_gpu') or 0),
        type_k=type_k,
        type_v=type_v,
        use_mmap=bool(cfg.get('use_mmap')),
        use_mlock=bool(cfg.get('use_mlock')),
        override_tensor=cfg.get('override_tensor'),
        server_port=int(cfg.get('server_port') or 8080),
        extra_args=cfg.get('extra_args'),
        progress=progress_cb,
    )


def stop_server():
    manager = get_manager()
    if manager is None:
        return "llama.cpp backend unavailable"
    return manager.unload_model()


# ------------------------------------------------------------------------------ the LLM


def _run_llm(cfg, messages):
    """Stream one completion, yielding the text as it grows.

    With Show Thinking on, the reasoning block comes through as well; extract_prompt()
    strips it again before the prompt is used.
    """
    manager = _manager()
    show_thinking = bool(cfg['show_thinking'])

    text = ""
    for display, raw, _stats, _ctx in manager.generate(
        messages=messages,
        max_new_tokens=int(cfg['max_tokens']),
        temperature=float(cfg['temperature']),
        top_p=float(cfg['top_p']),
        repeat_penalty=float(cfg['repeat_penalty']),
        seed=int(cfg['seed']),
        video_max_frames=int(cfg['video_max_frames']),
        every_other_frame=bool(cfg['every_other_frame']),
        stream=True,
        reasoning_level=cfg['reasoning_level'],
        thinking=bool(cfg['thinking_mode']),
    ):
        text = vlm_llamacpp.format_think_tags_for_display(raw) if show_thinking else display
        yield text


_FENCE_RE = re.compile(r"```(?:[\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)
#   DOTALL so a multi-line prompt after the label is taken whole - anchoring on $ instead
#   would silently keep only its first line.
_LABEL_RE = re.compile(r"^[ \t]*(?:revised |final |new |updated )?prompt\s*:[ \t]*\n?(.+)\Z",
                       re.IGNORECASE | re.MULTILINE | re.DOTALL)
_CRITIQUE_RE = re.compile(r"^\s*(?:critique|review|analysis|notes)\s*:", re.IGNORECASE)


def extract_prompt(text):
    """Pull the image prompt out of a model reply, or "" when there isn't one.

    Returning "" rather than guessing is the point. There used to be a "just take the last
    paragraph" fallback, and when a reply was cut short - a thinking model spends the same
    Max New Tokens budget on its reasoning and its answer, so the answer is what gets
    truncated - that fallback happily handed back a fragment of the critique and the next
    round generated images from "CRITIQUE:\\n- Style and". The caller now keeps the prompt
    it already had and says so.
    """
    if not text:
        return ""

    text = vlm_llamacpp.strip_reasoning_for_context(text)

    def acceptable(candidate):
        candidate = (candidate or "").strip().strip("`").strip()
        if not candidate or _CRITIQUE_RE.match(candidate):
            return None
        return candidate

    #   Last fence: a chatty model puts the critique first and the prompt last.
    for fence in reversed(_FENCE_RE.findall(text)):
        found = acceptable(fence)
        if found:
            return found

    labelled = _LABEL_RE.search(text)
    if labelled:
        found = acceptable(labelled.group(1))
        if found:
            return found

    return ""


def _report_unparsed(kind, text):
    """Say why a round could not move on, in the console where the reply is visible."""
    tail = (text or "").strip()[-400:]
    print(f"[llm2img] No prompt found in the {kind} reply ({len(text or '')} chars). "
          f"Tail of the reply:\n...{tail}")


def _build_review_message(idea, prompt_used, images, limit):
    """One user turn holding the batch, the prompt that produced it, and the original idea.

    The idea has to be repeated every round. The reviewer only ever sees one prompt and one
    batch, so without it each revision is judged against the previous revision and the run
    drifts away from what was actually asked for.
    """
    content = []
    for image in images[:limit]:
        content.append({"type": "image", "image": image})

    shown = min(len(images), limit)
    note = ""
    if len(images) > shown:
        note = f" (showing the first {shown} of {len(images)})"

    goal = ""
    if idea and idea.strip():
        goal = ("The user asked for this, and every revision must still serve it:\n\n"
                f"{idea.strip()}\n\n")

    content.append({"type": "text", "text": (
        f"{goal}"
        f"These {shown} image(s){note} were generated from this prompt:\n\n"
        f"{prompt_used}\n\n"
        "Review the batch and give me the revised prompt."
    )})
    return {"role": "user", "content": content}


def _messages(system_prompt, *turns):
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(turns)
    return messages


# ------------------------------------------------------------------------ image batches


def _pil_images(value):
    """Normalise a ForgeCanvas background / gr.Image value to a PIL image or None."""
    if value is None:
        return None
    if hasattr(value, "convert"):
        return value
    if isinstance(value, dict):
        return value.get("image") or value.get("background")
    return None


def _generate_batch(id_task, request, img_args, script_runner):
    """Run one batch and return (images, generation_info_js, infotext_html, html_log).

    Uses img2img when the selected mode has an image to work from, and plain txt2img when
    it does not - which is the normal case for round one.
    """
    import modules.img2img

    mode = int(img_args[arg_index('mode')] or 0)
    has_source = _source_image_for_mode(img_args, mode) is not None

    if has_source:
        return main_thread.run_and_wait_result(
            modules.img2img.img2img_function, id_task, request, *img_args
        )

    return main_thread.run_and_wait_result(
        _txt2img_from_img2img_args, id_task, request, img_args, script_runner
    )


def _source_image_for_mode(img_args, mode):
    """Whichever image the selected img2img mode would start from, or None."""
    by_mode = {
        0: 'init_img',
        1: 'ref_edit_main_img',
        2: 'sketch',
        3: 'init_img_with_mask',
        4: 'inpaint_color_sketch',
        5: 'init_img_inpaint',
    }
    name = by_mode.get(mode)
    if name is None:
        #   mode 6 is Batch, which reads from a directory or an upload rather than a
        #   canvas; leave it to img2img_function to validate.
        return True if mode == 6 else None
    return _pil_images(img_args[arg_index(name)])


def _txt2img_from_img2img_args(id_task, request, img_args, script_runner):
    """txt2img fallback built from the LLM2img panel's values.

    The panel is an img2img panel, so there is no hires-fix section to read - hires is off.
    Everything else (size, batch, CFG, styles, override settings, checkpoint, script args)
    comes straight from the tab.
    """
    from modules import processing
    from modules.infotext_utils import create_override_settings_dict
    from modules.ui import plaintext_to_html

    def value(name):
        return img_args[arg_index(name)]

    p = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=value('prompt'),
        styles=value('prompt_styles'),
        negative_prompt=value('negative_prompt'),
        batch_size=int(value('batch_size')),
        n_iter=int(value('n_iter')),
        cfg_scale=value('cfg_scale'),
        distilled_cfg_scale=value('distilled_cfg_scale'),
        zimage_shift=value('zimage_shift'),
        width=int(value('width')),
        height=int(value('height')),
        enable_hr=False,
        denoising_strength=value('denoising_strength'),
        override_settings=create_override_settings_dict(value('override_settings_texts')),
        checkpoint_override=shared.opts.sd_model_checkpoint,
    )

    p.scripts = script_runner
    p.script_args = tuple(img_args[len(IMG2IMG_ARG_NAMES):])
    p.user = request.username if request is not None else None

    with closing(p):
        processed = script_runner.run(p, *p.script_args)
        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    return (
        processed.images + processed.extra_images,
        processed.js(),
        plaintext_to_html(processed.info),
        plaintext_to_html(processed.comments, classname="comments"),
    )


def _merge_generation_info(accumulated, round_js, n_images):
    """Concatenate this round's infotexts onto the running total so gallery index N still
    maps to infotext N when several rounds share one gallery."""
    try:
        data = json.loads(round_js) if round_js else {}
    except (TypeError, ValueError):
        data = {}

    infotexts = list(data.get("infotexts") or [])
    #   extra_images are appended to processed.images but carry no infotext of their own.
    infotexts += [""] * max(0, n_images - len(infotexts))

    merged = dict(data)
    merged["infotexts"] = list(accumulated.get("infotexts") or []) + infotexts[:n_images]
    merged["index_of_first_image"] = accumulated.get("index_of_first_image", data.get("index_of_first_image", 0))
    return merged


# ------------------------------------------------------------------------------- the run


def _interrupted():
    return shared.state.interrupted or shared.state.stopping_generation


def llm2img_run(id_task, request, script_runner, *args):
    """The round loop. A generator, so the round log and gallery fill in as it goes.

    Yields (gallery, generation_info_js, infotext_html, html_log, round_log, prompt).
    """
    cfg = dict(zip(LLM_ARG_NAMES, args[:N_LLM_INPUTS]))
    img_args = list(args[N_LLM_INPUTS:])

    gallery = []
    geninfo = {}
    infotext_html = ""
    html_log = ""
    log_lines = []
    prompt = ""

    def snapshot(extra=None):
        body = "\n\n".join(log_lines + ([extra] if extra else []))
        return gallery, json.dumps(geninfo), infotext_html, html_log, body, prompt

    def streaming_snapshot(extra):
        """A yield that only advances the round log.

        Everything else is gr.update(), i.e. "leave this component alone". Returning the
        gallery here instead costs about 100ms per token, because gradio re-encodes every
        PIL image in it on each yield - enough to turn a 26s review into a 181s one.
        """
        keep = gr.update()
        body = "\n\n".join(log_lines + ([extra] if extra else []))
        return keep, keep, keep, keep, body, keep

    reply = ""

    def stream_llm(system_prompt, *turns):
        """Run one LLM turn, pushing the partial reply into the round log as it arrives.

        Leaves the finished text in `reply`. Pushes are rate-limited: the model emits ~60
        tokens a second and the browser cannot make use of more than a few frames of that.
        """
        nonlocal reply
        reply = ""
        last_push = 0.0
        for reply in _run_llm(cfg, _messages(system_prompt, *turns)):
            now = time.perf_counter()
            if now - last_push >= STREAM_PUSH_INTERVAL:
                last_push = now
                yield streaming_snapshot(_quoted(reply))
            if _interrupted():
                break

    if not VLM_AVAILABLE:
        log_lines.append(f"**LLM backend unavailable.** `vlm_llamacpp.py` failed to import: `{VLM_IMPORT_ERROR}`")
        yield snapshot()
        return

    manager = _manager()
    rounds = max(1, int(cfg['rounds']))
    review_limit = max(1, min(MAX_REVIEW_IMAGES, int(cfg['max_review_images'])))
    prompt = (img_args[arg_index('prompt')] or "").strip()

    started = time.perf_counter()

    try:
        for round_no in range(1, rounds + 1):
            if _interrupted():
                log_lines.append(f"_Interrupted before round {round_no}._")
                yield snapshot()
                return

            log_lines.append(f"### Round {round_no} of {rounds}")
            yield snapshot()

            # -- 1+2. the prompt for this round -----------------------------------------
            #   Only round 1 needs the LLM here; later rounds already hold the prompt the
            #   previous round's review produced, so starting the server would just cost a
            #   load and an unload for nothing.
            needs_writer = round_no == 1 and cfg['idea'] and cfg['idea'].strip()

            if needs_writer:
                if cfg['unload_diffusion']:
                    unload_diffusion_models()

                status = start_server(cfg)
                if isinstance(status, str) and status.startswith("Error"):
                    log_lines.append(f"**Could not start llama-server:** {status}")
                    yield snapshot()
                    return

                log_lines.append("**Writing the prompt…**")
                yield from stream_llm(
                    cfg['writer_system_prompt'],
                    {"role": "user", "content": cfg['idea'].strip()},
                )
                written = extract_prompt(reply)
                if written:
                    prompt = written
                    log_lines[-1] = f"**Prompt**\n\n```\n{prompt}\n```"
                else:
                    _report_unparsed("prompt-writer", reply)
                    log_lines[-1] = (
                        "**No prompt found in the reply** — using whatever is in the prompt "
                        f"box instead. {_TRUNCATION_HINT}"
                    )
            elif round_no == 1:
                log_lines.append(
                    "No idea given, so round 1 uses the prompt already in the prompt box."
                )

            if not prompt.strip():
                log_lines.append("**Nothing to generate:** no idea and an empty prompt box.")
                yield snapshot()
                return

            yield snapshot()

            if _interrupted():
                log_lines.append("_Interrupted._")
                yield snapshot()
                return

            # -- 3. hand the VRAM back to the diffusion model ---------------------------
            if not cfg['keep_llm_loaded']:
                stop_server()
            #   Round 2 onward never started the server above, so nothing to stop.

            # -- 4. generate -------------------------------------------------------------
            img_args[arg_index('prompt')] = prompt
            log_lines.append("**Generating…**")
            yield snapshot()

            images, round_js, infotext_html, html_log = _generate_batch(
                id_task, request, img_args, script_runner
            )
            geninfo = _merge_generation_info(geninfo, round_js, len(images))
            gallery = gallery + list(images)
            log_lines[-1] = f"**Generated {len(images)} image(s).**"
            yield snapshot()

            if _interrupted():
                log_lines.append("_Interrupted after generating; skipping the review._")
                yield snapshot()
                return

            # -- 5. review ---------------------------------------------------------------
            #   The last round is reviewed too: its critique and revised prompt are the
            #   run's actual output, ready for the next Run.
            if cfg['unload_diffusion']:
                unload_diffusion_models()

            status = start_server(cfg)
            if isinstance(status, str) and status.startswith("Error"):
                log_lines.append(f"**Could not restart llama-server for the review:** {status}")
                yield snapshot()
                return

            if manager.is_text_only_model:
                log_lines.append(
                    "**No vision projector loaded** - the review will be text-only. Put an "
                    "`mmproj-*.gguf` next to the weights to let the model actually see the batch."
                )

            log_lines.append(f"**Reviewing {min(len(images), review_limit)} image(s)…**")
            yield from stream_llm(
                cfg['reviewer_system_prompt'],
                _build_review_message(cfg['idea'], prompt, list(images), review_limit),
            )

            log_lines[-1] = f"**Review**\n\n{reply.strip()}"
            revised = extract_prompt(reply)
            if revised:
                prompt = revised
            else:
                _report_unparsed("review", reply)
                log_lines.append(
                    "**No revised prompt in that review** — keeping the current prompt for "
                    f"the next round. {_TRUNCATION_HINT}"
                )
            yield snapshot()

            if not cfg['keep_llm_loaded']:
                stop_server()

            if cfg['feed_forward'] and images:
                #   Later rounds refine the first image of the batch rather than starting
                #   over: switch to plain img2img and hand it that image.
                img_args[arg_index('mode')] = 0
                img_args[arg_index('init_img')] = images[0]

        elapsed = time.perf_counter() - started
        log_lines.append(f"_Finished {rounds} round(s) in {elapsed / 60:.1f} min._")
        yield snapshot()

    finally:
        if not cfg.get('keep_llm_loaded'):
            stop_server()


def _quoted(text):
    """Render a partial LLM reply as a blockquote in the round log."""
    if not text:
        return ""
    return "\n".join("> " + line for line in text.strip().split("\n"))


def run_with_job(id_task, request, script_runner, *args):
    """Wrap llm2img_run in the same job bookkeeping wrap_gradio_gpu_call does.

    Not wrap_gradio_gpu_call itself: that funnels the result through
    wrap_gradio_call_no_job, which does `list(func(...))` and so cannot stream a
    generator. A multi-round run is minutes long, so streaming the round log matters more
    here than sharing the wrapper.
    """
    if isinstance(id_task, str) and id_task.startswith("task(") and id_task.endswith(")"):
        progress.add_task_to_queue(id_task)
    else:
        id_task = None

    with call_queue.queue_lock:
        shared.state.begin(job=id_task)
        progress.start_task(id_task)
        try:
            yield from llm2img_run(id_task, request, script_runner, *args)
        except Exception as e:
            import html as html_module
            import traceback
            traceback.print_exc()
            message = f"{type(e).__name__}: {e}"
            yield [], "{}", "", f"<div class='error'>{html_module.escape(message)}</div>", \
                f"**Run failed:** `{message}`", ""
        finally:
            progress.finish_task(id_task)
            shared.state.end()
            shared.state.skipped = False
            shared.state.interrupted = False
            shared.state.stopping_generation = False
            shared.state.job_count = 0
            shared.state.job = ""
