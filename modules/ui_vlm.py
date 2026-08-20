"""
The VLM tab: vlm.py's own Gradio interface, rendered inside the webui.

vlm.py stays a standalone app and never imports anything from modules.* or backend.*.
Everything forge-specific it needs - the generation queue, freeing the diffusion model's
VRAM - is handed to it through the ForgeHooks object installed on `vlm.FORGE` below, which
is None when vlm.py is run on its own.
"""

import functools
import os
import sys
from contextlib import contextmanager

from modules import call_queue, errors, progress, shared
from modules.paths_internal import script_path


class ForgeHooks:
    """The webui side of vlm.py's optional integration points."""

    @contextmanager
    def job(self):
        """One slot in the generation queue, with the job bookkeeping the UI expects.

        Deliberately not wrap_gradio_gpu_call: that funnels the result through
        wrap_gradio_call_no_job, which does `list(func(...))` and so cannot stream a
        generator. vlm.py's chat streams token by token, so this mirrors the structure of
        modules.llm2img.run_with_job instead.

        id_task is None because vlm.py's buttons are plain gr.Button clicks and never emit
        the "task(...)" id the JS side generates; progress.start_task tolerates that.
        """
        id_task = None
        with call_queue.queue_lock:
            shared.state.begin(job=id_task)
            progress.start_task(id_task)
            try:
                yield
            finally:
                progress.finish_task(id_task)
                shared.state.end()
                shared.state.skipped = False
                shared.state.interrupted = False

    def queued(self, fn):
        """Serialise a plain handler against the queue.

        functools.wraps rather than call_queue.wrap_queued_call so the wrapper keeps the
        original signature - Gradio reads it to decide where to inject the progress
        tracker, and an opaque (*args, **kwargs) wrapper silently loses it.
        """

        @functools.wraps(fn)
        def f(*args, **kwargs):
            with call_queue.queue_lock:
                return fn(*args, **kwargs)

        return f

    def unload_diffusion(self):
        """Hand the diffusion model's VRAM over to the LLM."""
        from modules import llm2img

        #   LLM2img keeps its own LlamaCppVLM over its own copy of vlm.py, and both default
        #   to port 8080. The queue lock stops the two tabs running at once, but LLM2img's
        #   "Keep LLM loaded" option can leave its server alive past the end of its run, and
        #   then our llama-server cannot bind. Best effort: it is normal for this to be a
        #   no-op.
        try:
            llm2img.stop_server()
        except Exception:
            errors.report("VLM tab: could not stop the LLM2img llama-server", exc_info=True)

        llm2img.unload_diffusion_models()


def _disown_values(interface):
    """Keep ui-config.json out of the VLM tab's widgets.

    modules/ui.py calls loadsave.add_block(interface, "vlm"), and UiLoadsave.add_component()
    does `setattr(obj, 'value', saved_value)` for everything it tracks - which would clobber
    whatever vlm_settings.json restored with whatever ui-config recorded on the first run.
    Same trick as _own_the_value() in modules/ui_llm2img.py.
    """
    for component in interface.blocks.values():
        component.do_not_save_to_config = True


def create_vlm_interface():
    """Build the VLM tab. Returns the gr.Blocks for modules.ui to register."""

    #   vlm.py lives at the repo root next to webui.py. That is normally already on
    #   sys.path, but launch paths differ; modules/llm2img.py guards the same way.
    if script_path not in sys.path:
        sys.path.insert(0, script_path)

    import vlm

    #   create_ui() reads the model list straight off the manager, and falls back to a dead
    #   "Initialize manager first" placeholder if there is none.
    vlm.initialize_manager(os.path.join(script_path, "models", "LLM"))
    vlm.FORGE = ForgeHooks()

    interface = vlm.create_ui(nested=True)
    _disown_values(interface)
    return interface
