from __future__ import annotations
import base64
import io
import time

import gradio as gr
from pydantic import BaseModel, Field

# Import shared at the top to make its availability clear
from modules import shared # This is where shared.opts should come from

# from modules.shared import opts # This is also valid but 'shared.opts' is more explicit below

import modules.shared as shared_explicit_for_clarity # Using a different alias for clarity in the check
from collections import OrderedDict
import string
import random
from typing import List

current_task = None
pending_tasks = OrderedDict()
finished_tasks = []
recorded_results = []
recorded_results_limit = 2


def start_task(id_task):
    global current_task

    current_task = id_task
    pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)

def create_task_id(task_type):
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase +
    string.digits, k=N))
    return f"task({task_type}-{res})"

def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def add_task_to_queue(id_job):
    pending_tasks[id_job] = time.time()

class PendingTasksResponse(BaseModel):
    size: int = Field(title="Pending task size")
    tasks: List[str] = Field(title="Pending task ids")

class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")


class ProgressResponse(BaseModel):
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: float | None = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float | None = Field(default=None, title="ETA in secs")
    live_preview: str | None = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int | None = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: str | None = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


def get_pending_tasks():
    pending_tasks_ids = list(pending_tasks)
    pending_len = len(pending_tasks_ids)
    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


def progressapi(req: ProgressRequest): # req is allowed to be None by Gradio if no body is sent, but Pydantic might enforce it.
                                      # However, the default for ProgressRequest fields suggests it can be an empty request.
                                      # For safety, FastAPI treats `req: ProgressRequest` (no `= None`) as required body.
                                      # If it can be optional, it should be `req: ProgressRequest | None = None`.
                                      # Given the endpoint is POST, a body is usually expected.
                                      # The `req.live_preview` access implies `req` is not None.

    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks

    if not active:
        textinfo = "Waiting..."
        if queued:
            # Sort pending_tasks by their timestamp (value) to get a stable queue order
            # pending_tasks is an OrderedDict, so list(pending_tasks) preserves insertion order.
            # If you want to sort by time added (value in pending_tasks dict):
            # sorted_queued_by_time = sorted(pending_tasks.items(), key=lambda item: item[1])
            # sorted_queued_ids = [item[0] for item in sorted_queued_by_time]
            # For now, let's assume insertion order is sufficient for queue display
            
            # Using insertion order from OrderedDict directly for queue position
            queue_ids_in_order = list(pending_tasks.keys())
            try:
                queue_index = queue_ids_in_order.index(req.id_task)
                textinfo = "In queue: {}/{}".format(queue_index + 1, len(queue_ids_in_order))
            except ValueError: # Should not happen if req.id_task in pending_tasks is true
                textinfo = "In queue (error finding position)"

        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    progress = 0.0 # Ensure progress is float

    # Access shared.state attributes safely
    job_count = getattr(shared.state, 'job_count', 0)
    job_no = getattr(shared.state, 'job_no', 0)
    sampling_steps = getattr(shared.state, 'sampling_steps', 0)
    sampling_step = getattr(shared.state, 'sampling_step', 0)
    time_start = getattr(shared.state, 'time_start', time.time()) # Default to now if not set

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0: # Ensure job_count is not zero for division
        progress += (1 / job_count) * (sampling_step / sampling_steps)

    progress = min(progress, 1.0) # Ensure it's float and capped

    elapsed_since_start = time.time() - time_start
    predicted_duration = (elapsed_since_start / progress) if progress > 1e-6 else None # Avoid division by zero or tiny progress
    eta = (predicted_duration - elapsed_since_start) if predicted_duration is not None else None

    live_preview = None
    id_live_preview = req.id_live_preview # Start with the ID from the request

    # --- MODIFIED SECTION FOR ROBUST shared.opts ACCESS ---
    live_preview_check_passed = False
    if hasattr(shared_explicit_for_clarity, 'opts') and shared_explicit_for_clarity.opts is not None:
        if hasattr(shared_explicit_for_clarity.opts, 'live_previews_enable') and shared_explicit_for_clarity.opts.live_previews_enable:
            # Now check req and its attributes, assuming req is a valid ProgressRequest object
            if req is not None and hasattr(req, 'live_preview') and req.live_preview:
                live_preview_check_passed = True
    else:
        # This is where the critical error happens if shared.opts is None
        print(f"CRITICAL ERROR in progressapi at {time.strftime('%Y-%m-%d %H:%M:%S')}: shared.opts is None! Cannot check live_previews_enable.")
        # live_preview_check_passed remains False, so live preview will be skipped.
    # --- END OF MODIFIED SECTION ---

    if live_preview_check_passed:
        shared.state.set_current_image() # This might also access shared.opts internally
        if shared.state.id_live_preview != req.id_live_preview: # Compare with the ID from the request
            image = shared.state.current_image
            if image is not None:
                buffered = io.BytesIO()
                
                # Safe access to opts for live_previews_image_format
                preview_format = 'jpeg' # Default format
                if hasattr(shared_explicit_for_clarity, 'opts') and shared_explicit_for_clarity.opts is not None and \
                   hasattr(shared_explicit_for_clarity.opts, 'live_previews_image_format'):
                    preview_format = shared_explicit_for_clarity.opts.live_previews_image_format

                save_kwargs = {}
                if preview_format == "png":
                    if max(*image.size) <= 256: # Small optimization
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}
                # No specific kwargs for jpeg needed here, Pillow defaults are usually fine.

                try:
                    image.save(buffered, format=preview_format.upper(), **save_kwargs) # Ensure format is uppercase for Pillow
                    base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                    live_preview = f"data:image/{preview_format};base64,{base64_image}"
                    id_live_preview = shared.state.id_live_preview # Update with the new ID from shared.state
                except Exception as e:
                    print(f"Error saving live preview image: {e}")
                    live_preview = None # Don't send a broken preview
                    # id_live_preview remains the one from the request, so client won't re-request same broken state
                    # Or, set id_live_preview = shared.state.id_live_preview to acknowledge processing attempt

    return ProgressResponse(active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)


def restore_progress(id_task):
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)

    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"