# --- START OF FILE progress.py (MODIFIED ORIGINAL) ---
from __future__ import annotations # Keep if your Python version needs it
import base64
import io
import time

import gradio as gr
from pydantic import BaseModel, Field

from modules.shared import opts # Assuming this import is correct for your env

import modules.shared as shared # Assuming this import is correct
from collections import OrderedDict
import string
import random
from typing import List, Optional # Optional is good practice, or use | None

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
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
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
    # FIX: Use | None or Optional[] for fields that can be None
    progress: float | None = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float | None = Field(default=None, title="ETA in secs")
    live_preview: str | None = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int | None = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: str | None = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    # Ensure response_model matches the actual return possibilities
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


def get_pending_tasks():
    pending_tasks_ids = list(pending_tasks)
    pending_len = len(pending_tasks_ids)
    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


def progressapi(req: ProgressRequest):
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks
    #print(f"PYTHON progressapi: Request for id_task='{req.id_task}'. Current pending_tasks: {list(pending_tasks.keys())}. Current current_task: {current_task}. Timestamp: {time.time()}")
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks # Check membership *before* any potential modifications
    completed = req.id_task in finished_tasks

    # Initialize all potentially None fields for the response
    # This makes it clearer what's being returned, especially for the non-active case
    current_progress = None
    current_eta = None
    current_live_preview = None
    current_id_live_preview = req.id_live_preview # Start with requested id
    current_textinfo = None

    if not active:
        current_textinfo = "In queue :>"
        if queued:
            # Ensure req.id_task is actually in pending_tasks before calling index
            if req.id_task in pending_tasks:
                sorted_queued = sorted(pending_tasks.keys(), key=lambda x: pending_tasks[x])
                try:
                    queue_index = sorted_queued.index(req.id_task)
                    current_textinfo = "In queue: {}/{}".format(queue_index + 1, len(sorted_queued))
                except ValueError:
                    # Should not happen if req.id_task in pending_tasks, but good for robustness
                    current_textinfo = "In queue (error finding position)"
            else:
                # Task is not active, not in pending_tasks. Could be completed or unknown.
                if completed:
                    current_textinfo = "Completed"
                else:
                    current_textinfo = "Status unknown" # Or some other appropriate default

        # When not active, we only send back active, queued, completed, id_live_preview and textinfo.
        # The Pydantic model will use default=None for progress, eta, live_preview.
        #print(f"DEBUG: Responding for task {req.id_task}: active={active}, queued={queued}, completed={completed}, textinfo='{current_textinfo}'")
        return ProgressResponse(
            active=active,
            queued=queued,
            completed=completed,
            progress=None, # Explicitly None
            eta=None, # Explicitly None
            live_preview=None, # Explicitly None
            id_live_preview= -1 if not shared.state.id_live_preview else shared.state.id_live_preview, # or req.id_live_preview
            textinfo=current_textinfo
        )

    # This part is ONLY reached if active is True
    current_progress = 0.0 # Default to float if active

    # Ensure shared.state is initialized before accessing attributes
    if shared.state.job_count is not None and shared.state.job_no is not None:
        if shared.state.job_count > 0:
            current_progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps is not None and shared.state.sampling_step is not None and shared.state.job_count is not None:
        if shared.state.sampling_steps > 0 and shared.state.job_count > 0:
            current_progress += (1 / shared.state.job_count) * (shared.state.sampling_step / shared.state.sampling_steps)

    current_progress = min(current_progress, 1.0)

    if shared.state.time_start is not None:
        elapsed_since_start = time.time() - shared.state.time_start
        if current_progress > 0:
            predicted_duration = elapsed_since_start / current_progress
            current_eta = predicted_duration - elapsed_since_start
        else:
            current_eta = None # Explicitly None if progress is 0
    else:
        current_eta = None # Explicitly None if time_start is not set

    # current_live_preview remains None by default
    # current_id_live_preview is already req.id_live_preview
    if opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image() # Make sure this method exists and is safe
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            if image is not None:
                buffered = io.BytesIO()
                save_kwargs = {}
                if opts.live_previews_image_format == "png":
                    if max(*image.size) <= 256:
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}
                
                image.save(buffered, format=opts.live_previews_image_format, **save_kwargs)
                base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                current_live_preview = f"data:image/{opts.live_previews_image_format};base64,{base64_image}"
                current_id_live_preview = shared.state.id_live_preview

    current_textinfo = shared.state.textinfo # This could be None
    #print(f"DEBUG: Responding for task {req.id_task}: active={active}, queued={queued}, completed={completed}, textinfo='{current_textinfo}'")
    return ProgressResponse(
        active=active,
        queued=queued,
        completed=completed,
        progress=current_progress,
        eta=current_eta,
        live_preview=current_live_preview,
        id_live_preview=current_id_live_preview,
        textinfo=current_textinfo
    )


def restore_progress(id_task):
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)
    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res
    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"