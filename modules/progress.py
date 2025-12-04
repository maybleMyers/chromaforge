# Progress tracking module - now backed by thread-safe QueueManager
from __future__ import annotations
import base64
import io
import time

import gradio as gr
from pydantic import BaseModel, Field

from modules.shared import opts
import modules.shared as shared
from collections import OrderedDict
import string
import random
from typing import List, Optional
import threading

# Thread-safe lock for all queue operations
_queue_lock = threading.RLock()

current_task = None
pending_tasks = OrderedDict()
finished_tasks = []
recorded_results = []
recorded_results_limit = 2

# Sequence counter for deterministic ordering
_task_sequence = 0


def _next_sequence():
    """Get next sequence number"""
    global _task_sequence
    _task_sequence += 1
    return _task_sequence


def start_task(id_task):
    global current_task
    with _queue_lock:
        current_task = id_task
        pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task
    with _queue_lock:
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
    with _queue_lock:
        recorded_results.append((id_task, res))
        if len(recorded_results) > recorded_results_limit:
            recorded_results.pop(0)


def add_task_to_queue(id_job):
    with _queue_lock:
        # Use sequence number for ordering instead of timestamp
        # This ensures deterministic ordering even for rapid submissions
        pending_tasks[id_job] = _next_sequence()


def get_current_task():
    """Thread-safe getter for current_task"""
    with _queue_lock:
        return current_task


def get_pending_count():
    """Thread-safe getter for pending tasks count"""
    with _queue_lock:
        return len(pending_tasks)


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
    queue_position: int | None = Field(default=None, title="Queue position")
    queue_total: int | None = Field(default=None, title="Total queue size")


def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


def get_pending_tasks():
    with _queue_lock:
        pending_tasks_ids = list(pending_tasks)
        pending_len = len(pending_tasks_ids)
    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


def progressapi(req: ProgressRequest):
    # Take a consistent snapshot under lock
    with _queue_lock:
        snapshot_current = current_task
        snapshot_pending = dict(pending_tasks)
        snapshot_finished = list(finished_tasks)

    # Calculate status from snapshot (no lock needed)
    active = req.id_task == snapshot_current
    queued = req.id_task in snapshot_pending
    completed = req.id_task in snapshot_finished

    # Initialize response fields
    current_progress = None
    current_eta = None
    current_live_preview = None
    current_id_live_preview = req.id_live_preview
    current_textinfo = None
    queue_position = None
    queue_total = None

    if not active:
        if queued:
            # Calculate queue position from snapshot
            sorted_queued = sorted(snapshot_pending.keys(), key=lambda x: snapshot_pending[x])
            try:
                queue_position = sorted_queued.index(req.id_task) + 1
                queue_total = len(sorted_queued)
                current_textinfo = f"In queue: {queue_position}/{queue_total}"
            except ValueError:
                current_textinfo = "In queue"
        elif completed:
            current_textinfo = "Completed"
        elif snapshot_current is not None:
            # Task not found but there's an active task - show as queued behind it
            # This handles reconnection from another window
            queue_total = len(snapshot_pending) + 1
            queue_position = queue_total
            current_textinfo = f"In queue: {queue_position}/{queue_total}"
            queued = True
        else:
            current_textinfo = None

        return ProgressResponse(
            active=active,
            queued=queued,
            completed=completed,
            progress=None,
            eta=None,
            live_preview=None,
            id_live_preview=-1 if not shared.state.id_live_preview else shared.state.id_live_preview,
            textinfo=current_textinfo,
            queue_position=queue_position,
            queue_total=queue_total
        )

    # Task is active - calculate progress
    current_progress = 0.0

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
            current_eta = None
    else:
        current_eta = None

    # Live preview
    if opts is not None and opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image()
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            if image is not None:
                buffered = io.BytesIO()
                save_kwargs = {}
                image_format = getattr(opts, 'live_previews_image_format', 'png') if opts else 'png'
                if image_format == "png":
                    if max(*image.size) <= 256:
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}

                image.save(buffered, format=image_format, **save_kwargs)
                base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                current_live_preview = f"data:image/{image_format};base64,{base64_image}"
                current_id_live_preview = shared.state.id_live_preview

    current_textinfo = shared.state.textinfo

    return ProgressResponse(
        active=active,
        queued=queued,
        completed=completed,
        progress=current_progress,
        eta=current_eta,
        live_preview=current_live_preview,
        id_live_preview=current_id_live_preview,
        textinfo=current_textinfo,
        queue_position=None,
        queue_total=None
    )


def restore_progress(id_task):
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)
    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res
    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
