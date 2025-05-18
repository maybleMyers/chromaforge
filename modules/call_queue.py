import os.path
from functools import wraps
import html
import time
import traceback
import random

from modules_forge import main_thread
from modules import shared, progress, errors, devices, fifo_lock, profiling

from modules import fifo_lock as cq_fifo_lock
internal_queue_lock = cq_fifo_lock.FIFOLock()

def wrap_queued_call(func):
    def f(*args, **kwargs):
        with internal_queue_lock:
            res = func(*args, **kwargs)
        return res
    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    @wraps(func)
    def f(*args, **kwargs):
        id_task_for_gradio_progress = None
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            id_task_for_gradio_progress = args[0]
            progress.add_task_to_queue(id_task_for_gradio_progress)
        
        res = func(*args, **kwargs) 

        return res

    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    @wraps(func)
    def f(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        finally:
            shared.state.skipped = False
            shared.state.interrupted = False
            shared.state.stopping_generation = False
            
            if not main_thread.is_processing_task():
                shared.state.job_count = 0
                shared.state.job = ""
            else:
                shared.state.job_count = 0
                shared.state.job = ""
        return res

    return wrap_gradio_call_no_job(f, extra_outputs, add_stats)


def wrap_gradio_call_no_job(func, extra_outputs=None, add_stats=False):
    @wraps(func)
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            e_to_display = e
            if main_thread.last_exception is not None and isinstance(e, RuntimeError) and "Task raised an exception" in str(e):
                e_to_display = main_thread.last_exception
            
            max_debug_str_len = 131072
            message = "Error completing request"
            arg_str = f"Arguments: {args} {kwargs}"[:max_debug_str_len]
            if len(arg_str) > max_debug_str_len:
                arg_str += f" (Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)"
            
            errors.report(f"{message}\n{arg_str}", exc_info=(type(e), e, e.__traceback__))
            print(f"Error: {type(e_to_display).__name__}: {e_to_display}")

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            error_message_html = f'{type(e_to_display).__name__}: {e_to_display}'
            res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message_html)}</div>"]
            
        devices.torch_gc()

        if not add_stats:
            return tuple(res)

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.1f} sec."
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m} min. "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = sys_peak/max(sys_total, 1) * 100

            toltip_a = "Active: peak amount of video memory used during generation (excluding cached data)"
            toltip_r = "Reserved: total amount of video memory allocated by the Torch library "
            toltip_sys = "System: peak amount of video memory allocated by all running programs, out of total capacity"

            text_a = f"<abbr title='{toltip_a}'>A</abbr>: <span class='measurement'>{active_peak/1024:.2f} GB</span>"
            text_r = f"<abbr title='{toltip_r}'>R</abbr>: <span class='measurement'>{reserved_peak/1024:.2f} GB</span>"
            text_sys = f"<abbr title='{toltip_sys}'>Sys</abbr>: <span class='measurement'>{sys_peak/1024:.1f}/{sys_total/1024:g} GB</span> ({sys_pct:.1f}%)"

            vram_html = f"<p class='vram'>{text_a}, <wbr>{text_r}, <wbr>{text_sys}</p>"
        else:
            vram_html = ''

        if shared.opts.profiling_enable and os.path.exists(shared.opts.profiling_filename):
            profiling_html = f"<p class='profile'> [ <a href='{profiling.webpath()}' download>Profile</a> ] </p>"
        else:
            profiling_html = ''
        
        if res and isinstance(res[-1], str):
             res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}{profiling_html}</div>"
        elif res:
             res.append(f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}{profiling_html}</div>")
        else:
             res = [f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}{profiling_html}</div>"]

        return tuple(res)

    return f