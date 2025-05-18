import time
import traceback
import threading

from modules import shared, progress, fifo_lock

g_queue_lock = fifo_lock.FIFOLock()

lock = threading.Lock()
last_id = 0
waiting_list = []
finished_list = []
last_exception = None
current_task_id_processing = None


class Task:
    def __init__(self, task_id, func, args, kwargs, gradio_progress_id=None):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.gradio_progress_id = gradio_progress_id
        self.result = None
        self.exception = None

    def work(self):
        global last_exception, g_queue_lock, current_task_id_processing
        
        current_task_id_processing = self.task_id

        try:
            with g_queue_lock:
                shared.state.begin(job=self.gradio_progress_id)
                progress.start_task(self.gradio_progress_id)
                
                try:
                    self.result = self.func(*self.args, **self.kwargs)
                    progress.record_results(self.gradio_progress_id, self.result)
                except Exception as task_e:
                    self.exception = task_e
                    last_exception = task_e
                    raise 
                finally:
                    progress.finish_task(self.gradio_progress_id)
                    shared.state.end()
            
            self.exception = None
            last_exception = None

        except Exception as e:
            traceback.print_exc()
            print(f"Error in Task {self.task_id}: {e}")
            self.exception = e
            last_exception = e
        finally:
            current_task_id_processing = None


def loop():
    global lock, last_id, waiting_list, finished_list, current_task_id_processing
    while True:
        time.sleep(0.01)
        task_to_process = None
        if len(waiting_list) > 0:
            with lock:
                if len(waiting_list) > 0:
                    task_to_process = waiting_list.pop(0)

        if task_to_process:
            task_to_process.work()
            with lock:
                finished_list.append(task_to_process)


def async_run(func, *args, gradio_progress_id=None, **kwargs):
    global lock, last_id, waiting_list
    with lock:
        last_id += 1
        new_task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs, gradio_progress_id=gradio_progress_id)
        waiting_list.append(new_task)
    return new_task.task_id


def run_and_wait_result(func, *args, gradio_progress_id=None, **kwargs):
    global lock, finished_list

    current_id = async_run(func, *args, gradio_progress_id=gradio_progress_id, **kwargs)

    while True:
        time.sleep(0.01)
        finished_task_obj = None
        
        temp_finished_list_snapshot = finished_list[:]

        for t in temp_finished_list_snapshot:
            if t.task_id == current_id:
                finished_task_obj = t
                break
        
        if finished_task_obj is not None:
            with lock:
                if finished_task_obj in finished_list :
                    finished_list.remove(finished_task_obj)
            
            if finished_task_obj.exception is not None:
                raise finished_task_obj.exception 
            
            return finished_task_obj.result

def is_processing_task():
    return current_task_id_processing is not None

def get_queue_size():
    with lock:
        return len(waiting_list)