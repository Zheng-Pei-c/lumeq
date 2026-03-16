import os, sys, time, psutil
import functools
import logging

# optional GPU support (CuPy)
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    GPU_AVAILABLE = False

# --- global performance level ---
# 0 disables monitoring; higher values enable more detailed logging.
_PERF_LEVEL = 0

# --- global logger setup ---
_logger_performance = logging.getLogger("performance_monitor")
_logger_performance.setLevel(logging.INFO)

_default_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# default handler: stdout
_default_handler = logging.StreamHandler(sys.stdout)
_default_handler.setFormatter(_default_formatter)
if not _logger_performance.handlers:
    _logger_performance.addHandler(_default_handler)

### do not change _PERF_LEVEL and _logger_performance outside of this module ###


def set_performance_log(level=1, filename=None, debug=None):
    """
    Configure performance monitoring and optionally change the output destination.
    - level = 0 disables monitoring
    - level >= 1 enables monitoring with increasing detail
    - debug is kept for backward compatibility and maps True/False to level 1/0
    - filename = None keeps stdout logging
    """
    global _PERF_LEVEL
    if debug is not None:
        level = 1 if debug else 0
    _PERF_LEVEL = level

    if filename is None:
        return # keep default stdout handler

    global _logger_performance
    _logger_performance.handlers.clear() # remove existing handlers
    handler = logging.FileHandler(filename)
    handler.setFormatter(_default_formatter)
    _logger_performance.addHandler(handler)


def monitor_performance(_func=None, *, level=1):
    """
    decorator for monitoring memory of CPU (GPU) and timing of wall, CPU (GPU).
    Use either as @monitor_performance or @monitor_performance(level=1).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _PERF_LEVEL < level:
                return func(*args, **kwargs)  # bypass monitoring when disabled

            process = psutil.Process(os.getpid())

            # --- Start metrics ---
            mem_before = process.memory_info().rss / 1024 / 1024 # in MB
            wall_start = time.time()
            cpu_start = time.process_time()

            if GPU_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
                gpu_mem_before = cp.get_default_memory_pool().used_bytes() / 1024 / 1024
                gpu_event_start = cp.cuda.Event()
                gpu_event_end = cp.cuda.Event()
                gpu_event_start.record()
            else:
                gpu_mem_before = 0

            # --- execute target function ---
            result = func(*args, **kwargs)

            # --- end metrics ---
            mem_after = process.memory_info().rss / 1024 / 1024 # in MB
            wall_end = time.time()
            cpu_end = time.process_time()

            if GPU_AVAILABLE:
                gpu_event_end.record()
                gpu_event_end.synchronize()
                gpu_mem_after = cp.get_default_memory_pool().used_bytes() / 1024 / 1024
                gpu_time = cp.cuda.get_elapsed_time(gpu_event_start, gpu_event_end) / 1000.0
            else:
                gpu_mem_after = gpu_time = 0

            # --- logging results ---
            msg = [f"Performance report for function: {func.__name__}() [level={level}]",
                   f"    Wall time: %.4f s  CPU time: %.4f s  GPU time: %.4f s" % (wall_end-wall_start, cpu_end-cpu_start, gpu_time),
                   f"    RAM memory usage: %.2f MB  CUDA GPU memory usage: %.2f MB\n" % (mem_after, gpu_mem_after)]

            _logger_performance.info('\n' + '\n'.join(msg))
            return result

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)


# ---------------------------------------------------------
# exported public interface only
# ---------------------------------------------------------
__all__ = ['monitor_performance', 'set_performance_log']
