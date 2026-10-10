"""Local, sampled resource supervision for explicit CPU experiments.

The caller confines cwd and log_dir to its permitted workspace/private store.
Memory is sampled process-tree RSS (shared pages can be counted more than once),
not an instantaneous OS memory quota. Overshoot between samples is possible.
Thread environment caps do not impose a kernel CPU quota; model thread counts
must also be passed explicitly to native estimators by the experiment worker.
"""

import math
import os
import stat
import subprocess
import time
from pathlib import Path

SAMPLING_INTERVAL_SECONDS = 0.1
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _checked_directory(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute_controlled_directory_required")
    for part in (path, *path.parents):
        info = part.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("ordinary_directory_required")
    return path


def _positive_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"invalid_{name}")


def run_bounded(
    command: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    wall_seconds: float = 900,
    memory_mib: float = 4096,
    cpu_threads: int = 2,
) -> dict:
    """Run one owned process; persist output locally and return no paths or values.

    On cancellation, terminate only the launched process and descendants observed
    under it, then propagate KeyboardInterrupt. A missing dependency fails before
    launching. No network, shell, cloud provisioning, or model promotion occurs.
    """
    _positive_number(wall_seconds, "wall_budget")
    _positive_number(memory_mib, "memory_budget")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("invalid_cpu_threads")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) for argument in command)
        or not command[0]
    ):
        raise ValueError("explicit_command_arguments_required")
    cwd = _checked_directory(cwd)
    log_dir = _checked_directory(log_dir)
    if any(log_dir.iterdir()):
        raise ValueError("empty_controlled_log_directory_required")
    import psutil

    result = dict(
        status="failed",
        exit_code=None,
        wall_seconds=0.0,
        peak_process_tree_rss_bytes=0,
        sampling_interval_seconds=SAMPLING_INTERVAL_SECONDS,
        memory_method="sampled_psutil_sum_process_tree_rss",
        memory_limit_method="sampled_monitor_with_possible_overshoot",
        cpu_threads=cpu_threads,
        cpu_limit_method="thread_environment_caps",
        budget_reason=None,
        launch_error=None,
        process_id=None,
        measurement_incomplete=False,
    )
    environment = os.environ.copy()
    environment.update({key: str(cpu_threads) for key in _THREAD_ENV})
    owned = {}
    process = None
    root_process = None
    start = time.perf_counter()

    def remember(item):
        try:
            owned[(item.pid, item.create_time())] = item
        except psutil.NoSuchProcess:
            return
        except psutil.AccessDenied:
            result["measurement_incomplete"] = True

    def sample():
        if root_process is not None:
            try:
                for child in root_process.children(recursive=True):
                    remember(child)
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                result["measurement_incomplete"] = True
        rss = 0
        for item in list(owned.values()):
            try:
                if item.is_running():
                    rss += item.memory_info().rss
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                result["measurement_incomplete"] = True
        result["peak_process_tree_rss_bytes"] = max(result["peak_process_tree_rss_bytes"], rss)
        return rss

    def terminate_owned():
        # Refresh descendants before terminating the parent. psutil guards reused
        # PIDs with creation times for these process-object operations.
        sample()
        children = [item for item in owned.values() if item.pid != process.pid]
        for item in children:
            try:
                item.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        # Reap the immediate child through Popen, preserving its actual exit code
        # on POSIX as well as Windows; psutil must not consume that wait status.
        if process.poll() is None:
            process.terminate()
        _, alive = psutil.wait_procs(children, timeout=2)
        for item in alive:
            try:
                item.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(alive, timeout=2)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    with (
        (log_dir / "stdout.log").open("xb") as stdout,
        (log_dir / "stderr.log").open("xb") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError:
            result["launch_error"] = "process_launch_failed"
            result["wall_seconds"] = time.perf_counter() - start
            return result
        result["process_id"] = process.pid
        try:
            try:
                root_process = psutil.Process(process.pid)
                remember(root_process)
            except psutil.NoSuchProcess:
                pass
            while True:
                rss = sample()
                elapsed = time.perf_counter() - start
                if result["measurement_incomplete"]:
                    result.update(status="failed", budget_reason="resource_monitor_incomplete")
                    terminate_owned()
                    break
                if rss > memory_mib * 1024 * 1024:
                    result.update(status="budget_exceeded", budget_reason="process_tree_memory")
                    terminate_owned()
                    break
                if elapsed > wall_seconds:
                    result.update(status="budget_exceeded", budget_reason="wall_time")
                    terminate_owned()
                    break
                exit_code = process.poll()
                if exit_code is not None:
                    result["status"] = "success" if exit_code == 0 else "failed"
                    # Do not leave observed descendant workers running after the
                    # command returns. These handles originate from this process.
                    if any(
                        item.pid != process.pid and item.is_running() for item in owned.values()
                    ):
                        terminate_owned()
                    break
                time.sleep(min(SAMPLING_INTERVAL_SECONDS, max(0, wall_seconds - elapsed)))
        except BaseException:
            terminate_owned()
            raise
        finally:
            result["exit_code"] = process.poll()
            result["wall_seconds"] = time.perf_counter() - start
    return result
