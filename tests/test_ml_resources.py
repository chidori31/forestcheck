"""Resource supervision uses only tiny, freshly created synthetic subprocesses."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from forestcheck.ml.resources import run_bounded


def run(tmp_path, script, **limits):
    logs = tmp_path / "private-synthetic-logs"
    logs.mkdir()
    return run_bounded([sys.executable, "-c", script], cwd=tmp_path, log_dir=logs, **limits), logs


def test_success_captures_logs_and_measures_process_rss_without_console(tmp_path, capsys):
    result, logs = run(tmp_path, "import time; print('synthetic'); time.sleep(0.2)")
    assert result["status"] == "success"
    assert result["exit_code"] == 0
    assert result["wall_seconds"] > 0
    assert result["peak_process_tree_rss_bytes"] > 0
    assert result["memory_method"] == "sampled_psutil_sum_process_tree_rss"
    assert result["sampling_interval_seconds"] == 0.1
    assert (logs / "stdout.log").read_text().strip() == "synthetic"
    assert capsys.readouterr().out == ""


def test_failure_is_not_success_and_stderr_remains_local(tmp_path):
    result, logs = run(
        tmp_path, "import sys; print('synthetic failure', file=sys.stderr); sys.exit(7)"
    )
    assert result["status"] == "failed"
    assert result["exit_code"] == 7
    assert "synthetic failure" in (logs / "stderr.log").read_text()


def test_wall_budget_terminates_owned_child(tmp_path):
    result, _ = run(tmp_path, "import time; time.sleep(60)", wall_seconds=0.2)
    assert result["status"] == "budget_exceeded"
    assert result["budget_reason"] == "wall_time"
    assert result["exit_code"] is not None
    assert result["wall_seconds"] < 10


def test_memory_budget_includes_interpreter_rss(tmp_path):
    result, _ = run(
        tmp_path, "import time; block=bytearray(32*1024*1024); time.sleep(60)", memory_mib=1
    )
    assert result["status"] == "budget_exceeded"
    assert result["budget_reason"] == "process_tree_memory"
    assert result["peak_process_tree_rss_bytes"] > 1024 * 1024


def test_requested_thread_environment_caps_override_parent_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    script = (
        "import os; print(','.join(os.environ[n] for n in "
        "['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']))"
    )
    result, logs = run(tmp_path, script, cpu_threads=2)
    assert result["status"] == "success"
    assert (logs / "stdout.log").read_text().strip() == "2,2,2,2"


@pytest.mark.parametrize(
    "limits",
    [
        {"wall_seconds": 0},
        {"memory_mib": 0},
        {"cpu_threads": 0},
        {"cpu_threads": True},
        {"wall_seconds": float("nan")},
    ],
)
def test_invalid_resource_limits_rejected_before_execution(tmp_path, limits):
    with pytest.raises(ValueError):
        run(tmp_path, "raise Exception('must not start')", **limits)


def test_refuses_nonempty_output_directory(tmp_path):
    logs = tmp_path / "existing"
    logs.mkdir()
    (logs / "keep.txt").write_text("synthetic")
    with pytest.raises(ValueError):
        run_bounded([sys.executable, "-c", "pass"], cwd=tmp_path, log_dir=logs)
    assert (logs / "keep.txt").read_text() == "synthetic"


def test_process_tree_budget_terminates_observed_descendant(tmp_path):
    import psutil

    pid_file = tmp_path / "synthetic-child.pid"
    child_script = "import time; time.sleep(60)"
    script = (
        "import subprocess,sys,time,pathlib; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    result, _ = run(tmp_path, script, wall_seconds=0.6)
    assert result["status"] == "budget_exceeded"
    if pid_file.exists():
        assert not psutil.pid_exists(int(pid_file.read_text()))
    assert os.getpid() != result["process_id"]


def test_missing_executable_records_failed_without_leaking_error_details(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    result = run_bounded([str(tmp_path / "missing-executable")], cwd=tmp_path, log_dir=logs)
    assert result["status"] == "failed"
    assert result["launch_error"] == "process_launch_failed"
    assert result["exit_code"] is None
    assert str(tmp_path) not in str(result)


def test_relative_log_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_bounded([sys.executable, "-c", "pass"], cwd=tmp_path, log_dir=Path("relative"))


def test_keyboard_interrupt_reaps_only_launched_process(tmp_path, monkeypatch):
    from forestcheck.ml import resources

    captured = []
    original_popen = resources.subprocess.Popen

    def capture(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        captured.append(child)
        return child

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(resources.subprocess, "Popen", capture)
    monkeypatch.setattr(
        resources,
        "time",
        SimpleNamespace(
            perf_counter=resources.time.perf_counter,
            sleep=interrupt,
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "import time; time.sleep(60)")
    assert len(captured) == 1
    assert captured[0].poll() is not None
