"""Release operations; no training, cloud access or demo fallback in real mode."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd=ROOT):
    subprocess.run(command, cwd=cwd, check=True)


def preflight_real(config):
    if config is None or not config.is_file():
        raise ValueError("REAL_CONFIG_REQUIRED")
    from forestcheck.ml.data import LocalData

    try:
        return LocalData(config, workspace_root=ROOT)
    except (OSError, ValueError, KeyError):
        raise ValueError("REAL_DATA_OR_VERIFIED_ARTIFACTS_UNAVAILABLE") from None


def prepare_workspace(root):
    (root / "work").mkdir(exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "demo", "real", "tests", "benchmark"])
    p.add_argument("--config", type=Path)
    p.add_argument("--private-output", type=Path)
    a = p.parse_args()
    if a.command in {"tests", "benchmark"}:
        prepare_workspace(ROOT)
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    py = sys.executable
    if a.command == "build":
        run([npm, "run", "build"], ROOT / "frontend")
        return
    if a.command == "tests":
        run([py, "-m", "pytest", "-q"])
        run([py, "-m", "ruff", "check", "backend", "scripts", "tests"])
        for command in ("build", "lint", "test", "test:e2e"):
            run([npm, "run", command], ROOT / "frontend")
        return
    if a.command == "benchmark":
        import json

        from forestcheck.api.settings import ordinary_path
        from forestcheck.ml.resources import run_bounded

        if a.private_output is None:
            raise ValueError("PRIVATE_OUTPUT_REQUIRED")
        dest = ordinary_path(a.private_output, directory=True, missing=True)
        if dest.exists() or dest.is_relative_to(ROOT) or ROOT.is_relative_to(dest):
            raise ValueError("NEW_EXTERNAL_OUTPUT_REQUIRED")
        dest.mkdir(parents=True)
        result = run_bounded(
            [py, "-m", "pytest", "tests/test_recovery.py", "tests/test_backend_workflow.py", "-q"],
            cwd=ROOT,
            log_dir=dest,
            wall_seconds=900,
            memory_mib=4096,
            cpu_threads=2,
        )
        (dest / "resources.json").write_text(json.dumps(result, indent=2), "utf-8")
        print("SYNTHETIC_BENCHMARK_" + result["status"].upper())
        if result["status"] != "success":
            raise SystemExit(3)
        return
    if not (ROOT / "frontend/dist/index.html").is_file():
        raise ValueError("BUILD_REQUIRED")
    from forestcheck.api.settings import Settings

    if a.command == "real":
        preflight_real(a.config)
    os.environ["FORESTCHECK_DATA_MODE"] = "synthetic" if a.command == "demo" else "real"
    os.environ["FORESTCHECK_ALLOWED_ORIGINS"] = "http://127.0.0.1:4173"
    settings = Settings.from_env()
    children = []
    try:
        if a.command == "real":
            run([py, "-B", "scripts/backend_local.py", "init"])
            run(
                [
                    py,
                    "-B",
                    "scripts/backend_local.py",
                    "register-local",
                    "--config",
                    str(a.config.resolve()),
                ]
            )
            commands = [
                [py, "-B", "scripts/backend_local.py", "worker"],
                [py, "-B", "scripts/backend_local.py", "serve"],
            ]
        else:
            if settings.root.exists() and any(settings.root.iterdir()):
                raise ValueError("DEMO_REQUIRES_EMPTY_STORAGE")
            commands = [[py, "-B", "scripts/ui_demo.py", "--port", "8000"]]
        commands.append([py, "-B", "scripts/static_local.py"])
        for command in commands:
            children.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            )
        print("LOCAL_UI_127.0.0.1_4173_BACKEND_8000", flush=True)
        while all(c.poll() is None for c in children):
            time.sleep(0.5)
        raise ValueError("CHILD_PROCESS_STOPPED")
    finally:
        # Stop demo's descendants as well as its parent; no unrelated processes.
        import psutil

        for c in children:
            try:
                proc = psutil.Process(c.pid)
                owned = proc.children(recursive=True) + [proc]
                for item in owned:
                    item.terminate()
                _, alive = psutil.wait_procs(owned, timeout=5)
                for item in alive:
                    item.kill()
            except psutil.NoSuchProcess:
                pass
        for c in children:
            c.wait(timeout=10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("LOCAL_STOPPED")
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        codes = {
            "REAL_CONFIG_REQUIRED",
            "REAL_DATA_OR_VERIFIED_ARTIFACTS_UNAVAILABLE",
            "BUILD_REQUIRED",
            "PRIVATE_OUTPUT_REQUIRED",
            "NEW_EXTERNAL_OUTPUT_REQUIRED",
            "DEMO_REQUIRES_EMPTY_STORAGE",
            "CHILD_PROCESS_STOPPED",
        }
        print(str(e) if str(e) in codes else "RELEASE_FAILED_CHECK_LOCAL_CONFIGURATION")
        raise SystemExit(3)
