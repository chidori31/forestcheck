"""Read-only browser check of an existing stage8 development run; private screenshots only."""

import argparse
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4


def main():
    root = Path(__file__).resolve().parents[1]
    for port in (8124, 5174):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text("utf-8-sig"))
    private = Path(config["private_dir"])
    previous = Path(json.loads((private / "stage8/latest-run.json").read_text("utf-8"))["run"])
    service = previous / "service"
    run = private / "stage9" / str(uuid4())
    run.mkdir(parents=True)
    (private / "stage9/latest-run.json").write_text(json.dumps({"run": str(run)}), "utf-8")
    inventory = [
        {"name": p.name, "size": p.stat().st_size}
        for p in Path(config["source_dir"]).iterdir()
        if p.is_file()
    ]
    (run / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False), "utf-8")

    def state():
        with sqlite3.connect(
            f"file:{(service / 'application.sqlite3').as_posix()}?mode=ro", uri=True
        ) as cx:
            names = [x[0] for x in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            return {
                name: cx.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0] for name in names
            }

    before = state()
    env = os.environ.copy()
    env.update(
        FORESTCHECK_BACKEND_ROOT=str(service),
        FORESTCHECK_OPERATOR_TOKEN=secrets.token_urlsafe(48),
        FORESTCHECK_OPERATOR_ID="local-ui-inspection",
        FORESTCHECK_ALLOWED_ORIGINS="http://127.0.0.1:5174",
        FORESTCHECK_DATA_MODE="real",
        FORESTCHECK_UI_REPORT=str(run),
    )
    processes = []
    try:
        with (run / "process.log").open("wb") as log:
            for command, cwd in [
                (
                    [sys.executable, "-B", "scripts/backend_local.py", "serve", "--port", "8124"],
                    root,
                ),
                (
                    [
                        "node",
                        "node_modules/vite/bin/vite.js",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "5174",
                        "--strictPort",
                    ],
                    root / "frontend",
                ),
            ]:
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=log,
                        stderr=log,
                        stdin=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                )
            for url in ("http://127.0.0.1:8124/health", "http://127.0.0.1:5174"):
                for attempt in range(100):
                    try:
                        if any(p.poll() is not None for p in processes):
                            raise RuntimeError("owned_server_stopped")
                        with urllib.request.urlopen(url, timeout=1) as r:
                            assert r.status == 200
                        break
                    except Exception:
                        time.sleep(0.2)
                else:
                    raise RuntimeError("server_not_ready")
            outcome = subprocess.run(
                ["node", "scripts/ui_smoke.mjs"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
                timeout=120,
            )
            assert outcome.returncode == 0
        after = state()
        assert before == after
        (run / "integrity.json").write_text(
            json.dumps({"before": before, "after": after, "read_only": True}), "utf-8"
        )
        print("REAL_UI_PASSED_READ_ONLY")
    except Exception:
        import traceback

        (run / "error.txt").write_text(traceback.format_exc(), "utf-8")
        print("REAL_UI_FAILED")
        sys.exit(1)
    finally:
        for p in reversed(processes):
            p.terminate()
            p.wait(timeout=10)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("REAL_UI_FAILED")
        raise SystemExit(1) from None
