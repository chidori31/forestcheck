"""Owned loopback server and separate workers; detailed results remain private."""

import json
import os
import secrets
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

from forestcheck.api.admin import register_forestcheck
from forestcheck.api.settings import Settings, ordinary_path
from forestcheck.ingest.csv_import import sha256
from forestcheck.ml.data import LocalData
from forestcheck.ml.resources import run_bounded
from forestcheck.storage.application import Store


def run_smoke(config):
    root = Path(__file__).resolve().parents[1]
    local = json.loads(config.read_text("utf-8-sig"))
    source_root = Path(local["source_dir"])
    # Only immediate entries of the already-authorized task 6 folder; no other tasks.
    inventory = [
        {"name": p.name, "size": p.stat().st_size} for p in source_root.iterdir() if p.is_file()
    ]
    data = LocalData(config, workspace_root=root)
    stage = ordinary_path(data.private / "stage8", directory=True, missing=True)
    stage.mkdir(exist_ok=True)
    run = stage / str(uuid4())
    run.mkdir()
    ordinary_path(stage / "latest-run.json", missing=True).write_text(
        json.dumps({"run": str(run)}), "utf-8"
    )
    (run / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False), "utf-8")
    server = None
    try:
        original_hash = sha256(data.source)
        token = secrets.token_urlsafe(48)
        settings = Settings(
            root=run / "service", workspace=root, token=token, operator="local-smoke-operator"
        )
        store = Store(settings)
        store.initialize()
        print("SMOKE_REGISTERING", flush=True)
        profile_id = register_forestcheck(store, config, workspace_root=root)
        os.environ.update(
            FORESTCHECK_BACKEND_ROOT=str(settings.root),
            FORESTCHECK_OPERATOR_TOKEN=token,
            FORESTCHECK_OPERATOR_ID=settings.operator,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server_log = (run / "server.log").open("xb")
        server = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(root / "scripts/backend_local.py"),
                "serve",
                "--port",
                str(port),
            ],
            cwd=root,
            stdout=server_log,
            stderr=server_log,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        base = f"http://127.0.0.1:{port}"

        def call(path, body=None, *, raw=False, key=None, authenticated=True):
            headers = {"Authorization": "Bearer " + token} if authenticated else {}
            if key:
                headers["Idempotency-Key"] = key
            if raw:
                headers.update({"Content-Type": "text/csv", "X-Filename": "observations.csv"})
            elif body is not None:
                headers["Content-Type"] = "application/json"
                body = json.dumps(body).encode()
            request = urllib.request.Request(base + path, data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
                return (
                    json.loads(content)
                    if response.headers.get_content_type() == "application/json"
                    else content
                )

        for _ in range(100):
            try:
                if call("/health", authenticated=False)["status"] == "ok":
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            raise RuntimeError("server_start_failed")
        try:
            call("/api/v1/datasets", authenticated=False)
            raise AssertionError("unauthorized_read")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        resources = {}

        def work(name, job):
            logs = run / name
            logs.mkdir()
            result = run_bounded(
                [sys.executable, "-B", str(root / "scripts/backend_local.py"), "worker", "--once"],
                cwd=root,
                log_dir=logs,
                wall_seconds=900,
                memory_mib=4096,
                cpu_threads=2,
            )
            resources[name] = result
            status = call("/api/v1/jobs/" + job["id"])
            (logs / "job.json").write_text(json.dumps(status, ensure_ascii=False), "utf-8")
            assert result["status"] == "success" and status["status"] == "succeeded", status[
                "error_code"
            ]
            print("SMOKE_" + name.upper() + "_COMPLETE", flush=True)
            return status

        content = data.source.read_bytes()
        dataset = call("/api/v1/datasets", content, raw=True, key="source-v1")
        assert call("/api/v1/datasets", content, raw=True, key="source-v1") == dataset
        del content
        schema = call(
            "/api/v1/datasets/" + dataset["id"] + "/schema",
            {
                "schema_version": "1.0",
                "encoding": data.mapping.encoding,
                "delimiter": data.mapping.delimiter,
            },
        )
        assert schema["headers"] == [c.source_header for c in data.mapping.columns]
        mapping = call(
            "/api/v1/datasets/" + dataset["id"] + "/mapping",
            {
                "schema_version": "1.0",
                "expected_version": dataset["etag"],
                "mapping": data.mapping.model_dump(mode="json"),
            },
            key="mapping-v1",
        )
        imported = work("import", mapping)
        version = imported["result"]["version_id"]
        payload = {
            "schema_version": "1.0",
            "version_id": version,
            "profile_id": profile_id,
            "use_models": True,
        }
        analysis = call("/api/v1/analyses", payload, key="analysis-v1")
        assert call("/api/v1/analyses", payload, key="analysis-v1") == analysis
        analysed = work("analysis", analysis)
        findings = call("/api/v1/findings?analysis_id=" + analysis["id"] + "&limit=5")
        assert findings["total"] == analysed["result"]["findings"]
        if findings["items"]:
            detail = call("/api/v1/findings/" + findings["items"][0]["finding"]["id"])
            (run / "finding.json").write_text(json.dumps(detail, ensure_ascii=False), "utf-8")
        exports = []
        for fmt, kind in (("json", "report"), ("csv", "original"), ("xlsx", "original")):
            export = call(
                "/api/v1/exports",
                {
                    "schema_version": "1.0",
                    "version_id": version,
                    "kind": kind,
                    "format": fmt,
                    "analysis_id": analysis["id"] if kind == "report" else None,
                },
                key=kind + fmt,
            )
            exported = work("export_" + fmt, export)
            downloaded = call("/api/v1/exports/" + export["id"] + "/download")
            if fmt == "json":
                assert len(downloaded["findings"]) == findings["total"]
                assert downloaded["reviews"] == []
            else:
                assert len(downloaded) > 0
            exports.append(exported["result"])
        data.verify_inputs()
        assert sha256(data.source) == original_hash
        summary = {
            "schema_version": "1.0",
            "status": "passed",
            "http_loopback": True,
            "separate_worker_processes": True,
            "unauthorized_read_blocked": True,
            "idempotency_checked": True,
            "import_report": imported["result"]["import_report"],
            "analysis": analysed["result"],
            "resources": resources,
            "exports": exports,
            "source_unchanged": True,
            "expert_reviews_created": 0,
            "real_corrections_applied": 0,
            "holdout_evaluation": "NOT_RUN",
        }
        (run / "verification.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), "utf-8"
        )
        print("SMOKE_PASSED", flush=True)
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), "utf-8")
        print("SMOKE_FAILED", flush=True)
        sys.exit(3)
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=10)
            server_log.close()


def main():
    from backend_local import StatusParser

    parser = StatusParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    try:
        args = parser.parse_args()
        run_smoke(args.config)
        return 0
    except Exception:
        print("SMOKE_FAILED", flush=True)
        return 3
    except KeyboardInterrupt:
        print("SMOKE_INTERRUPTED", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
