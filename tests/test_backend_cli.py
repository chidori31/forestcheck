import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from forestcheck.domain.contracts import AnalysisRun
from test_backend_workflow import analysed

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/backend_local.py"


def test_cli_init_worker_openapi_and_safe_failure(tmp_path):
    # CLI uses the real workspace boundary, so its private fixture must be outside it.
    with tempfile.TemporaryDirectory(prefix="forestcheck-synthetic-backend-") as folder:
        _cli_check(tmp_path, Path(folder))


def _cli_check(tmp_path, private):
    env = dict(os.environ)
    env.update(
        FORESTCHECK_BACKEND_ROOT=str(private),
        FORESTCHECK_OPERATOR_TOKEN="synthetic-token-" + "x" * 32,
    )

    def run(*args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), *args], env=env, capture_output=True, text=True
        )

    for args in (("init",), ("worker", "--once")):
        result = run(*args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stderr == "" and result.stdout.strip() == "BACKEND_COMPLETE"
    output = tmp_path / "openapi.json"
    result = run("openapi", "--output", str(output))
    assert result.returncode == 0
    schema = json.loads(output.read_text("utf-8"))
    assert "/api/v1/datasets/{dataset_id}/versions" in schema["paths"]
    assert "get" in schema["paths"]["/api/v1/exports"]
    assert "RecommendationResult" in schema["components"]["schemas"]
    assert "security" in schema["paths"]["/api/v1/findings"]["get"]
    result = run("serve", "--host", "0.0.0.0")
    assert result.returncode == 3 and result.stdout.strip() == "BACKEND_FAILED"
    assert result.stderr == ""


def test_native_analysis_binding_and_version_metadata(tmp_path):
    _, _, service, dataset, version, _, run, _, _ = analysed(tmp_path)
    view = service.job(run["id"])
    native = AnalysisRun.model_validate(view["analysis_run"])
    assert native.status == "completed"
    assert str(native.dataset_id) == view["request"]["binding"]["dataset_id"]
    assert native.config_digest == view["request_sha256"]
    versions = service.versions(dataset["id"])
    assert versions["total"] == 1
    assert versions["items"][0]["id"] == version
    assert "artifact_id" not in versions["items"][0]


def test_smoke_invalid_config_does_not_disclose_paths(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT.with_name("backend_smoke.py")),
            "--config",
            str(tmp_path / "PRIVATE"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert result.stdout.strip() == "SMOKE_FAILED" and result.stderr == ""


def test_smoke_rejects_private_directory_in_repository_before_writes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    config = tmp_path / "local.json"
    config.write_text(json.dumps({"source_dir": str(source), "private_dir": str(private)}))
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT.with_name("backend_smoke.py")), "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert not (private / "stage8").exists()
