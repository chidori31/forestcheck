import importlib.util
from pathlib import Path

import pytest
from forestcheck.api.contracts import ExportRequest, ReviewRequest
from forestcheck.api.execution import Worker
from forestcheck.api.service import Service
from forestcheck.api.settings import Settings
from forestcheck.storage.application import Store
from test_backend_workflow import analysed


def module():
    spec = importlib.util.spec_from_file_location("recovery", Path("scripts/recovery.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_recovery_retains_analysis_decision_export_and_raw(tmp_path):
    mod = module()
    settings, store, svc, dataset, version, _, run, detail, suggestion = analysed(tmp_path)
    svc.review(
        ReviewRequest(
            schema_version="1.0",
            finding_id=detail["finding"]["id"],
            expected_version=0,
            error_assessment="confirmed_error",
            suggestion_id=suggestion["suggestion"]["id"],
            suggestion_decision="accepted",
        ),
        "test-operator",
        "review",
    )
    job = svc.export(
        ExportRequest(
            schema_version="1.0",
            version_id=version,
            analysis_id=run["id"],
            kind="report",
            format="json",
        ),
        "test-operator",
        "export",
    )
    Worker(store).run_once()
    backup = tmp_path / "backup"
    restored = tmp_path / "restored"
    mod.backup(settings, backup, stopped=True)
    mod.restore(backup, restored, workspace=settings.workspace)
    rs = Settings(root=restored, workspace=settings.workspace, token=settings.token)
    target = Store(rs)
    target.initialize()
    service = Service(target)
    assert service.job(run["id"])["status"] == "succeeded"
    assert service.reviews(detail["finding"]["id"])["total"] == 1
    assert service.job(job["id"])["status"] == "succeeded"
    for row in (store, target):
        with row.transaction() as cx:
            artifacts = cx.execute("SELECT id FROM artifacts").fetchall()
        for artifact in artifacts:
            row.resolve(artifact[0])
    assert service.records(version) == svc.records(version)
    from fastapi.testclient import TestClient
    from forestcheck.api.app import create_app

    with TestClient(create_app(rs), base_url="http://127.0.0.1") as client:
        headers = {"Authorization": "Bearer " + rs.token}
        assert client.get("/api/v1/analyses/" + run["id"], headers=headers).status_code == 200
        response = client.get("/api/v1/exports/" + job["id"] + "/download", headers=headers)
        assert response.status_code == 200
        assert len(response.json()["reviews"]) == 1
    with pytest.raises(ValueError):
        mod.restore(backup, restored, workspace=settings.workspace)
    (backup / "store/application.sqlite3").write_bytes(b"changed")
    with pytest.raises(ValueError):
        mod.restore(backup, tmp_path / "bad", workspace=settings.workspace)


def test_backup_requires_stopped_and_idle(tmp_path):
    mod = module()
    settings, store, *_ = analysed(tmp_path)
    with pytest.raises(ValueError):
        mod.backup(settings, tmp_path / "b1", stopped=False)
    store.enqueue("analysis", {}, "operator", "pending")
    with pytest.raises(ValueError):
        mod.backup(settings, tmp_path / "b2", stopped=True)


def test_backup_captures_committed_wal_and_external_dependencies(tmp_path):
    import json
    import sqlite3

    from forestcheck.storage.application import digest

    settings, store, svc, _, _, profile, *_ = analysed(tmp_path)
    dependency = tmp_path / "synthetic-reference.bin"
    dependency.write_bytes(b"independently invented reference")
    altered = profile.model_copy(update={"reference_parquet_path": str(dependency)})
    with store.transaction() as cx:
        cx.execute(
            "UPDATE profiles SET content_json=?,sha256=? WHERE id=?",
            (altered.model_dump_json(), digest(altered.model_dump(mode="json")), str(profile.id)),
        )
    with sqlite3.connect(store.database) as keeper:
        keeper.execute("PRAGMA journal_mode=WAL")
        keeper.execute("UPDATE datasets SET etag=etag+1")
        keeper.commit()
        assert Path(str(store.database) + "-wal").is_file()
        m = module().backup(settings, tmp_path / "snapshot", stopped=True)
        assert m["dependencies"][0]["sha256"]
        module().restore(tmp_path / "snapshot", tmp_path / "restored", workspace=settings.workspace)
        with sqlite3.connect(tmp_path / "restored/application.sqlite3") as restored:
            assert (
                restored.execute("SELECT etag FROM datasets").fetchall()
                == keeper.execute("SELECT etag FROM datasets").fetchall()
            )
    assert dependency.read_bytes() == b"independently invented reference"
    assert settings.token not in json.dumps(m)


def test_real_preflight_never_substitutes_demo(tmp_path):
    spec = importlib.util.spec_from_file_location("release", Path("scripts/release.py"))
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    with pytest.raises(ValueError, match="REAL_CONFIG_REQUIRED"):
        release.preflight_real(tmp_path / "missing.json")


def test_release_creates_scratch_parent_in_clean_checkout(tmp_path):
    spec = importlib.util.spec_from_file_location("release_clean", Path("scripts/release.py"))
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    release.prepare_workspace(tmp_path)
    assert (tmp_path / "work").is_dir()
