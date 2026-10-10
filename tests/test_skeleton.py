import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from forestcheck.api.app import app
from forestcheck.storage.metadata import migrate_metadata


def test_health_has_no_storage_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "forestcheck", "schema_version": "1.0"}
        assert client.post("/analyze", json={}).status_code == 404
    assert list(tmp_path.iterdir()) == []


def test_migration_idempotent_and_future_version_refused(tmp_path):
    repo = tmp_path / "repo"
    private = tmp_path / "private"
    repo.mkdir()
    private.mkdir()
    database = private / "metadata.sqlite3"
    migrate_metadata(database, workspace_root=repo)
    migrate_metadata(database, workspace_root=repo)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"datasets", "analysis_runs", "schema_migrations"} <= tables
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 1
        connection.execute("PRAGMA user_version=99")
    before = database.read_bytes()
    with pytest.raises(ValueError, match="unsupported_database_version"):
        migrate_metadata(database, workspace_root=repo)
    assert database.read_bytes() == before


def test_storage_requires_explicit_absolute_path_outside_workspace(tmp_path):
    with pytest.raises(ValueError):
        migrate_metadata(tmp_path / "metadata.sqlite3", workspace_root=tmp_path)
    with pytest.raises(ValueError):
        migrate_metadata(Path("metadata.sqlite3"), workspace_root=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ddl", ["PRAGMA user_version=1", "CREATE VIEW unrelated AS SELECT 1"])
def test_migration_rejects_unrelated_database_without_modifying_it(tmp_path, ddl):
    repo = tmp_path / "repo"
    repo.mkdir()
    database = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(ddl)
    before = database.read_bytes()
    with pytest.raises(ValueError):
        migrate_metadata(database, workspace_root=repo)
    assert database.read_bytes() == before


def test_external_hardlink_cannot_modify_database_inside_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    original = repo / "original.sqlite3"
    original.write_bytes(b"")
    link = tmp_path / "alias.sqlite3"
    os.link(original, link)
    with pytest.raises(ValueError):
        migrate_metadata(link, workspace_root=repo)
    assert original.read_bytes() == b""
