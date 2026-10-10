import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from forestcheck.api.settings import Settings
from forestcheck.storage.application import Store, WorkflowError


def configured(tmp_path):
    root, workspace = tmp_path / "private", tmp_path / "workspace"
    root.mkdir()
    workspace.mkdir()
    settings = Settings(
        root=root, workspace=workspace, token="x" * 40, operator="synthetic-operator"
    )
    store = Store(settings)
    store.initialize()
    return settings, store


def test_persistent_idempotency_and_changed_payload_conflict(tmp_path):
    settings, store = configured(tmp_path)
    first = store.enqueue("import", {"dataset_id": "synthetic"}, "operator", "request-a")
    assert (
        Store(settings).enqueue("import", {"dataset_id": "synthetic"}, "operator", "request-a")
        == first
    )
    assert store.list_jobs()["total"] == 1
    with pytest.raises(WorkflowError) as error:
        store.enqueue("import", {"dataset_id": "different"}, "operator", "request-a")
    assert error.value.status == 409


def test_atomic_claim_restart_and_stale_worker_cannot_publish(tmp_path):
    settings, store = configured(tmp_path)
    job = store.enqueue("import", {}, "operator", "once")
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(lambda _: Store(settings).claim(now=100, lease_seconds=10), range(2))
        )
    claimed = [c for c in claims if c]
    assert len(claimed) == 1 and claimed[0]["id"] == job["id"]
    restarted = Store(settings).claim(now=111, lease_seconds=10)
    assert restarted["id"] == job["id"] and restarted["attempts"] == 2
    with pytest.raises(WorkflowError):
        store.finish(claimed[0], {"wrong": True}, now=112)
    store.finish(restarted, {"correct": True}, now=112)
    assert store.job(job["id"])["result"] == {"correct": True}
    assert store.claim(now=113) is None


def test_repeated_crash_is_terminal_and_cancel_survives_restart(tmp_path):
    settings, store = configured(tmp_path)
    bad = store.enqueue("import", {}, "operator", "bad")
    store.claim(now=100, lease_seconds=1)
    store.claim(now=102, lease_seconds=1)
    assert Store(settings).claim(now=104) is None
    assert store.job(bad["id"])["status"] == "failed"
    waiting = store.enqueue("import", {}, "operator", "cancel")
    store.cancel(waiting["id"])
    assert Store(settings).claim() is None
    assert store.job(waiting["id"])["status"] == "cancelled"


def test_running_cancel_prevents_publish_and_preserves_outcome(tmp_path):
    _, store = configured(tmp_path)
    item = store.enqueue("export", {}, "operator", "cancel-running")
    claim = store.claim()
    store.cancel(item["id"])
    with pytest.raises(WorkflowError):
        store.finish(claim, {"artifact_id": "must-not-publish"})
    store.fail(claim, "cancelled", retryable=False)
    assert store.job(item["id"])["status"] == "cancelled"


def test_migration_rejects_foreign_and_future_database(tmp_path):
    settings, store = configured(tmp_path)
    store.initialize()
    with sqlite3.connect(store.database) as connection:
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(WorkflowError):
        Store(settings).initialize()


def test_private_paths_and_secret_constraints(tmp_path):
    with pytest.raises(ValueError):
        Settings(root=tmp_path, workspace=tmp_path, token="x" * 40)
    with pytest.raises(ValueError):
        Settings(root=tmp_path / "private", workspace=tmp_path / "repo", token="short")


def test_retryable_failure_is_bounded_and_publication_rolls_back(tmp_path):
    _, store = configured(tmp_path)
    queued = store.enqueue("export", {}, "operator", "retry")
    first = store.claim()
    store.fail(first, "local_io_failure", retryable=True)
    assert store.job(queued["id"])["status"] == "queued"
    second = store.claim()

    def failed_publish(cx):
        cx.execute("UPDATE jobs SET result_json='{}' WHERE id=?", (queued["id"],))
        raise OSError("synthetic interrupted publication")

    with pytest.raises(OSError):
        store.finish(second, {"published": True}, publish=failed_publish)
    assert store.job(queued["id"])["result"] is None
    store.fail(second, "local_io_failure", retryable=True)
    assert store.job(queued["id"])["status"] == "failed"
    assert store.claim() is None
