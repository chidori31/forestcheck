import os
import sqlite3
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from forestcheck.api.app import create_app
from forestcheck.api.contracts import (
    AnalysisProfile,
    AnalysisRequest,
    MappingRequest,
    ReviewRequest,
    VersionRequest,
)
from forestcheck.api.execution import Worker
from forestcheck.api.service import Service
from forestcheck.storage.application import Store, WorkflowError
from test_backend_api import auth
from test_backend_store import configured
from test_backend_workflow import analysed, fixture_mapping, fixture_profile


def test_chunked_upload_and_json_limits_cleanup_partial_file(tmp_path):
    settings, store = configured(tmp_path)
    small = replace(settings, max_upload_bytes=32, max_json_bytes=32)
    with TestClient(
        create_app(small), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as client:
        upload = client.post(
            "/api/v1/datasets",
            headers=auth(small, "large") | {"Content-Type": "text/csv", "X-Filename": "safe.csv"},
            content=iter([b"column\n", b"x" * 40]),
        )
        assert upload.status_code == 413
        body = client.post(
            "/api/v1/analyses",
            headers=auth(small, "large-json") | {"Content-Type": "application/json"},
            content=iter([b'{"x":"', b"x" * 100, b'"}']),
        )
        assert body.status_code == 413
    assert not list((store.root / "staging").iterdir())
    assert Service(store).datasets()["total"] == 0


def test_registry_rejects_hardlinks_traversal_and_changed_bytes(tmp_path):
    _, store = configured(tmp_path)
    source = store.root / "uploads/source.csv"
    source.write_bytes(b"x\n1\n")
    artifact = store.artifact(source, "text/csv")
    link = tmp_path / "linked.csv"
    os.link(source, link)
    with pytest.raises(WorkflowError):
        store.resolve(artifact)
    link.unlink()
    with store.transaction() as cx:
        cx.execute("UPDATE artifacts SET relative_path='../outside.csv' WHERE id=?", (artifact,))
    with pytest.raises(WorkflowError):
        store.resolve(artifact)


def test_worker_restart_completes_real_import_without_duplicate_version(tmp_path):
    settings, store = configured(tmp_path)
    service = Service(store)
    path = store.root / "staging/source.csv"
    path.write_bytes(b"stand;per;area;total;label\n001;10;2;;synthetic\n")
    dataset = service.register_upload(path, "source.csv", "operator", "u")
    job = service.confirm_mapping(
        dataset["id"],
        MappingRequest(schema_version="1.0", expected_version=1, mapping=fixture_mapping()),
        "operator",
        "m",
    )
    abandoned = store.claim(now=100, lease_seconds=1)
    assert abandoned["id"] == job["id"]
    assert Worker(Store(settings)).run_once()
    assert service.job(job["id"])["status"] == "succeeded"
    assert not Worker(Store(settings)).run_once()
    with store.transaction() as cx:
        assert cx.execute("SELECT count(*) FROM versions").fetchone()[0] == 1


def test_append_only_review_history(tmp_path):
    _, store, service, _, _, _, _, detail, suggestion = analysed(tmp_path)
    event = service.review(
        ReviewRequest(
            schema_version="1.0",
            finding_id=detail["finding"]["id"],
            expected_version=0,
            error_assessment="uncertain",
            suggestion_id=suggestion["suggestion"]["id"],
            suggestion_decision="deferred",
        ),
        "operator",
        "review",
    )
    for sql in (
        "DELETE FROM review_events WHERE id=?",
        "UPDATE review_events SET operator='forged' WHERE id=?",
    ):
        with pytest.raises(sqlite3.IntegrityError), store.transaction() as cx:
            cx.execute(sql, (event["id"],))
    assert service.reviews(detail["finding"]["id"])["total"] == 1


def test_combined_patches_new_key_collision_is_atomic_failure(tmp_path):
    _, store = configured(tmp_path)
    service = Service(store)
    mapping = fixture_mapping()
    base = fixture_profile(mapping)
    # Purely synthetic uniqueness policy: no forestry uniqueness is implied.
    unique = {
        "schema_version": "1.0",
        "id": "synthetic-unique-total",
        "version": "1",
        "record_levels": ["stand"],
        "dependencies": ["total"],
        "source": {
            "kind": "synthetic",
            "reference_id": "synthetic-test",
            "locator": "atomic collision",
        },
        "severity": "error",
        "priority": 1,
        "explanation": "Synthetic collision check",
        "check": {"kind": "unique_key", "fields": ["total"], "grain_confirmed": True},
    }
    profile = AnalysisProfile.model_validate(base.model_dump() | {"rules": [*base.rules, unique]})
    service.register_profile(profile)
    path = store.root / "staging/source.csv"
    path.write_bytes(b"stand;per;area;total;label\n001;10;2;;synthetic\n002;10;2;;synthetic\n")
    dataset = service.register_upload(path, "source.csv", "operator", "u")
    service.confirm_mapping(
        dataset["id"],
        MappingRequest(schema_version="1.0", expected_version=1, mapping=mapping),
        "operator",
        "m",
    )
    Worker(store).run_once()
    version = service.dataset(dataset["id"])["current_version_id"]
    analysis = service.start_analysis(
        AnalysisRequest(
            schema_version="1.0", version_id=version, profile_id=profile.id, use_models=False
        ),
        "operator",
        "a",
    )
    Worker(store).run_once()
    events = []
    for item in service.findings(analysis["id"])["items"]:
        detail = service.finding(item["finding"]["id"])
        suggestion = detail["suggestions"][0]
        assert suggestion["patch"] and suggestion["suggestion"]["proposed_value"] == 20
        event = service.review(
            ReviewRequest(
                schema_version="1.0",
                finding_id=detail["finding"]["id"],
                expected_version=0,
                error_assessment="confirmed_error",
                suggestion_id=suggestion["suggestion"]["id"],
                suggestion_decision="accepted",
            ),
            "operator",
            str(uuid4()),
        )
        events.append(event["id"])
    assert len(events) == 2
    job = service.create_version(
        dataset["id"],
        VersionRequest(
            schema_version="1.0",
            base_version_id=version,
            analysis_id=analysis["id"],
            expected_version=service.dataset(dataset["id"])["etag"],
            review_event_ids=tuple(events),
        ),
        "operator",
        "v",
    )
    Worker(store).run_once()
    assert service.job(job["id"])["status"] == "failed"
    assert service.job(job["id"])["error_code"] == "patch_creates_rule_violation"
    assert service.dataset(dataset["id"])["current_version_id"] == version
    assert all(
        next(f for f in r["fields"] if f["name"] == "total")["state"] == "missing"
        for r in service.records(version)["items"]
    )


def test_local_profile_runs_trusted_synthetic_models_without_mock(tmp_path):
    from forestcheck.api.admin import register_forestcheck
    from forestcheck.ml.data import LocalData
    from forestcheck.repair.local import prepare
    from test_recommendation_local import environment

    source_root = tmp_path / "previous"
    source_root.mkdir()
    config, repo, _, source, _ = environment(source_root)
    prepare(config, workspace_root=repo)
    app_root = tmp_path / "backend"
    app_root.mkdir()
    settings, store = configured(app_root)
    profile_id = register_forestcheck(store, config, workspace_root=repo)
    data = LocalData(config, workspace_root=repo)
    service = Service(store)
    upload = store.root / "staging/source.csv"
    upload.write_bytes(source.read_bytes())
    dataset = service.register_upload(upload, "source.csv", "operator", "upload")
    service.confirm_mapping(
        dataset["id"],
        MappingRequest(schema_version="1.0", expected_version=1, mapping=data.mapping),
        "operator",
        "mapping",
    )
    Worker(store).run_once()
    version = service.dataset(dataset["id"])["current_version_id"]
    run = service.start_analysis(
        AnalysisRequest(
            schema_version="1.0", version_id=version, profile_id=profile_id, use_models=True
        ),
        "operator",
        "model-run",
    )
    Worker(store).run_once()
    job = service.job(run["id"])
    assert job["status"] == "succeeded", job["error_code"]
    assert job["result"]["model_count"] == len(data.plan.targets)
    assert job["result"]["comparable_reference"] == "train_only_registered_snapshot"
    assert sum(job["result"]["reference_queries"].values()) > 0


def test_version_streams_unmodified_rows_beyond_analysis_memory_limit(tmp_path):
    settings, store, service, dataset, version, _, run, detail, suggestion = analysed(tmp_path)
    event = service.review(
        ReviewRequest(
            schema_version="1.0",
            finding_id=detail["finding"]["id"],
            expected_version=0,
            error_assessment="confirmed_error",
            suggestion_id=suggestion["suggestion"]["id"],
            suggestion_decision="accepted",
        ),
        "operator",
        "review",
    )
    job = service.create_version(
        dataset["id"],
        VersionRequest(
            schema_version="1.0",
            base_version_id=version,
            analysis_id=run["id"],
            expected_version=service.dataset(dataset["id"])["etag"],
            review_event_ids=(event["id"],),
        ),
        "operator",
        "version",
    )
    Worker(Store(replace(settings, max_analysis_records=1))).run_once()
    outcome = service.job(job["id"])
    assert outcome["status"] == "succeeded", outcome["error_code"]
    assert service.records(outcome["result"]["version_id"])["total"] == 2


def test_two_accepted_patches_cannot_jointly_break_declared_aggregate(tmp_path):
    from forestcheck.repair.models import AggregateRelation
    from forestcheck.rules.models import RuleDefinition

    _, store = configured(tmp_path)
    service = Service(store)
    mapping = fixture_mapping()
    path = store.root / "staging/source.csv"
    path.write_bytes(
        b"stand;per;area;total;label\n001;10;2;19.6;x\n002;10;2;19.6;y\n003;19.5;2;39;z\n"
    )
    dataset = service.register_upload(path, "source.csv", "operator", "u")
    service.confirm_mapping(
        dataset["id"],
        MappingRequest(schema_version="1.0", expected_version=1, mapping=mapping),
        "operator",
        "m",
    )
    Worker(store).run_once()
    version = service.dataset(dataset["id"])["current_version_id"]
    records = service.records(version)["items"]
    base = fixture_profile(mapping)
    balance = base.rules[1].model_dump()
    balance["check"]["absolute_tolerance"] = "0.5"
    rounding = RuleDefinition(
        schema_version="1.0",
        id="total-step",
        version="1",
        record_levels=("stand",),
        dependencies=("total",),
        source=base.rules[0].source,
        severity="warning",
        priority=1,
        explanation="Synthetic representation step",
        check=dict(kind="rounding", field="total", unit="m3", step="1"),
    )
    relation = AggregateRelation(
        schema_version="1.0",
        id="synthetic-sum",
        version="1",
        component_record_ids=(records[0]["id"], records[1]["id"]),
        component_field="total",
        total_record_id=records[2]["id"],
        total_field="total",
        unit="m3",
        quantity_kind="synthetic_live_stock",
        component_level="stand",
        total_level="stand",
        relationship_confirmed=True,
        absolute_tolerance="0.6",
        source=base.rules[0].source,
    )
    profile = AnalysisProfile.model_validate(
        base.model_dump() | {"rules": (base.rules[0], balance, rounding), "aggregates": (relation,)}
    )
    service.register_profile(profile)
    run = service.start_analysis(
        AnalysisRequest(
            schema_version="1.0", version_id=version, profile_id=profile.id, use_models=False
        ),
        "operator",
        "a",
    )
    Worker(store).run_once()
    events = []
    for item in service.findings(run["id"])["items"]:
        detail = service.finding(item["finding"]["id"])
        recommendation = detail["suggestions"][0]
        assert recommendation["patch"], recommendation["suggestion"]["reason"]
        event = service.review(
            ReviewRequest(
                schema_version="1.0",
                finding_id=detail["finding"]["id"],
                expected_version=0,
                error_assessment="confirmed_error",
                suggestion_id=recommendation["suggestion"]["id"],
                suggestion_decision="accepted",
            ),
            "operator",
            str(uuid4()),
        )
        events.append(event["id"])
    assert len(events) == 2
    job = service.create_version(
        dataset["id"],
        VersionRequest(
            schema_version="1.0",
            base_version_id=version,
            analysis_id=run["id"],
            expected_version=service.dataset(dataset["id"])["etag"],
            review_event_ids=tuple(events),
        ),
        "operator",
        "v",
    )
    Worker(store).run_once()
    assert service.job(job["id"])["error_code"] == "patch_aggregate_not_validated"
    assert service.dataset(dataset["id"])["current_version_id"] == version
