import json
from uuid import uuid4

import pytest
from forestcheck.api.contracts import (
    AnalysisProfile,
    AnalysisRequest,
    MappingRequest,
    ReviewRequest,
    VersionRequest,
)
from forestcheck.api.execution import Worker
from forestcheck.api.service import Service
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.repair.models import PrecisionPolicy
from forestcheck.rules.models import RuleDefinition, RuleSource
from forestcheck.storage.application import WorkflowError, digest
from test_backend_store import configured


def fixture_mapping():
    fields = [
        ("stand", "text", None, "stand"),
        ("per", "decimal_point", "m3/ha", None),
        ("area", "decimal_point", "ha", None),
        ("total", "decimal_point", "m3", None),
        ("label", "text", None, None),
    ]
    return ImportMapping(
        schema_version="1.0",
        mapping_version="synthetic-v1",
        source_format="csv",
        encoding="utf-8",
        delimiter=";",
        table="synthetic",
        record_level="stand",
        grain_basis="synthetic test only",
        columns=tuple(
            dict(
                position=i,
                source_header=name,
                field=name,
                transform=transform,
                unit=unit,
                key_role=role,
                basis="synthetic declared contract",
            )
            for i, (name, transform, unit, role) in enumerate(fields, 1)
        ),
    )


def fixture_profile(mapping):
    source = RuleSource(
        reference_id="synthetic-fixture", locator="test relationship", kind="synthetic"
    )
    required = RuleDefinition(
        schema_version="1.0",
        id="required-total",
        version="1",
        record_levels=("stand",),
        categories=("forest",),
        dependencies=("total",),
        source=source,
        severity="error",
        priority=1,
        explanation="Synthetic category requires total",
        check=dict(kind="required", field="total"),
    )
    balance = RuleDefinition(
        schema_version="1.0",
        id="stock-balance",
        version="1",
        record_levels=("stand",),
        dependencies=("per", "area", "total"),
        source=source,
        severity="error",
        priority=1,
        explanation="Synthetic compatible quantities",
        check=dict(
            kind="stock_balance",
            per_ha="per",
            area="area",
            total="total",
            quantity_levels=("stand", "stand", "stand"),
            stock_kinds=("live", "live"),
            relationship_confirmed=True,
            tolerance_kind="explicit",
            absolute_tolerance="0",
        ),
    )
    return AnalysisProfile(
        schema_version="1.0",
        id=uuid4(),
        name="Synthetic stock",
        version="1",
        mapping_sha256=digest(mapping.model_dump(mode="json")),
        record_level="stand",
        category="forest",
        source_kind="synthetic",
        rules=(required, balance),
        numeric_fields={"total": "m3"},
        precision=(PrecisionPolicy(field_name="total", unit="m3", step="1", source=source),),
    )


def environment(tmp_path):
    settings, store = configured(tmp_path)
    service = Service(store)
    mapping = fixture_mapping()
    profile = fixture_profile(mapping)
    service.register_profile(profile)
    upload = store.root / "staging" / (str(uuid4()) + ".csv")
    upload.write_bytes(b"stand;per;area;total;label\n001;10;2;;=1+1\n002;-3;0;0;ordinary\n")
    dataset = service.register_upload(upload, "synthetic.csv", "operator", "upload")
    queued = service.confirm_mapping(
        dataset["id"],
        MappingRequest(schema_version="1.0", expected_version=1, mapping=mapping),
        "operator",
        "mapping",
    )
    assert queued["status"] == "queued"
    assert Worker(store).run_once()
    version = service.dataset(dataset["id"])["current_version_id"]
    assert version
    return settings, store, service, dataset, version, profile


def analysed(tmp_path):
    settings, store, service, dataset, version, profile = environment(tmp_path)
    request = AnalysisRequest(
        schema_version="1.0", version_id=version, profile_id=profile.id, use_models=False
    )
    run = service.start_analysis(request, "operator", "analysis")
    assert Worker(store).run_once()
    assert service.job(run["id"])["status"] == "succeeded"
    findings = service.findings(run["id"])["items"]
    assert len(findings) == 1 and findings[0]["finding"]["signal_kind"] == "missing"
    detail = service.finding(findings[0]["finding"]["id"])
    suggestion = detail["suggestions"][0]
    assert suggestion["patch"] and suggestion["suggestion"]["proposed_value"] == 20
    return settings, store, service, dataset, version, profile, run, detail, suggestion


def test_real_components_full_cycle_preserves_original_and_records_operator(tmp_path):
    _, store, service, dataset, version, _, run, detail, suggestion = analysed(tmp_path)
    source = store.resolve(store.one("datasets", dataset["id"])["source_artifact_id"])
    original = source.read_bytes()
    review = ReviewRequest(
        schema_version="1.0",
        finding_id=detail["finding"]["id"],
        expected_version=0,
        error_assessment="confirmed_error",
        suggestion_id=suggestion["suggestion"]["id"],
        suggestion_decision="accepted",
        comment="Synthetic expert test",
    )
    event = service.review(review, "server-operator", "review-once")
    assert event["operator"] == "server-operator" and event["after"]["normalized_value"] == 20
    assert service.review(review, "server-operator", "review-once") == event
    assert service.dataset(dataset["id"])["current_version_id"] == version
    new = service.create_version(
        dataset["id"],
        VersionRequest(
            schema_version="1.0",
            base_version_id=version,
            analysis_id=run["id"],
            review_event_ids=(event["id"],),
            expected_version=service.dataset(dataset["id"])["etag"],
        ),
        "operator",
        "new-version",
    )
    assert Worker(store).run_once()
    outcome = service.job(new["id"])
    assert outcome["status"] == "succeeded"
    changed = service.records(outcome["result"]["version_id"])["items"]
    first = next(r for r in changed if r["business_key"]["stand"] == "001")
    value = next(f for f in first["fields"] if f["name"] == "total")
    assert value["normalized_value"] == 20 and value["original_value"] == ""
    assert source.read_bytes() == original
    assert len(service.reviews(detail["finding"]["id"])["items"]) == 1


def test_review_version_conflict_and_revocation_invalidate_queued_version(tmp_path):
    _, store, service, dataset, version, _, run, detail, suggestion = analysed(tmp_path)
    request = dict(
        schema_version="1.0",
        finding_id=detail["finding"]["id"],
        expected_version=0,
        error_assessment="confirmed_error",
        suggestion_id=suggestion["suggestion"]["id"],
        suggestion_decision="accepted",
    )
    event = service.review(ReviewRequest(**request), "operator", "first")
    with pytest.raises(WorkflowError) as error:
        service.review(ReviewRequest(**request), "operator", "stale")
    assert error.value.code == "version_conflict"
    queued = service.create_version(
        dataset["id"],
        VersionRequest(
            schema_version="1.0",
            base_version_id=version,
            analysis_id=run["id"],
            review_event_ids=(event["id"],),
            expected_version=service.dataset(dataset["id"])["etag"],
        ),
        "operator",
        "new",
    )
    revoked = service.review(
        ReviewRequest(
            **(
                request
                | dict(
                    expected_version=1,
                    action="revoke",
                    revoke_event_id=event["id"],
                    suggestion_decision="deferred",
                    error_assessment="uncertain",
                )
            )
        ),
        "operator",
        "revoke",
    )
    assert revoked["action"] == "revoke"
    Worker(store).run_once()
    assert service.job(queued["id"])["status"] == "failed"
    assert service.dataset(dataset["id"])["current_version_id"] == version
    assert len(service.reviews(detail["finding"]["id"])["items"]) == 2


def test_bound_profile_and_idempotent_analysis(tmp_path):
    _, store, service, _, version, profile = environment(tmp_path)
    req = AnalysisRequest(
        schema_version="1.0", version_id=version, profile_id=profile.id, use_models=False
    )
    first = service.start_analysis(req, "operator", "same")
    assert service.start_analysis(req, "operator", "same") == first
    Worker(store).run_once()
    result = service.job(first["id"])
    assert result["request"]["profile_sha256"] == digest(profile.model_dump(mode="json"))
    assert result["request"]["binding"]["snapshot_sha256"]
    with store.transaction() as cx:
        content = profile.model_dump(mode="json") | {"version": "tampered"}
        cx.execute(
            "UPDATE profiles SET content_json=? WHERE id=?", (json.dumps(content), str(profile.id))
        )
    with pytest.raises(WorkflowError):
        service.start_analysis(req, "operator", "altered")
