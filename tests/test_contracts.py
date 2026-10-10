"""Contract tests use invented values only; no dataset access."""
import json
from uuid import uuid4

import pytest
from forestcheck.domain.contracts import (
    AnalysisRun,
    Dataset,
    EvaluationLabel,
    FieldValue,
    Finding,
    ModelVersion,
    Record,
    Review,
    Suggestion,
)
from pydantic import ValidationError


def examples():
    dataset, record, finding, suggestion, run, model = [str(uuid4()) for _ in range(6)]
    base = {"schema_version": "1.0"}
    field = dict(base, name="synthetic_height", original_value="001", normalized_value=1,
                 unit="synthetic_unit", state="observed",
                 source={"artifact_id": str(uuid4()), "table": "synthetic", "row": 2,
                         "column": "height"})
    return {
        Dataset: dict(base, id=dataset, name="Synthetic", created_at="2026-10-09T12:00:00Z"),
        FieldValue: field,
        Record: dict(base, id=record, dataset_id=dataset, fields=[field],
                     business_key={"stand": "001", "layer": "1", "element": "SYN"}),
        Finding: dict(base, id=finding, record_id=record, analysis_run_id=run,
                      field_name="synthetic_height", signal_kind="anomaly",
                      severity="warning", priority=2, review_status="unreviewed",
                      evidence=[{"code": "synthetic_check", "message": "Synthetic evidence"}],
                      anomaly_score={"value": -2.4, "scale": "synthetic_raw", "higher_is_worse": False}),
        Suggestion: dict(base, id=suggestion, finding_id=finding, record_id=record,
                         field_name="synthetic_height", proposed_value=2, unit="synthetic_unit",
                         rationale="Synthetic candidate"),
        Review: dict(base, id=str(uuid4()), finding_id=finding, reviewer_id="synthetic-reviewer",
                     reviewed_at="2026-10-09T12:00:00Z", error_assessment="confirmed_error",
                     suggestion_id=suggestion, suggestion_decision="rejected"),
        AnalysisRun: dict(base, id=run, dataset_id=dataset, status="pending",
                          created_at="2026-10-09T12:00:00Z", config_digest="0" * 64),
        ModelVersion: dict(base, id=model, name="Synthetic descriptor", model_version="0.0.1",
                          created_at="2026-10-09T12:00:00Z", artifact_id=str(uuid4()),
                          feature_schema_version="1.0", artifact_sha256="1" * 64),
    }


@pytest.mark.parametrize("contract", [Dataset, Record, FieldValue, Finding, Suggestion, Review,
                                     AnalysisRun, ModelVersion])
def test_entity_roundtrip_and_reject_incompatible_versions(contract):
    payload = examples()[contract]
    entity = contract.model_validate(payload)
    assert contract.model_validate_json(entity.model_dump_json()) == entity
    for version in ("0.9", "1.1", "2.0", 1, None):
        with pytest.raises(ValidationError):
            contract.model_validate(dict(payload, schema_version=version))
    payload.pop("schema_version")
    with pytest.raises(ValidationError):
        contract.model_validate(payload)


def test_field_preserves_original_and_missing_is_not_zero():
    payload = examples()[FieldValue]
    field = FieldValue.model_validate(payload)
    assert field.original_value == "001" and field.normalized_value == 1
    with pytest.raises(ValidationError):
        field.normalized_value = 7
    assert FieldValue.model_validate(dict(payload, normalized_value=0)).normalized_value == 0
    for state in ("missing", "not_applicable", "parse_error", "unknown"):
        assert FieldValue.model_validate(dict(payload, state=state, normalized_value=None)).state == state
        with pytest.raises(ValidationError):
            FieldValue.model_validate(dict(payload, state=state, normalized_value=0))
    with pytest.raises(ValidationError):
        FieldValue.model_validate(dict(payload, normalized_value=None))
    with pytest.raises(ValidationError):
        FieldValue.model_validate(dict(payload, normalized_value=float("nan")))


def test_record_identity_independent_of_business_key_and_no_labels_in_features():
    payload = examples()[Record]
    first = Record.model_validate(payload)
    second = Record.model_validate(dict(payload, id=str(uuid4())))
    assert first.business_key == second.business_key and first.id != second.id
    with pytest.raises(ValidationError):
        Record.model_validate(dict(payload, labels={"is_error": True}))
    label = EvaluationLabel(schema_version="1.0", record_id=first.id,
                            field_name="synthetic_height", is_error=True, split="test")
    assert "is_error" not in first.model_dump_json()
    assert label.record_id == first.id
    with pytest.raises(ValidationError):
        Record.model_validate(dict(payload, fields=payload["fields"] * 2))


def test_score_is_not_probability_and_rejects_nonfinite():
    payload = examples()[Finding]
    assert Finding.model_validate(payload).anomaly_score.value == -2.4
    with pytest.raises(ValidationError):
        Finding.model_validate(dict(payload, probability=0.9))
    payload["anomaly_score"]["value"] = float("inf")
    with pytest.raises(ValidationError):
        Finding.model_validate(payload)


def test_suggestion_does_not_mutate_record_and_review_axes_are_separate():
    payloads = examples()
    record = Record.model_validate(payloads[Record])
    before = record.model_dump_json()
    suggestion = Suggestion.model_validate(payloads[Suggestion])
    review = Review.model_validate(payloads[Review])
    assert suggestion.proposed_value == 2 and record.model_dump_json() == before
    assert review.error_assessment == "confirmed_error" and review.suggestion_decision == "rejected"
    with pytest.raises(ValidationError):
        Review.model_validate(dict(payloads[Review], suggestion_id=None))
    with pytest.raises(ValidationError):
        Review.model_validate(dict(payloads[Review], error_assessment="no_error",
                                   suggestion_decision="accepted"))


def test_unknown_fields_and_ambiguous_numbers_rejected():
    for contract, payload in examples().items():
        with pytest.raises(ValidationError):
            contract.model_validate(dict(payload, future_field="x"))
    payload = examples()[FieldValue]
    assert FieldValue.model_validate(dict(payload, original_value=True)).original_value is True
    assert isinstance(FieldValue.model_validate(dict(payload, original_value="2")).original_value, str)
    payload["source"]["row"] = 0
    with pytest.raises(ValidationError):
        FieldValue.model_validate(payload)


def test_run_lifecycle_and_model_feature_version():
    payloads = examples()
    with pytest.raises(ValidationError):
        AnalysisRun.model_validate(dict(payloads[AnalysisRun], status="completed"))
    with pytest.raises(ValidationError):
        ModelVersion.model_validate(dict(payloads[ModelVersion], feature_schema_version="2.0"))
    payload = payloads[Dataset]
    payload["created_at"] = "2026-10-09T12:00:00"
    with pytest.raises(ValidationError):
        Dataset.model_validate_json(json.dumps(payload))
