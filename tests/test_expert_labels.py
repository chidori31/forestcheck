"""Expert import boundary tests; all records and answers are hand-written synthetic data."""

import json
from uuid import UUID

import pytest
from forestcheck.domain.contracts import Record
from forestcheck.evaluation.expert_labels import (
    ExpertLabelBatch,
    ExpertLabelExposure,
    ExpertLabelSnapshot,
    import_expert_labels,
    report_preprocessing_compatibility,
)
from forestcheck.evaluation.split import Assignment, SplitIdentity, SplitManifest
from pydantic import ValidationError
from test_corruption import field, record


def fixture(*, role="train", new_object=False):
    row = record(field("height", "001", "m"), field("diameter", 4, "cm"))
    identity = SplitIdentity(
        schema_version="1.0", record_id=row.id, dataset_id=row.dataset_id,
        group_id="a" * 64, parent_id="b" * 64,
    )
    frozen = identity.model_dump()
    if new_object:
        frozen |= {"record_id": UUID(int=99), "group_id": "c" * 64, "parent_id": "d" * 64}
    manifest = SplitManifest(
        schema_version="1.0", seed="synthetic", strategy="official",
        grouping_basis="synthetic groups", inputs={"source": "e" * 64},
        assignments=(Assignment(**frozen, role=role),), official_source="synthetic declaration",
    )
    snapshot = ExpertLabelSnapshot(
        schema_version="1.0", dataset_id=row.dataset_id, dataset_revision=1,
        preprocessing_version="expert-preprocessing-1.0.0",
        records=(row,), identities=(identity,), source_sha256={str(UUID(int=1)): "e" * 64},
    )
    payload = {
        "schema_version": "1.0", "import_version": "expert-labels-1.0.0",
        "dataset_id": row.dataset_id, "dataset_revision": 1,
        "snapshot_sha256": snapshot.digest, "split_digest": manifest.digest,
        "preprocessing_version": snapshot.preprocessing_version, "purpose": "dev",
        "labels": [{
            "schema_version": "1.0", "id": UUID(int=10), "record_id": row.id,
            "record_revision": 1, "field_name": "height", "object_id": identity.parent_id,
            "source": row.fields[0].source.model_dump(), "original_value": "001",
            "error_assessment": "confirmed_error", "error_kind": "synthetic transcription",
            "reviewer_id": "synthetic reviewer", "note": "A synthetic judgement, not real truth.",
        }],
    }
    exposure = ExpertLabelExposure(
        schema_version="1.0", split_digest=manifest.digest,
        opened_holdout_groups=(), tuning_identities=(), object_links=(),
        unresolved_linkage=False,
    )
    return row, snapshot, manifest, exposure, payload


def perform(payload, snapshot, manifest, exposure):
    return import_expert_labels(
        ExpertLabelBatch.model_validate(payload), snapshot=snapshot,
        manifest=manifest, exposure=exposure,
    )


def test_import_keeps_partial_labels_separate_and_does_not_invent_negative_answers():
    row, snapshot, manifest, exposure, payload = fixture()
    before = row.model_dump_json(), snapshot.model_dump_json(), manifest.model_dump_json()
    result = perform(payload, snapshot, manifest, exposure)
    assert len(result.labels) == 1
    assert result.labels[0].original_value == "001"
    assert result.labels[0].correction_assessment is None
    assert result.labels[0].suggestion_decision is None
    assert result.report.confirmed_errors == 1
    assert result.report.confirmed_no_errors == 0
    assert result.report.unlabelled_fields == 1
    assert result.report.metric_status == "NOT_RUN"
    assert result.report.overlaps[0].roles == ("train",)
    assert result.report.dataset_id == row.dataset_id
    assert result.report.dataset_revision == row.dataset_revision
    assert result.report.preprocessing_version == "expert-preprocessing-1.0.0"
    assert before == (row.model_dump_json(), snapshot.model_dump_json(), manifest.model_dump_json())


def test_optional_correct_value_preserves_absent_versus_explicit_null_and_zero():
    _, snapshot, manifest, exposure, payload = fixture()
    result = perform(payload, snapshot, manifest, exposure)
    assert result.labels[0].correct_value is None
    payload["labels"][0]["correct_value"] = {"schema_version": "1.0", "value": None, "unit": "m"}
    result = perform(payload, snapshot, manifest, exposure)
    assert result.labels[0].correct_value.value is None
    serialized = result.model_dump_json()
    assert json.loads(serialized)["labels"][0]["correct_value"]["value"] is None
    payload["labels"][0]["correct_value"]["value"] = 0
    assert perform(payload, snapshot, manifest, exposure).labels[0].correct_value.value == 0


@pytest.mark.parametrize("update", [
    {"schema_version": "2.0"}, {"import_version": "expert-labels-2.0.0"},
    {"fake_truth": True}, {"purpose": "test"},
])
def test_import_rejects_unknown_versions_fields_and_undeclared_purpose(update):
    _, snapshot, manifest, exposure, payload = fixture()
    with pytest.raises(ValidationError):
        perform(payload | update, snapshot, manifest, exposure)


@pytest.mark.parametrize("update", [
    {"dataset_id": UUID(int=55)}, {"dataset_revision": 2},
    {"snapshot_sha256": "f" * 64}, {"split_digest": "f" * 64},
    {"preprocessing_version": "other-1.0.0"},
])
def test_stale_batch_binding_is_rejected_before_labels_are_returned(update):
    _, snapshot, manifest, exposure, payload = fixture()
    with pytest.raises(ValueError, match="binding"):
        perform(payload | update, snapshot, manifest, exposure)


@pytest.mark.parametrize("update", [
    {"record_id": UUID(int=66)}, {"record_revision": 2}, {"object_id": "f" * 64},
    {"field_name": "unavailable"}, {"original_value": 1}, {"original_value": "1"},
])
def test_wrong_record_object_field_revision_or_before_value_cannot_be_imported(update):
    _, snapshot, manifest, exposure, payload = fixture()
    payload["labels"][0].update(update)
    with pytest.raises(ValueError):
        perform(payload, snapshot, manifest, exposure)


@pytest.mark.parametrize("coordinate,value", [
    ("artifact_id", UUID(int=9)), ("table", "other sheet"), ("row", 2), ("column", "other"),
])
def test_source_cell_must_match_all_coordinates(coordinate, value):
    _, snapshot, manifest, exposure, payload = fixture()
    payload["labels"][0]["source"][coordinate] = value
    with pytest.raises(ValueError, match="source"):
        perform(payload, snapshot, manifest, exposure)


def test_error_and_correction_assessments_remain_independent_and_suggestion_is_bound():
    row, snapshot, manifest, exposure, payload = fixture()
    label = payload["labels"][0]
    label.update(suggestion_id=UUID(int=19), correction_assessment="incorrect",
                 suggestion_decision="rejected")
    with pytest.raises(ValueError, match="suggestion"):
        perform(payload, snapshot, manifest, exposure)
    suggestion = {
        "schema_version": "1.0", "id": UUID(int=19), "record_id": row.id,
        "record_revision": 1, "field_name": "height", "finding_id": UUID(int=20),
        "proposed_value": 2, "unit": "m", "rationale": "synthetic candidate",
    }
    snapshot = ExpertLabelSnapshot.model_validate(snapshot.model_dump() | {"suggestions": [suggestion]})
    payload["snapshot_sha256"] = snapshot.digest
    result = perform(payload, snapshot, manifest, exposure)
    assert result.labels[0].error_assessment == "confirmed_error"
    assert result.labels[0].correction_assessment == "incorrect"
    assert result.labels[0].suggestion_decision == "rejected"
    label["suggestion_decision"] = "accepted"
    with pytest.raises(ValidationError):
        perform(payload, snapshot, manifest, exposure)


def test_duplicate_annotation_id_or_same_cell_is_not_silently_overwritten():
    _, snapshot, manifest, exposure, payload = fixture()
    payload["labels"].append(dict(payload["labels"][0], id=UUID(int=11)))
    with pytest.raises(ValueError, match="duplicate"):
        perform(payload, snapshot, manifest, exposure)


@pytest.mark.parametrize("role", ["train", "tune", "calibration"])
def test_known_frozen_object_overlap_blocks_independent_evaluation(role):
    _, snapshot, manifest, exposure, payload = fixture(role=role)
    payload["purpose"] = "independent_evaluation"
    with pytest.raises(ValueError, match="independent"):
        perform(payload, snapshot, manifest, exposure)


def test_new_record_id_in_old_object_still_blocks_independence():
    _, snapshot, manifest, exposure, payload = fixture()
    row = Record.model_validate(snapshot.records[0].model_dump() | {"id": UUID(int=77)})
    identity = SplitIdentity.model_validate(snapshot.identities[0].model_dump() | {"record_id": row.id})
    snapshot = ExpertLabelSnapshot.model_validate(snapshot.model_dump() | {"records": [row], "identities": [identity]})
    payload["snapshot_sha256"] = snapshot.digest
    payload["labels"][0]["record_id"] = row.id
    payload["purpose"] = "independent_evaluation"
    with pytest.raises(ValueError, match="independent"):
        perform(payload, snapshot, manifest, exposure)


def test_prior_tuning_and_confirmed_links_block_new_objects_with_new_ids():
    _, snapshot, manifest, exposure, payload = fixture(new_object=True)
    payload["purpose"] = "independent_evaluation"
    result = perform(payload, snapshot, manifest, exposure)
    assert result.report.independent_evaluation_eligible
    exposure = ExpertLabelExposure.model_validate(exposure.model_dump() | {
        "tuning_identities": [snapshot.identities[0]],
    })
    with pytest.raises(ValueError, match="independent"):
        perform(payload, snapshot, manifest, exposure)
    exposure = ExpertLabelExposure.model_validate(exposure.model_dump() | {
        "tuning_identities": [], "object_links": [{
            "schema_version": "1.0", "object_id": snapshot.identities[0].parent_id,
            "related_record_id": UUID(int=99), "basis": "synthetic full bundle copy proof",
        }],
    })
    with pytest.raises(ValueError, match="independent"):
        perform(payload, snapshot, manifest, exposure)


@pytest.mark.parametrize("shared_identity", ["record_id", "parent_id", "group_id", "duplicate_id"])
def test_confirmed_link_propagates_tuning_of_related_holdout_identity(shared_identity):
    _, snapshot, manifest, exposure, payload = fixture(role="holdout", new_object=True)
    frozen = manifest.assignments[0].model_dump() | {"duplicate_id": "7" * 64}
    manifest = SplitManifest.model_validate(manifest.model_dump() | {"assignments": [frozen]})
    tuning = {
        "schema_version": "1.0", "record_id": UUID(int=101), "dataset_id": snapshot.dataset_id,
        "parent_id": "1" * 64, "group_id": "2" * 64, "duplicate_id": None,
        shared_identity: frozen[shared_identity],
    }
    exposure = ExpertLabelExposure.model_validate(exposure.model_dump() | {
        "split_digest": manifest.digest, "tuning_identities": [tuning],
        "object_links": [{
            "schema_version": "1.0", "object_id": snapshot.identities[0].parent_id,
            "related_record_id": frozen["record_id"], "basis": "synthetic confirmed copy",
        }],
    })
    payload.update(split_digest=manifest.digest, purpose="review_only")
    report = perform(payload, snapshot, manifest, exposure).report
    assert report.overlaps[0].roles == ("holdout",)
    assert report.overlaps[0].tuning_overlap
    payload["purpose"] = "independent_evaluation"
    with pytest.raises(ValueError, match="independent"):
        perform(payload, snapshot, manifest, exposure)


@pytest.mark.parametrize("reverse_second_link", [False, True])
@pytest.mark.parametrize("exposure_kind", ["train", "calibration", "opened", "tuned", "prior"])
def test_confirmed_links_are_undirected_and_transitive_for_exposure(exposure_kind, reverse_second_link):
    _, snapshot, manifest, exposure, payload = fixture(role="holdout", new_object=True)
    first = manifest.assignments[0]
    last = first.model_dump() | {
        "record_id": UUID(int=102), "parent_id": "3" * 64, "group_id": "4" * 64,
        "role": "train" if exposure_kind in {"train", "prior"} else
                "calibration" if exposure_kind == "calibration" else "holdout",
    }
    manifest = SplitManifest.model_validate(manifest.model_dump() | {
        "assignments": [first, last],
        "prior_exposed_groups": [last["group_id"]] if exposure_kind == "prior" else [],
    })
    last_identity = {key: val for key, val in last.items() if key not in {"role", "fold"}}
    links = [{
        "schema_version": "1.0", "object_id": snapshot.identities[0].parent_id,
        "related_record_id": first.record_id, "basis": "synthetic B to A relation",
    }, {
        "schema_version": "1.0",
        "object_id": last["parent_id"] if reverse_second_link else first.parent_id,
        "related_record_id": first.record_id if reverse_second_link else last["record_id"],
        "basis": "synthetic A to C relation, either orientation",
    }]
    exposure = ExpertLabelExposure.model_validate(exposure.model_dump() | {
        "split_digest": manifest.digest, "object_links": links,
        "opened_holdout_groups": [last["group_id"]] if exposure_kind == "opened" else [],
        "tuning_identities": [last_identity] if exposure_kind == "tuned" else [],
    })
    payload.update(split_digest=manifest.digest, purpose="review_only")
    overlap = perform(payload, snapshot, manifest, exposure).report.overlaps[0]
    assert overlap.related_record_ids == (UUID(int=99), UUID(int=102))
    assert overlap.opened_holdout == (exposure_kind == "opened")
    assert overlap.tuning_overlap == (exposure_kind in {"train", "tuned", "prior"})
    payload["purpose"] = "independent_evaluation"
    with pytest.raises(ValueError, match="independent"):
        perform(payload, snapshot, manifest, exposure)


def test_transitive_exposure_does_not_contaminate_disconnected_objects():
    _, snapshot, manifest, exposure, payload = fixture(role="holdout", new_object=True)
    frozen = manifest.assignments[0]
    exposure = ExpertLabelExposure.model_validate(exposure.model_dump() | {
        "tuning_identities": [{key: val for key, val in frozen.model_dump().items()
                               if key not in {"role", "fold"}}],
        "object_links": [{
            "schema_version": "1.0", "object_id": "1" * 64,
            "related_record_id": frozen.record_id, "basis": "unrelated synthetic relation",
        }],
    })
    payload["purpose"] = "independent_evaluation"
    report = perform(payload, snapshot, manifest, exposure).report
    assert report.independent_evaluation_eligible
    assert not report.overlaps[0].tuning_overlap
    assert report.overlaps[0].related_record_ids == ()


def test_opened_holdout_and_unresolved_links_cannot_be_claimed_independent():
    _, snapshot, manifest, exposure, payload = fixture(role="holdout")
    payload["purpose"] = "independent_evaluation"
    assert perform(payload, snapshot, manifest, exposure).report.independent_evaluation_eligible
    for change in ({"opened_holdout_groups": ["a" * 64]}, {"unresolved_linkage": True}):
        changed = ExpertLabelExposure.model_validate(exposure.model_dump() | change)
        with pytest.raises(ValueError, match="independent"):
            perform(payload, snapshot, manifest, changed)


def test_holdout_is_review_only_and_cannot_be_reassigned_to_development():
    _, snapshot, manifest, exposure, payload = fixture(role="holdout")
    with pytest.raises(ValueError, match="holdout"):
        perform(payload, snapshot, manifest, exposure)
    payload["purpose"] = "review_only"
    result = perform(payload, snapshot, manifest, exposure)
    assert result.report.overlaps[0].roles == ("holdout",)
    assert not result.report.independent_evaluation_eligible


def test_snapshot_rejects_unbound_record_or_missing_artifact_hash():
    _, snapshot, _, _, _ = fixture()
    for change in ({"identities": []}, {"source_sha256": {}}, {"dataset_revision": 2}):
        with pytest.raises(ValidationError):
            ExpertLabelSnapshot.model_validate(snapshot.model_dump() | change)


def test_v2_suggestion_cannot_claim_another_before_value_or_dataset_binding():
    row, snapshot, _, _, _ = fixture()
    binding = {
        "schema_version": "1.0", "dataset_id": row.dataset_id, "dataset_revision": 1,
        "source_sha256": "e" * 64, "snapshot_sha256": "1" * 64, "mapping_sha256": "2" * 64,
    }
    suggestion = {
        "schema_version": "2.0", "id": UUID(int=19), "record_id": row.id,
        "record_revision": 1, "field_name": "height", "finding_id": UUID(int=20),
        "proposed_value": None, "unit": "m", "rationale": "synthetic abstention",
        "before": row.fields[0].model_dump(), "binding": binding,
        "method": None, "evidence": [], "status": "refused", "refusal_reason": "synthetic",
        "limitations": [], "uncertainty": {"category": "unavailable", "basis": "synthetic"},
        "support_families": [], "explanation": {
            "observed": "synthetic", "trigger": [], "comparisons": [], "applicability": [],
            "considered_range": "none", "missing": [], "specialist_actions": [],
        },
    }
    with pytest.raises(ValidationError, match="suggestion"):
        ExpertLabelSnapshot.model_validate(snapshot.model_dump() | {"suggestions": [suggestion]})
    full = snapshot.model_dump() | {"suggestions": [suggestion], "dataset_bindings": [binding]}
    validated = ExpertLabelSnapshot.model_validate(full)
    assert validated.suggestions[0].schema_version == "2.0"
    for bad in (
        suggestion | {"before": row.fields[0].model_dump() | {"original_value": "999"}},
        suggestion | {"binding": binding | {"mapping_sha256": "3" * 64}},
        suggestion | {"binding": binding | {"dataset_id": UUID(int=900)}},
    ):
        with pytest.raises(ValidationError, match="suggestion"):
            ExpertLabelSnapshot.model_validate(full | {"suggestions": [bad]})


def test_calibration_cannot_be_silently_reassigned_to_development():
    _, snapshot, manifest, exposure, payload = fixture(role="calibration")
    with pytest.raises(ValueError, match="calibration"):
        perform(payload, snapshot, manifest, exposure)


def test_known_duplicate_identity_connects_different_parent_and_group():
    _, snapshot, manifest, exposure, payload = fixture(new_object=True)
    manifest = SplitManifest.model_validate(manifest.model_dump() | {
        "assignments": [manifest.assignments[0].model_dump() | {"duplicate_id": "9" * 64}],
    })
    snapshot = ExpertLabelSnapshot.model_validate(snapshot.model_dump() | {
        "identities": [snapshot.identities[0].model_dump() | {"duplicate_id": "9" * 64}],
    })
    exposure = ExpertLabelExposure.model_validate(exposure.model_dump() | {"split_digest": manifest.digest})
    payload.update(snapshot_sha256=snapshot.digest, split_digest=manifest.digest,
                   purpose="independent_evaluation")
    with pytest.raises(ValueError, match="independent"):
        perform(payload, snapshot, manifest, exposure)


def test_semantic_changes_require_version_bump_and_report_old_artifacts_incompatible():
    artifacts = ({
        "schema_version": "1.0", "artifact_id": "synthetic-model", "artifact_kind": "model",
        "preprocessing_version": "preprocessing-1.0.0", "snapshot_sha256": "a" * 64,
        "mapping_sha256": "b" * 64,
    },)
    before = json.dumps(artifacts, sort_keys=True)
    common = {"snapshot_sha256": "a" * 64, "mapping_sha256": "b" * 64}
    unchanged = report_preprocessing_compatibility(artifacts, preprocessing_version="preprocessing-1.0.0", **common)
    assert unchanged.decisions[0].status == "compatible"
    with pytest.raises(ValueError, match="version"):
        report_preprocessing_compatibility(artifacts, preprocessing_version="preprocessing-1.0.0", changes=("height_semantics",), **common)
    changed = report_preprocessing_compatibility(artifacts, preprocessing_version="expert-preprocessing-1.0.0", changes=("height_semantics", "bundle_deduplication"), **common)
    assert changed.decisions[0].status == "incompatible"
    assert changed.decisions[0].reasons == ("preprocessing_version_changed",)
    assert changed.action == "preserve_old_artifacts_new_experiment_required"
    assert json.dumps(artifacts, sort_keys=True) == before


@pytest.mark.parametrize("key", ["snapshot_sha256", "mapping_sha256"])
def test_equal_semantic_version_does_not_hide_changed_snapshot_or_mapping(key):
    artifact = {
        "schema_version": "1.0", "artifact_id": "synthetic-result", "artifact_kind": "result",
        "preprocessing_version": "expert-preprocessing-1.0.0", "snapshot_sha256": "a" * 64,
        "mapping_sha256": "b" * 64,
    }
    requested = {name: artifact[name] for name in ("preprocessing_version", "snapshot_sha256", "mapping_sha256")}
    requested[key] = "c" * 64
    result = report_preprocessing_compatibility((artifact,), **requested)
    assert result.decisions[0].status == "incompatible"
