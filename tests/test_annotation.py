"""Blank independent decisions and deterministic sampling on synthetic IDs."""

from uuid import UUID

import pytest
from forestcheck.evaluation.annotation import AnnotationForm, build_annotation_queue
from pydantic import ValidationError
from test_corruption import cv_manifest_for, field, manifest_for, record


def inputs():
    row = record(field("x", 3, "m"))
    manifest = manifest_for(row)
    candidates = tuple(a.record_id for a in manifest.assignments if a.role in {"train", "tune"})
    return manifest, candidates


def test_queue_includes_unflagged_controls_but_no_automatic_answers():
    manifest, candidates = inputs()
    flagged = candidates[:3]
    queue = build_annotation_queue(
        candidates,
        flagged=flagged,
        manifest=manifest,
        seed="queue-independent",
        control_count=5,
        max_flagged=2,
    )
    assert len(queue.forms) == 7
    assert queue.split_digest == manifest.digest
    assert all(form.record_revision == 1 for form in queue.forms)
    assert sum(item.stratum == "control" for item in queue.sampling) == 5
    assert sum(item.stratum == "flagged" for item in queue.sampling) == 2
    assert all(form.error_assessment is None for form in queue.forms)
    assert all(form.correction_assessment is None for form in queue.forms)
    assert all(form.suggestion_decision is None for form in queue.forms)
    assert all("stratum" not in form.model_dump() for form in queue.forms)
    assert {item.record_id for item in queue.sampling if item.stratum == "control"}.isdisjoint(
        flagged
    )
    assert queue == build_annotation_queue(
        candidates,
        flagged=flagged,
        manifest=manifest,
        seed="queue-independent",
        control_count=5,
        max_flagged=2,
    )


def test_correction_assessment_is_independent_of_error_assessment():
    form = AnnotationForm(
        schema_version="1.0",
        id=UUID(int=1),
        record_id=UUID(int=2),
        suggestion_id=UUID(int=3),
        error_assessment="confirmed_error",
        correction_assessment="incorrect",
        suggestion_decision="rejected",
        reviewer_id="synthetic-expert",
        note="The error and remedy are different decisions.",
    )
    assert form.error_assessment == "confirmed_error" and form.correction_assessment == "incorrect"
    with pytest.raises(ValidationError):
        AnnotationForm.model_validate(form.model_dump() | {"suggestion_id": None})


def test_queue_can_audit_no_findings_with_unlabelled_controls():
    manifest, candidates = inputs()
    queue = build_annotation_queue(
        candidates,
        flagged=(),
        manifest=manifest,
        seed="no-findings",
        control_count=3,
        max_flagged=2,
    )
    assert len(queue.forms) == 3
    assert all(item.stratum == "control" for item in queue.sampling)
    assert all(
        form.error_assessment is None
        and form.correction_assessment is None
        and form.suggestion_decision is None
        for form in queue.forms
    )


def test_development_cv_can_receive_blank_manual_review_without_holdout_claim():
    row = record(field("x", 7, "m"))
    manifest = cv_manifest_for(row)
    candidates = tuple(a.record_id for a in manifest.assignments)
    queue = build_annotation_queue(
        candidates,
        flagged=candidates[:1],
        manifest=manifest,
        seed="cv-annotation",
        control_count=3,
        max_flagged=2,
    )
    assert len(queue.forms) == 4 and all(form.error_assessment is None for form in queue.forms)


def test_queue_rejects_holdout_unknown_candidates_and_no_control_sample():
    manifest, candidates = inputs()
    held = next(a.record_id for a in manifest.assignments if a.role == "holdout")
    with pytest.raises(ValueError):
        build_annotation_queue(
            (held, *candidates),
            flagged=candidates[:1],
            manifest=manifest,
            seed="q",
            control_count=2,
            max_flagged=2,
        )
    with pytest.raises(ValueError):
        build_annotation_queue(
            candidates,
            flagged=(UUID(int=9999),),
            manifest=manifest,
            seed="q",
            control_count=2,
            max_flagged=2,
        )
    with pytest.raises(ValueError):
        build_annotation_queue(
            candidates,
            flagged=candidates,
            manifest=manifest,
            seed="q",
            control_count=2,
            max_flagged=2,
        )
    with pytest.raises(ValueError):
        build_annotation_queue(
            candidates,
            flagged=candidates[:1],
            manifest=manifest,
            seed="q",
            control_count=0,
            max_flagged=2,
        )


def test_unknown_annotation_versions_and_fields_are_rejected():
    base = {"schema_version": "1.0", "id": UUID(int=1), "record_id": UUID(int=2)}
    for update in ({"schema_version": "2.0"}, {"auto_expert_answer": True}, {"record_revision": 0}):
        with pytest.raises(ValidationError):
            AnnotationForm.model_validate(base | update)
