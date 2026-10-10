"""Independent synthetic checks for evaluation semantics, not model performance."""
from uuid import UUID

import pytest
from forestcheck.evaluation.scoring import (
    DetectionCase,
    NumericCase,
    ResourceMeasurement,
    ScoringProtocol,
    UncertaintyPlan,
    score_evaluation,
    score_frozen_cohort,
)
from forestcheck.evaluation.split import SplitIdentity, freeze_split, hash_value
from pydantic import ValidationError


def protocol(**changes):
    return ScoringProtocol.model_validate({
        "schema_version": "1.0", "mode": "real_expert",
        "label_basis": "expert_assessed", "evaluated_fields": ["height", "age"],
        "evaluated_method_id": "candidate", "complete_labels": True,
        "label_producer_ids": ["expert-a"], **changes,
    })


def detection(row, field="height", label=True, prediction=True, **changes):
    return DetectionCase.model_validate({
        "schema_version": "1.0", "record_id": str(UUID(int=row)), "group_id": f"g{row}",
        "field_name": field, "is_error": label, "predicted_error": prediction, **changes,
    })


def numeric(row, **changes):
    return NumericCase.model_validate({
        "schema_version": "1.0", "record_id": str(UUID(int=row)), "group_id": f"g{row}",
        "field_name": "height", "unit": "m", "reference": 10., "estimate": 12.,
        **changes,
    })


def test_complete_field_and_row_metrics_use_independent_truth():
    cases = (detection(1), detection(1, "age", False, False),
             detection(2, label=False), detection(2, "age", True, False))
    report = score_evaluation(protocol(), cases)
    assert report.fields[0].field_name == "age"
    assert report.pooled_fields.precision.value == .5
    assert report.pooled_fields.recall.value == .5
    assert report.pooled_fields.f1.value == .5
    assert report.rows.precision.value == 1
    assert report.rows.recall.value == 1
    assert report.sample_records == 2
    assert report.sample_groups == 2


def test_partial_labels_are_not_negative_and_do_not_publish_recall():
    cases = (detection(1), detection(1, "age", None, True),
             detection(2, label=False), detection(2, "age", None, False))
    report = score_evaluation(protocol(complete_labels=False), cases)
    assert report.pooled_fields.unlabelled == 2
    assert report.pooled_fields.false_positive == 1
    assert report.pooled_fields.precision.status == "NOT_RUN"
    assert report.pooled_fields.precision_on_judged.value == .5
    assert report.pooled_fields.recall.status == "NOT_RUN"
    assert report.pooled_fields.f1.status == "NOT_RUN"
    assert report.rows.unlabelled == 1
    assert report.rows.true_positive == 1


def test_false_complete_label_declaration_fails_closed():
    with pytest.raises(ValueError, match="complete_label_coverage_not_met"):
        score_evaluation(protocol(), (detection(1),))
    with pytest.raises(ValueError, match="complete_label_coverage_not_met"):
        score_evaluation(protocol(), (detection(1), detection(1, "age", None)))


def test_row_negative_requires_every_declared_field_to_be_judged():
    report = score_evaluation(protocol(complete_labels=False), (detection(1, label=False),))
    assert report.rows.unlabelled == 1
    assert report.rows.false_positive == 0


def test_abstentions_are_reported_and_count_as_missed_positive_for_recall():
    cases = (detection(1, prediction=None), detection(1, "age", False, False))
    report = score_evaluation(protocol(), cases)
    assert report.pooled_fields.abstained == 1
    assert report.pooled_fields.abstention_rate.value == .5
    assert report.pooled_fields.recall.value == 0
    assert report.pooled_fields.false_negative == 1
    assert report.pooled_fields.precision.status == "NOT_RUN"


def test_precision_at_k_requires_all_selected_cases_judged_and_explicit_queue():
    cases = (detection(1, queue_rank=1), detection(2, label=False, queue_rank=2),
             detection(3, label=None, queue_rank=3))
    p = protocol(evaluated_fields=["height"], complete_labels=False)
    assert score_evaluation(p, cases, queue_k=2).queue.precision_at_k.value == .5
    report = score_evaluation(p, cases, queue_k=3)
    assert report.queue.precision_at_k.status == "NOT_RUN"
    assert report.queue.judged == 2
    assert report.queue.selected == 3
    assert score_evaluation(p, (), queue_k=3).queue.precision_at_k.status == "NOT_RUN"


def test_numerical_errors_and_intervals_never_pool_units():
    cases = (numeric(1, interval_lower=9., interval_upper=13.),
             numeric(2, estimate=14., interval_lower=11., interval_upper=15.),
             numeric(3, unit="cm", reference=1000., estimate=900.),
             numeric(4, estimate=None))
    report = score_evaluation(protocol(complete_labels=False), numeric_cases=cases)
    cm, m = report.numeric
    assert (cm.unit, cm.mae.value) == ("cm", 100.)
    assert (m.unit, m.mae.value, m.medae.value) == ("m", 3., 3.)
    assert m.interval_coverage.value == .5
    assert m.mean_interval_width.value == 4.
    assert m.interval_coverage.denominator == 2
    assert m.abstention_rate.value == pytest.approx(1/3)
    assert m.reference_count == 3


def test_hidden_observation_mode_never_claims_detection_truth():
    p = protocol(mode="masked_observation", label_basis="hidden_observation",
                 complete_labels=False, label_producer_ids=[])
    report = score_evaluation(p, (detection(1),), (numeric(1),))
    assert report.reference_interpretation == "reconstruction_of_unverified_observation"
    assert report.rows.precision.status == "NOT_RUN"
    assert report.pooled_fields.recall.status == "NOT_RUN"
    assert report.numeric[0].mae.value == 2.


def test_synthetic_metrics_are_marked_as_synthetic_only():
    p = protocol(mode="synthetic_stress", label_basis="synthetic_truth",
                 evaluated_fields=["height"], label_producer_ids=["fixture"])
    report = score_evaluation(p, (detection(1),))
    assert report.mode == "synthetic_stress"
    assert report.reference_interpretation == "synthetic_truth_only"
    assert report.rows.recall.value == 1.


@pytest.mark.parametrize("changes", [
    {"schema_version": "2.0"}, {"unexpected": 1},
    {"mode": "real_expert", "label_basis": "synthetic_truth"},
    {"label_producer_ids": ["candidate"], "independent_judge_ids": []},
    {"label_producer_ids": ["candidate"], "independent_judge_ids": ["candidate"]},
])
def test_invalid_protocol_or_circular_sole_judge_rejected(changes):
    with pytest.raises(ValidationError):
        protocol(**changes)


def test_independent_assessment_can_audit_rule_generated_labels():
    assert protocol(label_producer_ids=["candidate"], independent_judge_ids=["expert-b"])


def test_duplicate_or_conflicting_identity_rejected():
    p = protocol(complete_labels=False)
    with pytest.raises(ValueError, match="duplicate_detection_case"):
        score_evaluation(p, (detection(1), detection(1)))
    with pytest.raises(ValueError, match="inconsistent_record_group"):
        score_evaluation(p, (detection(1), detection(1, "age", group_id="elsewhere")))
    with pytest.raises(ValueError, match="field_outside_protocol"):
        score_evaluation(p, (detection(1, "secret_label"),))
    with pytest.raises(ValueError, match="duplicate_queue_rank"):
        score_evaluation(p, (detection(1, queue_rank=1), detection(2, queue_rank=1)))


@pytest.mark.parametrize("changes", [
    {"interval_lower": 2., "interval_upper": 1.},
    {"interval_lower": 2.}, {"estimate": float("nan")}, {"unit": ""},
    {"reference": True}, {"schema_version": "9.0"},
])
def test_invalid_numeric_cases_rejected(changes):
    with pytest.raises(ValidationError):
        numeric(1, **changes)


def test_resources_are_unavailable_unless_explicitly_measured():
    p = protocol(complete_labels=False)
    empty = score_evaluation(p)
    assert empty.wall_seconds.status == "NOT_RUN"
    assert empty.peak_memory_bytes.status == "NOT_RUN"
    measurement = ResourceMeasurement(schema_version="1.0", wall_seconds=.4,
                                     measurement_method="perf_counter and process peak RSS",
                                     peak_memory_bytes=4096)
    report = score_evaluation(p, resources=measurement)
    assert report.wall_seconds.value == .4
    assert report.peak_memory_bytes.value == 4096
    with pytest.raises(ValidationError):
        ResourceMeasurement(schema_version="1.0", wall_seconds=.4)


def test_group_bootstrap_requires_justification_and_enough_groups():
    with pytest.raises(ValidationError):
        UncertaintyPlan(schema_version="1.0", independent_groups=True)
    plan = UncertaintyPlan(schema_version="1.0", independent_groups=True,
                           independence_basis="synthetic independent objects", seed=9,
                           resamples=100)
    p = protocol(evaluated_fields=["height"])
    short = score_evaluation(p, (detection(1),), uncertainty=plan)
    assert all(x.status == "NOT_RUN" for x in short.uncertainty)
    cases = tuple(detection(i, prediction=i % 2 == 0) for i in range(1, 11))
    a = score_evaluation(p, cases, uncertainty=plan)
    b = score_evaluation(p, cases, uncertainty=plan)
    assert a == b
    interval = next(x for x in a.uncertainty if x.metric == "row_recall")
    assert interval.status == "COMPUTED"
    assert interval.group_count == 10
    assert 0 <= interval.lower <= .5 <= interval.upper <= 1
    assert len(a.groups) == 10


def test_unlabelled_numerical_targets_and_no_estimate_are_not_invented():
    p = protocol(complete_labels=False)
    report = score_evaluation(p, numeric_cases=(numeric(1, reference=None),
                                              numeric(2, reference=None, estimate=None)))
    values = report.numeric[0]
    assert values.reference_count == 0
    assert values.mae.status == values.medae.status == "NOT_RUN"
    assert values.interval_coverage.status == "NOT_RUN"
    assert values.abstention_rate.value == .5
    assert report.rows.recall.status == "NOT_RUN"


def test_zero_and_rare_valid_values_do_not_become_missing_or_biological_errors():
    report = score_evaluation(protocol(complete_labels=False), numeric_cases=(
        numeric(1, reference=0., estimate=0.),
        numeric(2, reference=1234567., estimate=1234567.),
    ))
    assert report.numeric[0].mae.value == 0
    assert report.numeric[0].reference_count == 2
    assert report.numeric[0].abstention_rate.value == 0


def test_numeric_duplicate_and_cross_modality_group_conflict_rejected():
    p = protocol(complete_labels=False)
    with pytest.raises(ValueError, match="duplicate_numeric_case"):
        score_evaluation(p, numeric_cases=(numeric(1), numeric(1, unit="cm")))
    with pytest.raises(ValueError, match="inconsistent_record_group"):
        score_evaluation(p, (detection(1),), (numeric(1, group_id="different"),))


def test_scorer_report_preserves_declared_scope_and_judgement_provenance():
    p = protocol(evaluated_fields=["height"])
    report = score_evaluation(p, (detection(1),))
    assert report.protocol == p
    assert report.uncertainty_plan is None
    assert report.groups[0].rows.precision.value == 1
    with pytest.raises(ValidationError):
        type(report).model_validate({**report.model_dump(), "schema_version": "2.0"})


def split_manifest(count=20):
    identities = tuple(SplitIdentity(
        schema_version="1.0", record_id=UUID(int=i), dataset_id=UUID(int=999),
        group_id=hash_value(["group", i]), parent_id=hash_value(["parent", i]),
    ) for i in range(1, count + 1))
    return freeze_split(identities, seed="test-scorer-cohort", grouping_basis="synthetic objects",
                        inputs={"fixture": "a" * 64})


def cohort_cases(manifest, role="tune", fold=None):
    selected = tuple(a for a in manifest.assignments if a.role == role and a.fold == fold)
    return (tuple(a.record_id for a in selected),
            tuple(detection(a.record_id.int, group_id=a.group_id) for a in selected))


def test_manifest_bound_scoring_preserves_frozen_cohort_and_scope():
    manifest = split_manifest()
    ids, cases = cohort_cases(manifest)
    report = score_frozen_cohort(protocol(evaluated_fields=["height"]), manifest, ids, cases,
                                 role="tune")
    assert report.cohort.manifest_digest == manifest.digest
    assert report.cohort.role == "tune"
    assert report.cohort.record_count == len(ids)
    assert report.cohort.cohort_digest == hash_value(sorted(str(rid) for rid in ids))
    assert report.rows.recall.value == 1


def test_manifest_bound_scoring_catches_wholly_omitted_extra_and_misgrouped_records():
    manifest = split_manifest()
    ids, cases = cohort_cases(manifest)
    p = protocol(evaluated_fields=["height"])
    with pytest.raises(ValueError, match="cohort_records_missing_or_extra"):
        score_frozen_cohort(p, manifest, ids, cases[:-1], role="tune")
    with pytest.raises(ValueError, match="cohort_records_missing_or_extra"):
        score_frozen_cohort(p, manifest, ids, (*cases, detection(999)), role="tune")
    wrong_group = cases[0].model_copy(update={"group_id": "wrong"})
    with pytest.raises(ValueError, match="case_group_does_not_match_manifest"):
        score_frozen_cohort(p, manifest, ids, (wrong_group, *cases[1:]), role="tune")
    with pytest.raises(ValueError, match="empty_cohort"):
        score_frozen_cohort(p, manifest, (), (), role="tune")
    with pytest.raises(ValueError, match="duplicate_cohort_record"):
        score_frozen_cohort(p, manifest, (*ids, ids[0]), cases, role="tune")


def test_holdout_has_no_stage5_score_override():
    manifest = split_manifest()
    ids, cases = cohort_cases(manifest, "holdout")
    p = protocol(evaluated_fields=["height"])
    with pytest.raises(ValueError, match="holdout_reserved_for_stage10"):
        score_frozen_cohort(p, manifest, ids, cases, role="holdout")
    with pytest.raises(ValueError, match="role_not_allowed"):
        score_frozen_cohort(p, manifest, ids, cases, role="tune")


def test_cv_scoring_requires_one_validation_fold():
    manifest = split_manifest(9)
    ids, cases = cohort_cases(manifest, "development_cv", 1)
    p = protocol(evaluated_fields=["height"])
    with pytest.raises(ValueError, match="validation_fold_required"):
        score_frozen_cohort(p, manifest, ids, cases, role="development_cv")
    with pytest.raises(ValueError, match="record_outside_validation_fold"):
        score_frozen_cohort(p, manifest, ids, cases, role="development_cv", validation_fold=0)
    report = score_frozen_cohort(p, manifest, ids, cases, role="development_cv", validation_fold=1)
    assert report.cohort.validation_fold == 1


def test_numeric_channel_cannot_cover_omitted_detection_record():
    manifest = split_manifest()
    ids, cases = cohort_cases(manifest)
    numbers = tuple(numeric(c.record_id.int, group_id=c.group_id) for c in cases)
    with pytest.raises(ValueError, match="cohort_records_missing_or_extra"):
        score_frozen_cohort(protocol(evaluated_fields=["height"]), manifest,
                            ids, cases[:-1], numbers, role="tune")


def test_top_k_cannot_skip_unprovided_queue_prefix():
    with pytest.raises(ValueError, match="queue_ranks_must_be_contiguous"):
        score_evaluation(protocol(evaluated_fields=["height"]),
                         (detection(1, queue_rank=100),), queue_k=10)


def test_manifest_bound_subset_is_explicit_and_individual_roles_are_respected():
    manifest = split_manifest()
    for role in ("train", "calibration"):
        ids, cases = cohort_cases(manifest, role)
        report = score_frozen_cohort(protocol(evaluated_fields=["height"]), manifest,
                                     ids[:1], cases[:1], role=role)
        assert report.cohort.record_count == 1
        with pytest.raises(ValueError, match="unexpected_validation_fold"):
            score_frozen_cohort(protocol(evaluated_fields=["height"]), manifest,
                                ids, cases, role=role, validation_fold=0)


def test_revision_cannot_duplicate_one_frozen_record():
    manifest = split_manifest()
    ids, cases = cohort_cases(manifest)
    repeated = cases[0].model_copy(update={"record_revision": 2})
    with pytest.raises(ValueError, match="inconsistent_record_revision"):
        score_frozen_cohort(protocol(evaluated_fields=["height"]), manifest,
                            ids, (*cases, repeated), role="tune")
