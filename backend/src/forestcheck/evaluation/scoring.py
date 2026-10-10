"""Evaluation-only sidecars and scorers. No fitting, data loading or truth inference."""
from collections import defaultdict
from math import floor
from random import Random
from statistics import mean, median
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from forestcheck.domain.contracts import Contract, Digest, PositiveInt, Text, ValueObject
from forestcheck.evaluation.split import SplitManifest, hash_value

EvaluationMode = Literal["real_expert", "masked_observation", "synthetic_stress"]
Number = StrictFloat | StrictInt
Nonnegative = Annotated[float, Field(strict=True, ge=0)]
Count = Annotated[int, Field(strict=True, ge=0)]


class ScoringProtocol(Contract):
    mode: EvaluationMode
    label_basis: Literal["expert_assessed", "hidden_observation", "synthetic_truth"]
    evaluated_fields: Annotated[tuple[Text, ...], Field(min_length=1)]
    evaluated_method_id: Text
    complete_labels: StrictBool = False
    label_producer_ids: tuple[Text, ...] = ()
    independent_judge_ids: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def consistent_basis(self) -> Self:
        expected = {"real_expert": "expert_assessed", "masked_observation": "hidden_observation",
                    "synthetic_stress": "synthetic_truth"}
        if self.label_basis != expected[self.mode]:
            raise ValueError("label_basis_incompatible_with_mode")
        if len(set(self.evaluated_fields)) != len(self.evaluated_fields):
            raise ValueError("duplicate_evaluated_field")
        if self.mode != "masked_observation" and not self.label_producer_ids:
            raise ValueError("label_provenance_required")
        if self.evaluated_method_id in self.label_producer_ids and not (
            set(self.independent_judge_ids) - set(self.label_producer_ids)
        ):
            raise ValueError("generator_cannot_be_sole_judge_of_itself")
        return self


class DetectionCase(Contract):
    record_id: UUID
    record_revision: PositiveInt = 1
    group_id: Text
    field_name: Text
    is_error: StrictBool | None
    predicted_error: StrictBool | None
    queue_rank: PositiveInt | None = None


class NumericCase(Contract):
    record_id: UUID
    record_revision: PositiveInt = 1
    group_id: Text
    field_name: Text
    unit: Text
    reference: Number | None
    estimate: Number | None
    interval_lower: Number | None = None
    interval_upper: Number | None = None

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("interval_requires_both_bounds")
        if self.interval_lower is not None and self.interval_lower > self.interval_upper:
            raise ValueError("interval_bounds_reversed")
        return self


class ResourceMeasurement(Contract):
    """Caller-measured resources for the evaluated run, never guessed by the scorer."""

    wall_seconds: Nonnegative | None = None
    peak_memory_bytes: Count | None = None
    measurement_method: Text | None = None

    @model_validator(mode="after")
    def measurement_has_method(self) -> Self:
        if (self.wall_seconds is not None or self.peak_memory_bytes is not None
                ) and self.measurement_method is None:
            raise ValueError("resource_measurement_method_required")
        return self


class UncertaintyPlan(Contract):
    independent_groups: StrictBool = False
    independence_basis: Text | None = None
    minimum_groups: Annotated[int, Field(strict=True, ge=5)] = 5
    resamples: Annotated[int, Field(strict=True, ge=100, le=5000)] = 1000
    seed: Count = 0

    @model_validator(mode="after")
    def independence_has_basis(self) -> Self:
        if self.independent_groups and self.independence_basis is None:
            raise ValueError("independence_basis_required")
        return self


class Metric(ValueObject):
    status: Literal["COMPUTED", "NOT_RUN"]
    value: Number | None = None
    denominator: Count = 0
    reason: Text
    unit: Text | None = None


class BinaryMetrics(ValueObject):
    sample_count: Count
    judged: Count
    unlabelled: Count
    abstained: Count
    true_positive: Count
    false_positive: Count
    false_negative: Count
    true_negative: Count
    precision: Metric
    precision_on_judged: Metric
    recall: Metric
    f1: Metric
    abstention_rate: Metric


class FieldMetrics(ValueObject):
    field_name: Text
    metrics: BinaryMetrics


class NumericMetrics(ValueObject):
    field_name: Text
    unit: Text
    sample_count: Count
    reference_count: Count
    paired_count: Count
    mae: Metric
    medae: Metric
    interval_coverage: Metric
    mean_interval_width: Metric
    abstention_rate: Metric


class QueueMetrics(ValueObject):
    requested_k: PositiveInt
    selected: Count
    judged: Count
    precision_at_k: Metric


class GroupMetrics(ValueObject):
    group_id: Text
    fields: BinaryMetrics
    rows: BinaryMetrics
    numeric: tuple[NumericMetrics, ...]


class UncertaintyInterval(ValueObject):
    metric: Text
    field_name: Text | None = None
    unit: Text | None = None
    status: Literal["COMPUTED", "NOT_RUN"]
    lower: Number | None = None
    upper: Number | None = None
    confidence_level: Literal[0.95] = 0.95
    group_count: Count
    resamples: Count
    seed: Count | None = None
    reason: Text


class CohortBinding(Contract):
    manifest_digest: Digest
    cohort_digest: Digest
    role: Literal["train", "tune", "calibration", "development_cv"]
    validation_fold: Count | None
    record_count: PositiveInt
    group_count: PositiveInt
    prior_exposed_group_count: Count


class EvaluationReport(Contract):
    mode: EvaluationMode
    protocol: ScoringProtocol
    uncertainty_plan: UncertaintyPlan | None
    cohort: CohortBinding | None = None
    reference_interpretation: Literal[
        "expert_judgement_in_declared_scope", "reconstruction_of_unverified_observation",
        "synthetic_truth_only",
    ]
    sample_records: Count
    sample_groups: Count
    pooled_fields: BinaryMetrics
    fields: tuple[FieldMetrics, ...]
    rows: BinaryMetrics
    queue: QueueMetrics
    numeric: tuple[NumericMetrics, ...]
    groups: tuple[GroupMetrics, ...]
    uncertainty: tuple[UncertaintyInterval, ...]
    wall_seconds: Metric
    peak_memory_bytes: Metric
    resource_measurement_method: Text | None


def _not_run(reason: str, denominator: int = 0, unit: str | None = None) -> Metric:
    return Metric(status="NOT_RUN", denominator=denominator, reason=reason, unit=unit)


def _value(value: float | int, denominator: int, reason: str,
           unit: str | None = None) -> Metric:
    return Metric(status="COMPUTED", value=value, denominator=denominator,
                  reason=reason, unit=unit)


def _ratio(numerator: int, denominator: int, reason: str) -> Metric:
    return (_value(numerator / denominator, denominator, reason) if denominator
            else _not_run("zero_denominator"))


def _binary(cases: list[tuple[bool | None, bool | None]], complete: bool,
            detection_truth: bool) -> BinaryMetrics:
    """Abstention misses a labelled positive for recall, but asserts no negative."""
    if not detection_truth:
        cases = [(None, pred) for _, pred in cases]
    n = len(cases)
    judged = sum(truth is not None for truth, _ in cases)
    abstained = sum(pred is None for _, pred in cases)
    tp = sum(truth is True and pred is True for truth, pred in cases)
    fp = sum(truth is False and pred is True for truth, pred in cases)
    fn = sum(truth is True and pred is not True for truth, pred in cases)
    tn = sum(truth is False and pred is False for truth, pred in cases)
    unjudged_positive = sum(truth is None and pred is True for truth, pred in cases)
    precision_judged = _ratio(tp, tp + fp, "precision_conditional_on_judged_flagged_subset")
    precision = (precision_judged if not unjudged_positive else
                 _not_run("unjudged_predicted_positives", tp + fp + unjudged_positive))
    if precision.status == "COMPUTED":
        precision = _value(precision.value, precision.denominator,
                           "all_predicted_positives_judged_in_declared_scope")
    recall = (_ratio(tp, tp + fn, "complete_labels_abstained_positives_count_as_missed")
              if complete else _not_run("complete_label_coverage_required", judged))
    f1 = (_ratio(2 * tp, 2 * tp + fp + fn, "complete_label_coverage") if complete
          else _not_run("complete_label_coverage_required", judged))
    if not detection_truth:
        precision = precision_judged = recall = f1 = _not_run("no_detection_truth_in_mode")
    return BinaryMetrics(sample_count=n, judged=judged, unlabelled=n - judged,
                         abstained=abstained, true_positive=tp, false_positive=fp,
                         false_negative=fn, true_negative=tn, precision=precision,
                         precision_on_judged=precision_judged, recall=recall, f1=f1,
                         abstention_rate=_ratio(abstained, n, "prediction_abstention"))


def _numeric(cases: tuple[NumericCase, ...]) -> tuple[NumericMetrics, ...]:
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case.field_name, case.unit)].append(case)
    output = []
    for (field, unit), rows in sorted(grouped.items()):
        paired = [c for c in rows if c.reference is not None and c.estimate is not None]
        errors = [abs(c.reference - c.estimate) for c in paired]
        bounded = [c for c in rows if c.reference is not None and c.interval_lower is not None]
        widths = [c.interval_upper - c.interval_lower for c in bounded]
        covered = sum(c.interval_lower <= c.reference <= c.interval_upper for c in bounded)
        output.append(NumericMetrics(
            field_name=field, unit=unit, sample_count=len(rows),
            reference_count=sum(c.reference is not None for c in rows), paired_count=len(paired),
            mae=(_value(mean(errors), len(errors), "paired_references_only", unit) if errors
                 else _not_run("no_paired_references", unit=unit)),
            medae=(_value(median(errors), len(errors), "paired_references_only", unit) if errors
                   else _not_run("no_paired_references", unit=unit)),
            interval_coverage=_ratio(covered, len(bounded), "provided_intervals_with_references"),
            mean_interval_width=(
                _value(mean(widths), len(widths), "provided_intervals_with_references", unit)
                if widths else _not_run("no_intervals_with_references", unit=unit)),
            abstention_rate=_ratio(sum(c.estimate is None for c in rows), len(rows),
                                   "numeric_point_estimate_abstention"),
        ))
    return tuple(output)


def _identity(case: DetectionCase | NumericCase) -> tuple[UUID, int]:
    return case.record_id, case.record_revision


def _validate_cases(protocol: ScoringProtocol, cases: tuple[DetectionCase, ...],
                    numeric: tuple[NumericCase, ...]) -> dict[UUID, str]:
    groups = {}
    revisions = {}
    for collection, kind in ((cases, "detection"), (numeric, "numeric")):
        seen = set()
        for case in collection:
            if case.field_name not in protocol.evaluated_fields:
                raise ValueError("field_outside_protocol")
            key = (*_identity(case), case.field_name)
            if key in seen:
                raise ValueError(f"duplicate_{kind}_case")
            seen.add(key)
            if case.record_id in groups and groups[case.record_id] != case.group_id:
                raise ValueError("inconsistent_record_group")
            groups[case.record_id] = case.group_id
            if case.record_id in revisions and revisions[case.record_id] != case.record_revision:
                raise ValueError("inconsistent_record_revision")
            revisions[case.record_id] = case.record_revision
    ranks = [c.queue_rank for c in cases if c.queue_rank is not None]
    if len(ranks) != len(set(ranks)):
        raise ValueError("duplicate_queue_rank")
    if set(ranks) != set(range(1, len(ranks) + 1)):
        raise ValueError("queue_ranks_must_be_contiguous")
    return groups


def _row_truth(cases: list[DetectionCase], fields: set[str]) -> tuple[bool | None, bool | None]:
    full = {c.field_name for c in cases} == fields
    truth = (True if any(c.is_error is True for c in cases) else
             False if full and all(c.is_error is False for c in cases) else None)
    predicted = (True if any(c.predicted_error is True for c in cases) else
                 False if full and all(c.predicted_error is False for c in cases) else None)
    return truth, predicted


def _quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _bootstrap(group_contributions: dict[str, tuple[float, int]], metric: str,
               plan: UncertaintyPlan | None, field: str | None = None,
               unit: str | None = None, eligible: bool = True) -> UncertaintyInterval:
    count = len(group_contributions)
    base = dict(metric=metric, field_name=field, unit=unit, group_count=count,
                resamples=0, seed=plan.seed if plan else None)
    if not eligible:
        reason = "metric_not_admissible"
    elif plan is None or not plan.independent_groups:
        reason = "independent_group_justification_required"
    elif count < plan.minimum_groups:
        reason = "insufficient_independent_groups"
    else:
        reason = None
    if reason:
        return UncertaintyInterval(**base, status="NOT_RUN", reason=reason)
    rng = Random(plan.seed)
    contributions = [group_contributions[group] for group in sorted(group_contributions)]
    values = []
    for _ in range(plan.resamples):
        sample = rng.choices(contributions, k=count)
        denominator = sum(n for _, n in sample)
        if denominator:
            values.append(sum(s for s, _ in sample) / denominator)
    base["resamples"] = len(values)
    if len(values) < .9 * plan.resamples:
        return UncertaintyInterval(**base, status="NOT_RUN", reason="unstable_bootstrap_denominators")
    return UncertaintyInterval(**base, status="COMPUTED", lower=_quantile(values, .025),
                               upper=_quantile(values, .975),
                               reason="percentile_cluster_bootstrap_groups_with_replacement")


def score_evaluation(
    protocol: ScoringProtocol,
    cases: tuple[DetectionCase, ...] = (),
    numeric_cases: tuple[NumericCase, ...] = (),
    *, queue_k: int = 10,
    resources: ResourceMeasurement | None = None,
    uncertainty: UncertaintyPlan | None = None,
) -> EvaluationReport:
    """Score explicit sidecars only; population coverage is a caller-audited precondition.

    No metric extrapolates beyond the submitted cohort. Full label coverage checks
    every declared field of each supplied record; it cannot prove a representative
    cohort was selected. Use score_frozen_cohort for a manifest-bound stage-5 run.
    A masked observation is a reconstruction target, not truth.
    """
    if isinstance(queue_k, bool) or not isinstance(queue_k, int) or queue_k < 1:
        raise ValueError("invalid_queue_k")
    groups = _validate_cases(protocol, cases, numeric_cases)
    by_record = defaultdict(list)
    fields_by_group = defaultdict(list)
    numeric_by_group = defaultdict(list)
    for case in cases:
        by_record[_identity(case)].append(case)
        fields_by_group[case.group_id].append((case.is_error, case.predicted_error))
    for case in numeric_cases:
        numeric_by_group[case.group_id].append(case)
    field_set = set(protocol.evaluated_fields)
    coverage = all({c.field_name for c in rows} == field_set
                   and all(c.is_error is not None for c in rows) for rows in by_record.values())
    if protocol.complete_labels and not coverage:
        raise ValueError("complete_label_coverage_not_met")
    complete = protocol.complete_labels and coverage
    detection_truth = protocol.mode != "masked_observation"
    rows_by_group = defaultdict(list)
    for rows in by_record.values():
        rows_by_group[rows[0].group_id].append(_row_truth(rows, field_set))
    all_rows = [row for group in sorted(rows_by_group) for row in rows_by_group[group]]
    all_fields = [(c.is_error, c.predicted_error) for c in cases]
    field_metrics = tuple(FieldMetrics(
        field_name=field, metrics=_binary([(c.is_error, c.predicted_error) for c in cases
                                          if c.field_name == field], complete, detection_truth),
    ) for field in sorted(protocol.evaluated_fields))
    ranked = sorted((c for c in cases if c.queue_rank is not None), key=lambda c: c.queue_rank)
    selected = ranked[:queue_k]
    judged = sum(c.is_error is not None for c in selected) if detection_truth else 0
    precision_k = (_ratio(sum(c.is_error is True for c in selected), len(selected),
                          "all_selected_queue_items_judged")
                   if detection_truth and judged == len(selected) else
                   _not_run("top_k_not_fully_judged", len(selected)))
    group_reports = tuple(GroupMetrics(
        group_id=group,
        fields=_binary(fields_by_group[group], complete, detection_truth),
        rows=_binary(rows_by_group[group], complete, detection_truth),
        numeric=_numeric(tuple(numeric_by_group[group])),
    ) for group in sorted(set(groups.values())))
    uncertainty_results = [_bootstrap({
        group: (sum(t is True and p is True for t, p in rows), sum(t is True for t, _ in rows))
        for group, rows in rows_by_group.items()
    }, "row_recall", uncertainty, eligible=complete and detection_truth and bool(cases))]
    numeric_metrics = _numeric(numeric_cases)
    for numeric_metric in numeric_metrics:
        contributions = defaultdict(lambda: [0., 0])
        for case in numeric_cases:
            if (case.field_name, case.unit) == (numeric_metric.field_name, numeric_metric.unit
                                               ) and case.reference is not None and case.estimate is not None:
                contributions[case.group_id][0] += abs(case.reference - case.estimate)
                contributions[case.group_id][1] += 1
        uncertainty_results.append(_bootstrap(
            dict(contributions), "mae", uncertainty, numeric_metric.field_name, numeric_metric.unit,
            eligible=numeric_metric.mae.status == "COMPUTED"))
    resources = resources or ResourceMeasurement(schema_version="1.0")
    interpretations = {"real_expert": "expert_judgement_in_declared_scope",
                       "masked_observation": "reconstruction_of_unverified_observation",
                       "synthetic_stress": "synthetic_truth_only"}
    return EvaluationReport(
        schema_version="1.0", mode=protocol.mode,
        protocol=protocol, uncertainty_plan=uncertainty,
        reference_interpretation=interpretations[protocol.mode],
        sample_records=len({_identity(c) for c in (*cases, *numeric_cases)}),
        sample_groups=len(set(groups.values())),
        pooled_fields=_binary(all_fields, complete, detection_truth), fields=field_metrics,
        rows=_binary(all_rows, complete, detection_truth),
        queue=QueueMetrics(requested_k=queue_k, selected=len(selected), judged=judged,
                           precision_at_k=precision_k),
        numeric=numeric_metrics, groups=group_reports, uncertainty=tuple(uncertainty_results),
        wall_seconds=(_value(resources.wall_seconds, 1, "caller_measured", "s")
                      if resources.wall_seconds is not None else _not_run("not_measured", unit="s")),
        peak_memory_bytes=(
            _value(resources.peak_memory_bytes, 1, "caller_measured", "byte")
            if resources.peak_memory_bytes is not None else _not_run("not_measured", unit="byte")),
        resource_measurement_method=resources.measurement_method,
    )


def score_frozen_cohort(
    protocol: ScoringProtocol,
    manifest: SplitManifest,
    cohort_ids: tuple[UUID, ...],
    cases: tuple[DetectionCase, ...] = (),
    numeric_cases: tuple[NumericCase, ...] = (),
    *,
    role: Literal["train", "tune", "calibration", "development_cv"],
    validation_fold: int | None = None,
    queue_k: int = 10,
    resources: ResourceMeasurement | None = None,
    uncertainty: UncertaintyPlan | None = None,
) -> EvaluationReport:
    """Bind an explicitly selected cohort to one frozen role or CV validation fold.

    Each nonempty scoring channel must cover exactly the declared record IDs.
    For different detection and reconstruction cohorts, call this wrapper separately;
    one channel cannot conceal omitted records of the other. A selected subset is
    allowed but is not automatically representative. Holdout has no stage-5 release
    switch: final evaluation belongs to a separate future stage-10 workflow.
    """
    if role == "holdout":
        raise ValueError("holdout_reserved_for_stage10")
    if role not in {"train", "tune", "calibration", "development_cv"}:
        raise ValueError("invalid_scoring_role")
    if not cohort_ids:
        raise ValueError("empty_cohort")
    if any(not isinstance(rid, UUID) for rid in cohort_ids):
        raise ValueError("cohort_uuid_required")
    requested = set(cohort_ids)
    if len(requested) != len(cohort_ids):
        raise ValueError("duplicate_cohort_record")
    if role == "development_cv":
        if (type(validation_fold) is not int or manifest.folds is None
                or not 0 <= validation_fold < manifest.folds):
            raise ValueError("validation_fold_required")
    elif validation_fold is not None:
        raise ValueError("unexpected_validation_fold")
    assignments = {rid: manifest.assert_allowed(rid, roles={role}) for rid in requested}
    if role == "development_cv" and any(a.fold != validation_fold for a in assignments.values()):
        raise ValueError("record_outside_validation_fold")
    if not cases and not numeric_cases:
        raise ValueError("cohort_records_missing_or_extra")
    for channel in (cases, numeric_cases):
        if channel and {c.record_id for c in channel} != requested:
            raise ValueError("cohort_records_missing_or_extra")
        for case in channel:
            if case.group_id != assignments[case.record_id].group_id:
                raise ValueError("case_group_does_not_match_manifest")
    report = score_evaluation(protocol, cases, numeric_cases, queue_k=queue_k,
                              resources=resources, uncertainty=uncertainty)
    groups = {a.group_id for a in assignments.values()}
    binding = CohortBinding(
        schema_version="1.0", manifest_digest=manifest.digest,
        cohort_digest=hash_value(sorted(str(rid) for rid in requested)),
        role=role, validation_fold=validation_fold, record_count=len(requested),
        group_count=len(groups),
        prior_exposed_group_count=len(groups & set(manifest.prior_exposed_groups)),
    )
    return EvaluationReport.model_validate({**report.model_dump(), "cohort": binding})
