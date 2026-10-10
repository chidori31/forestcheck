"""Independent synthetic checks: proposals do not change source observations."""

from uuid import UUID, uuid4

import pytest
from forestcheck.domain.contracts import Evidence, Finding, Suggestion
from forestcheck.repair.engine import RecommendationEngine, verify_patch
from forestcheck.repair.models import (
    AggregateRelation,
    Candidate,
    DatasetBinding,
    PrecisionPolicy,
    ProposedPatch,
    SourceSupport,
    SuggestionV2,
)
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.models import RuleSource
from pydantic import ValidationError
from test_rules import context, field, record, rule


def support(name="model", *, roots=("train-v1",), inputs=("age", "species"), **kwargs):
    return SourceSupport(
        source_id=name,
        version="1.0",
        method="conditional_model",
        status="available",
        basis="Synthetic conditional observation; no expert truth",
        dependency_roots=roots,
        input_fields=inputs,
        **kwargs,
    )


def candidate(value=12.25, *, lower=10.1, upper=14.1, unit="m", evidence=None):
    return Candidate(
        support=evidence or support(),
        value=value,
        lower=lower,
        upper=upper,
        unit=unit,
        uncertainty_basis="Synthetic independent calibration example, not error probability",
    )


def binding(row):
    return DatasetBinding(
        schema_version="1.0",
        dataset_id=row.dataset_id,
        dataset_revision=row.dataset_revision,
        source_sha256="1" * 64,
        mapping_sha256="2" * 64,
        snapshot_sha256="3" * 64,
    )


def finding(row, name="height"):
    return Finding(
        schema_version="1.0",
        id=uuid4(),
        record_id=row.id,
        record_revision=row.revision,
        analysis_run_id=UUID(int=3),
        field_name=name,
        signal_kind="anomaly",
        severity="warning",
        priority=1,
        evidence=(Evidence(code="synthetic", message="Independent test signal"),),
        review_status="unreviewed",
    )


def precision(name="height", unit="m", step="0.5"):
    return PrecisionPolicy(
        field_name=name,
        unit=unit,
        step=step,
        source=RuleSource(reference_id="fixture", locator="table 1", kind="synthetic"),
    )


def recommend(row=None, *, candidates=None, name="height", **kwargs):
    row = row or record(field("height", 20.0, unit="m", original="020.0"))
    options = {
        "candidates": (candidate(),) if candidates is None else candidates,
        "precision": precision(name),
        "binding": binding(row),
        "context": context(),
        "rule_engine": RuleEngine(
            [rule({"kind": "field_contract", "field": name, "value_type": "number", "unit": "m"})]
        ),
    }
    options.update(kwargs)
    return RecommendationEngine({name: "m"}).recommend(row, finding(row, name), name, **options)


def test_proposal_rounds_half_up_bounds_outward_without_modifying_raw():
    row = record(field("height", 20.0, unit="m", original="020.0"))
    before = row.model_dump_json()
    result = recommend(row)
    assert result.suggestion.status == "proposed"
    assert result.suggestion.proposed_value == 12.5
    assert (result.suggestion.candidate_range.lower, result.suggestion.candidate_range.upper) == (
        10,
        14.5,
    )
    assert row.model_dump_json() == before
    patch = result.patch
    assert patch.status == "proposed_only"
    assert patch.operation.before.original_value == patch.operation.after.original_value == "020.0"
    assert patch.operation.before.source == patch.operation.after.source
    assert patch.operation.after.normalized_value == 12.5
    verify_patch(patch, row, binding(row))


def test_conflicting_sources_keep_alternatives_and_refuse_averaging():
    result = recommend(
        candidates=(
            candidate(),
            candidate(
                30, lower=28, upper=32, evidence=support("other", roots=("other-reference",))
            ),
        )
    )
    assert result.suggestion.status == "refused"
    assert result.suggestion.refusal_reason == "conflicting_sources"
    assert result.suggestion.proposed_value is None and result.patch is None
    assert len(result.suggestion.alternatives) == 2


def test_dependent_candidates_do_not_count_as_three_independent_confirmations():
    result = recommend(candidates=tuple(candidate(evidence=support(str(i))) for i in range(3)))
    assert result.suggestion.support_families == (("0", "1", "2"),)
    assert "independence_not_proven" in result.suggestion.limitations
    assert result.suggestion.candidate_range.lower == 10


@pytest.mark.parametrize("unit", ["cm", "unknown"])
def test_wrong_units_are_separate_hypothesis_and_never_rescaled(unit):
    result = recommend(candidates=(candidate(unit=unit),))
    assert result.suggestion.refusal_reason == "unit_mismatch_hypothesis"
    assert result.patch is None and result.suggestion.proposed_value is None
    hypothesis = result.suggestion.unit_hypotheses[0]
    assert hypothesis.observed_unit == "m" and hypothesis.candidate_unit == unit
    assert hypothesis.status == "unconfirmed" and hypothesis.source_id == "model"


def test_multiple_suspect_inputs_refuse_affected_estimate():
    result = recommend(suspect_fields=("age", "species"))
    assert result.suggestion.refusal_reason == "multiple_suspect_inputs"
    assert result.patch is None


def test_one_suspect_dependency_is_visible_and_requires_manual_review():
    result = recommend(suspect_fields=("age",))
    assert result.suggestion.status == "manual_review"
    assert result.patch is None
    assert "suspect_input:age" in result.suggestion.limitations


def test_no_reference_is_explicit_not_success():
    missing = SourceSupport(
        source_id="growth",
        version="1",
        method="growth",
        status="unavailable_reference",
        basis="No table matches supplied applicability",
        dependency_roots=(),
        input_fields=(),
    )
    result = recommend(candidates=(), sources=(missing,))
    assert result.suggestion.refusal_reason == "unavailable_reference"
    assert "Нет подходящего эталона" in result.suggestion.rationale
    assert result.patch is None


@pytest.mark.parametrize("state", ["not_applicable", "unknown"])
def test_no_reconstruction_of_inapplicable_or_unknown_field(state):
    result = recommend(record(field("height", None, unit="m", state=state)))
    assert result.suggestion.proposed_value is None and result.patch is None


@pytest.mark.parametrize("name", ["region", "species", "composition", "record_id"])
def test_numeric_allowlist_cannot_enable_protected_semantics(name):
    result = recommend(record(field(name, 1, unit="m")), name=name)
    assert result.suggestion.refusal_reason == "protected_field"


def test_unknown_precision_refuses_fabricated_digits_and_heuristic_precision():
    assert (
        recommend(precision=None).suggestion.refusal_reason == "measurement_precision_unavailable"
    )
    policy = precision().model_copy(
        update={
            "source": RuleSource(reference_id="guess", locator="configuration", kind="heuristic")
        }
    )
    assert recommend(precision=policy).patch is None


def test_unchanged_value_produces_no_patch():
    result = recommend(record(field("height", 12.5, unit="m")))
    assert result.suggestion.status == "no_change" and result.patch is None


def test_candidate_cannot_create_another_rule_violation():
    row = record(
        field("height", 20, unit="m"), field("area", 2, unit="ha"), field("total", 40, unit="m3")
    )
    # This deliberately uses another numeric contract to isolate revalidation from forestry norms.
    engine = RuleEngine([rule({"kind": "rounding", "field": "height", "unit": "m", "step": "10"})])
    result = recommend(row, rule_engine=engine)
    assert result.suggestion.refusal_reason == "candidate_creates_or_hides_violation"
    assert result.validation.new_violations and result.patch is None


def test_related_aggregate_invalidated_by_change_requires_coupled_manual_review():
    row = record(field("height", 20, unit="m"))
    sibling = record(field("height", 10, unit="m"))
    total = record(field("height", 30, unit="m"))
    relation = AggregateRelation(
        schema_version="1.0",
        id="fixture-sum",
        version="1",
        component_record_ids=(row.id, sibling.id),
        component_field="height",
        total_record_id=total.id,
        total_field="height",
        unit="m",
        quantity_kind="synthetic-length",
        component_level="element",
        total_level="stand",
        relationship_confirmed=True,
        absolute_tolerance="0",
        source=precision().source,
    )
    result = recommend(
        row,
        related_records=(sibling, total),
        aggregate_relations=(relation,),
        related_contexts={
            sibling.id: context(),
            total.id: context().model_copy(update={"record_level": "stand"}),
        },
    )
    assert result.suggestion.refusal_reason == "linked_aggregate_requires_coupled_review"
    assert result.patch is None
    assert result.validation.aggregate_after[0].status == "violation"
    assert total.fields[0].normalized_value == 30


def test_incomplete_related_population_cannot_claim_aggregate_safety():
    row = record(field("height", 20, unit="m"))
    relation = AggregateRelation(
        schema_version="1.0",
        id="fixture-sum",
        version="1",
        component_record_ids=(row.id, uuid4()),
        component_field="height",
        total_record_id=uuid4(),
        total_field="height",
        unit="m",
        quantity_kind="synthetic-length",
        component_level="element",
        total_level="stand",
        relationship_confirmed=True,
        absolute_tolerance="0",
        source=precision().source,
    )
    result = recommend(row, aggregate_relations=(relation,))
    assert result.patch is None and result.suggestion.status == "manual_review"
    assert result.validation.aggregate_after[0].status == "insufficient_data"


def test_patch_rejects_changed_record_snapshot_and_tampered_after_raw():
    row = record(field("height", 20, unit="m", original="020"))
    patch = recommend(row).patch
    with pytest.raises(ValueError, match="record"):
        verify_patch(patch, row.model_copy(update={"revision": 2}), binding(row))
    with pytest.raises(ValueError, match="snapshot"):
        verify_patch(patch, row, binding(row).model_copy(update={"snapshot_sha256": "4" * 64}))
    data = patch.model_dump(mode="json")
    data["operation"]["after"]["original_value"] = "forged"
    with pytest.raises(ValidationError):
        ProposedPatch.model_validate(data)


def test_version_two_explicit_no_silent_v1_conversion_or_future_version():
    value = recommend().suggestion
    assert isinstance(value, Suggestion) and value.schema_version == "2.0"
    with pytest.raises(ValidationError):
        Suggestion.model_validate(value.model_dump())
    data = value.model_dump()
    data["schema_version"] = "3.0"
    with pytest.raises(ValidationError):
        SuggestionV2.model_validate(data)
    data["schema_version"] = "2.0"
    data["confidence"] = 0.95
    with pytest.raises(ValidationError):
        SuggestionV2.model_validate(data)


def test_mismatched_finding_or_dataset_is_rejected_before_candidate():
    row = record(field("height", 20, unit="m"))
    with pytest.raises(ValueError, match="dataset"):
        recommend(row, binding=binding(row).model_copy(update={"dataset_revision": 2}))
    with pytest.raises(ValueError, match="finding"):
        RecommendationEngine({"height": "m"}).recommend(
            row,
            finding(record()),
            "height",
            candidates=(candidate(),),
            precision=precision(),
            binding=binding(row),
            context=context(),
            rule_engine=RuleEngine([]),
        )


def test_missing_value_proposed_only_with_known_units_and_unchanged_raw():
    row = record(field("height", None, unit="m", state="missing", original=""))
    result = recommend(row)
    assert (
        result.suggestion.status == "proposed" and result.patch.operation.after.state == "observed"
    )
    assert result.patch.operation.after.original_value == ""
    assert row.fields[0].state == "missing" and row.fields[0].normalized_value is None


def test_rounding_must_not_hide_disjoint_source_intervals():
    result = recommend(
        candidates=(
            candidate(12.1, lower=12.05, upper=12.15),
            candidate(
                12.3, lower=12.25, upper=12.35, evidence=support("other", roots=("independent",))
            ),
        )
    )
    assert result.suggestion.refusal_reason == "conflicting_sources"
    assert result.patch is None


def test_changed_evidence_creates_new_immutable_suggestion_identity():
    row = record(field("height", 20, unit="m"))
    signal = finding(row)
    engine = RecommendationEngine({"height": "m"})
    options = {
        "precision": precision(),
        "binding": binding(row),
        "context": context(),
        "rule_engine": RuleEngine([]),
    }
    first = engine.recommend(row, signal, "height", candidates=(candidate(),), **options)
    second = engine.recommend(row, signal, "height", candidates=(candidate(13),), **options)
    repeated = engine.recommend(row, signal, "height", candidates=(candidate(),), **options)
    assert first.suggestion.id != second.suggestion.id
    assert first.patch.id != second.patch.id
    assert first.suggestion.id == repeated.suggestion.id


def test_result_cannot_attach_a_different_validation_to_verified_patch():
    from forestcheck.repair.models import RecommendationResult

    result = recommend()
    payload = result.model_dump(mode="json")
    payload["validation"]["limitations"] = ["forged_validation"]
    with pytest.raises(ValidationError, match="validation"):
        RecommendationResult.model_validate(payload)


def test_aggregate_rejects_wrong_declared_component_level():
    row = record(field("height", 20, unit="m"))
    total = record(field("height", 12.5, unit="m"))
    relation = AggregateRelation(
        schema_version="1.0",
        id="wrong-level",
        version="1",
        component_record_ids=(row.id,),
        component_field="height",
        total_record_id=total.id,
        total_field="height",
        unit="m",
        quantity_kind="synthetic-length",
        component_level="deadwood",
        total_level="stand",
        relationship_confirmed=True,
        absolute_tolerance="0",
        source=precision().source,
    )
    result = recommend(
        row,
        related_records=(total,),
        related_contexts={total.id: context().model_copy(update={"record_level": "stand"})},
        aggregate_relations=(relation,),
    )
    assert result.patch is None
    assert result.validation.aggregate_after[0].status == "unavailable_reference"


def test_related_rows_use_their_own_context_and_missing_context_requires_review():
    row = record(field("height", 20, unit="m"))
    related = record(field("height", "not-a-number", unit="m"))
    result = recommend(
        row,
        related_records=(related,),
        related_contexts={related.id: context().model_copy(update={"record_level": "stand"})},
    )
    assert (
        next(o for o in result.validation.after if o.record_id == related.id).status
        == "not_applicable"
    )
    unknown = recommend(row, related_records=(related,))
    assert unknown.patch is None and unknown.suggestion.status == "manual_review"


def test_explicit_version_parser_preserves_v1_and_refuses_unknown():
    from forestcheck.repair.models import parse_suggestion

    current = recommend().suggestion
    assert isinstance(parse_suggestion(current.model_dump(mode="json")), SuggestionV2)
    old = Suggestion(
        schema_version="1.0",
        id=uuid4(),
        finding_id=uuid4(),
        record_id=uuid4(),
        field_name="height",
        proposed_value=12,
        unit="m",
        rationale="Legacy proposal",
    )
    assert type(parse_suggestion(old.model_dump(mode="json"))) is Suggestion
    with pytest.raises(ValueError, match="version"):
        parse_suggestion({"schema_version": "7.0"})


def test_existing_target_violation_must_not_survive_as_a_proposed_patch():
    row = record(field("height", 20.1, unit="m"))
    engine = RuleEngine([rule({"kind": "rounding", "field": "height", "unit": "m", "step": "10"})])
    result = recommend(row, rule_engine=engine)
    assert result.suggestion.status == "manual_review"
    assert result.suggestion.refusal_reason == "target_violation_unresolved"
    assert result.patch is None
    assert result.validation.unresolved_violations


def test_existing_unrelated_violation_remains_explicit_without_blocking_target():
    row = record(field("height", 20, unit="m"), field("other", 1.1, unit="m"))
    engine = RuleEngine([rule({"kind": "rounding", "field": "other", "unit": "m", "step": "10"})])
    result = recommend(row, rule_engine=engine)
    assert result.suggestion.status == "proposed" and result.patch is not None
    assert "existing_violations_remain" in result.suggestion.limitations


@pytest.mark.parametrize("value", [None, 13, "12.5", True])
def test_no_change_contract_requires_exact_observed_numeric_value(value):
    result = recommend(record(field("height", 12.5, unit="m")))
    payload = result.suggestion.model_dump(mode="json")
    payload["proposed_value"] = value
    with pytest.raises(ValidationError, match="no_change"):
        SuggestionV2.model_validate(payload)


def test_no_change_contract_rejects_missing_before_even_when_numeric():
    result = recommend(record(field("height", 12.5, unit="m")))
    payload = result.suggestion.model_dump(mode="json")
    payload["before"]["state"] = "missing"
    payload["before"]["normalized_value"] = None
    with pytest.raises(ValidationError, match="no_change"):
        SuggestionV2.model_validate(payload)
