"""Synthetic contracts only; no competition schema or forestry norms."""

from uuid import UUID, uuid4

import pytest
from forestcheck.domain.contracts import BusinessKey, FieldValue, Record, SourceCoordinates
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.models import Codebook, EvaluationContext, RuleDefinition
from pydantic import ValidationError

SOURCE = {"reference_id": "synthetic-contract", "locator": "fixture section 1", "kind": "synthetic"}


def field(name, value, *, unit=None, state="observed", original=None):
    return FieldValue(
        schema_version="1.0",
        name=name,
        original_value=value if original is None else original,
        normalized_value=value,
        unit=unit,
        state=state,
        source=SourceCoordinates(artifact_id=UUID(int=1), table="synthetic", row=2, column=name),
    )


def record(*fields, layer="1"):
    return Record(
        schema_version="1.0",
        id=uuid4(),
        dataset_id=UUID(int=2),
        business_key=BusinessKey(territory="R", compartment="Q", stand="007", layer=layer),
        fields=fields,
    )


def rule(check, *, categories=(), levels=("element",), id="synthetic.rule"):
    dependencies = [v for k, v in check.items() if k in ("field", "per_ha", "area", "total")]
    if check["kind"] == "unique_key":
        dependencies = check["fields"]
    return RuleDefinition.model_validate(
        {
            "schema_version": "1.0",
            "id": id,
            "version": "1.0",
            "record_levels": levels,
            "categories": categories,
            "dependencies": dependencies,
            "source": SOURCE,
            "severity": "warning",
            "priority": 2,
            "explanation": "Synthetic declared contract",
            "check": check,
        }
    )


def context(**kwargs):
    return EvaluationContext(
        analysis_run_id=UUID(int=3), record_level="element", allow_synthetic=True, **kwargs
    )


def test_nonapplicable_missing_reference_and_zero_are_not_passed_or_errors():
    definition = rule(
        {"kind": "field_contract", "field": "height", "value_type": "number", "unit": "m"}
    )
    engine = RuleEngine([definition])
    na = engine.evaluate(
        record(field("height", None, unit="m", state="not_applicable")), context()
    )[0]
    assert na.status == "not_applicable" and na.finding is None
    missing = engine.evaluate(record(field("height", None, unit="m", state="missing")), context())[
        0
    ]
    assert missing.status == "insufficient_data" and missing.finding is None
    assert engine.evaluate(record(field("height", 0, unit="m")), context())[0].status == "passed"
    other = EvaluationContext(analysis_run_id=UUID(int=3), record_level="deadwood")
    assert (
        engine.evaluate(record(field("height", 0, unit="m")), other)[0].status == "not_applicable"
    )


def test_required_only_for_declared_category():
    engine = RuleEngine([rule({"kind": "required", "field": "height"}, categories=("living",))])
    row = record(field("height", None, state="missing"))
    assert engine.evaluate(row, context(category="dead"))[0].status == "not_applicable"
    assert engine.evaluate(row, context())[0].status == "insufficient_data"
    outcome = engine.evaluate(row, context(category="living"))[0]
    assert outcome.status == "violation" and outcome.finding.signal_kind == "missing"
    assert outcome.signal_class == "confirmed_contradiction"


def test_unknown_code_requires_real_dictionary():
    definition = rule(
        {"kind": "code", "field": "species", "codebook_id": "species-v1", "codebook_version": "1"}
    )
    row = record(field("species", "ZZ"))
    assert RuleEngine([definition]).evaluate(row, context())[0].status == "unavailable_reference"
    book = Codebook(
        schema_version="1.0", id="species-v1", version="1", source=SOURCE, codes=("AA", "BB")
    )
    outcome = RuleEngine([definition], codebooks=[book]).evaluate(row, context())[0]
    assert outcome.status == "violation" and outcome.finding.signal_kind == "reference_mismatch"
    assert row.fields[0].original_value == "ZZ"


def test_units_types_and_parse_errors_separate_from_domain_contradictions():
    engine = RuleEngine(
        [rule({"kind": "field_contract", "field": "height", "value_type": "number", "unit": "m"})]
    )
    for f in (
        field("height", 10, unit="cm"),
        field("height", "10", unit="m"),
        field("height", True, unit="m"),
        field("height", None, unit="m", state="parse_error", original="bad"),
    ):
        out = engine.evaluate(record(f), context())[0]
        assert out.status == "violation" and out.signal_class == "format_error"
        assert out.finding.anomaly_score is None
        assert out.finding.evidence[0].reference_id == "synthetic.rule@1.0"


def test_duplicate_requires_confirmed_complete_key_and_level_two_layers_survive():
    definition = rule(
        {"kind": "unique_key", "fields": ["stand", "layer", "species"], "grain_confirmed": True}
    )
    a = record(field("stand", "007"), field("layer", "1"), field("species", "AA"))
    b = record(field("stand", "007"), field("layer", "2"), field("species", "AA"), layer="2")
    assert all(
        o.status == "passed" for o in RuleEngine([definition]).evaluate_many([a, b], context())
    )
    c = record(field("stand", "007"), field("layer", "1"), field("species", "AA"))
    assert all(
        o.status == "violation" for o in RuleEngine([definition]).evaluate_many([a, c], context())
    )
    unconfirmed = rule({"kind": "unique_key", "fields": ["stand"], "grain_confirmed": False})
    assert all(
        o.status == "unavailable_reference"
        for o in RuleEngine([unconfirmed]).evaluate_many([a, b], context())
    )
    assert RuleEngine([definition]).evaluate(a, context())[0].status == "insufficient_data"


def stock_rule(**changes):
    check = {
        "kind": "stock_balance",
        "per_ha": "v",
        "area": "a",
        "total": "t",
        "quantity_levels": ["stand", "stand", "stand"],
        "stock_kinds": ["living", "living"],
        "relationship_confirmed": True,
        "tolerance_kind": "rounding",
        "rounding_steps": ["1", "0.1", "1"],
    }
    check.update(changes)
    return rule(check, levels=("stand",))


def stock_context():
    return EvaluationContext(
        analysis_run_id=UUID(int=3), record_level="stand", allow_synthetic=True
    )


def test_arithmetic_documented_rounding_uses_intervals():
    engine = RuleEngine([stock_rule()])
    # v in [9.5,10.5], a in [1.95,2.05]; total [20.5,21.5] overlaps product.
    row = record(field("v", 10, unit="m3/ha"), field("a", 2, unit="ha"), field("t", 21, unit="m3"))
    assert engine.evaluate(row, stock_context())[0].status == "passed"
    bad = record(field("v", 10, unit="m3/ha"), field("a", 2, unit="ha"), field("t", 23, unit="m3"))
    out = engine.evaluate(bad, stock_context())[0]
    assert out.status == "violation" and out.signal_class == "confirmed_contradiction"
    assert out.finding.signal_kind == "inconsistency"


def test_stock_incompatible_scope_units_or_missing_tolerance_abstains():
    row = record(field("v", 10, unit="m3/ha"), field("a", 2, unit="ha"), field("t", 999, unit="m3"))
    for r in (
        stock_rule(quantity_levels=["layer", "stand", "stand"]),
        stock_rule(stock_kinds=["living", "dead"]),
        stock_rule(relationship_confirmed=False),
        stock_rule(tolerance_kind="unavailable", rounding_steps=None),
    ):
        out = RuleEngine([r]).evaluate(row, stock_context())[0]
        assert out.status == "unavailable_reference" and out.finding is None
    wrong = record(
        field("v", 10, unit="m3/ha"), field("a", 2, unit="m2"), field("t", 999, unit="m3")
    )
    assert (
        RuleEngine([stock_rule()]).evaluate(wrong, stock_context())[0].status == "insufficient_data"
    )


def test_heuristic_tolerance_never_reported_as_confirmed_contradiction():
    r = stock_rule(tolerance_kind="heuristic", rounding_steps=None, absolute_tolerance="0.2")
    row = record(field("v", 10, unit="m3/ha"), field("a", 2, unit="ha"), field("t", 21, unit="m3"))
    out = RuleEngine([r]).evaluate(row, stock_context())[0]
    assert out.signal_class == "statistical_suspicion" and out.finding.signal_kind == "anomaly"
    assert out.finding.anomaly_score is None


def test_rounding_step_only_if_documented():
    engine = RuleEngine([rule({"kind": "rounding", "field": "h", "unit": "m", "step": "1"})])
    assert engine.evaluate(record(field("h", 12.0, unit="m")), context())[0].status == "passed"
    assert (
        engine.evaluate(record(field("h", 12.5, unit="m")), context())[0].signal_class
        == "format_error"
    )


def test_unknown_versions_fields_and_duplicates_rejected():
    payload = rule({"kind": "required", "field": "h"}, categories=("living",)).model_dump(
        mode="json"
    )
    for changes in ({"schema_version": "2.0"}, {"undocumented": True}):
        with pytest.raises(ValidationError):
            RuleDefinition.model_validate(dict(payload, **changes))
    with pytest.raises(ValueError):
        RuleEngine([RuleDefinition.model_validate(payload)] * 2)


def test_finding_ids_and_originals_are_stable():
    row = record(field("h", 1, unit="cm"))
    before = row.model_dump_json()
    engine = RuleEngine(
        [rule({"kind": "field_contract", "field": "h", "value_type": "number", "unit": "m"})]
    )
    a = engine.evaluate(row, context())[0]
    b = engine.evaluate(row, context())[0]
    assert a == b and row.model_dump_json() == before
    assert a.finding.review_status == "unreviewed"


@pytest.mark.parametrize("expected_unit", [None, "m"])
def test_unknown_numeric_unit_abstains(expected_unit):
    engine = RuleEngine(
        [
            rule(
                {
                    "kind": "field_contract",
                    "field": "h",
                    "value_type": "number",
                    "unit": expected_unit,
                }
            )
        ]
    )
    out = engine.evaluate(record(field("h", 12, unit=None)), context())[0]
    assert out.status == "insufficient_data" and out.finding is None


def test_heuristic_dictionary_is_not_confirmed_and_evidence_is_versioned():
    definition = rule(
        {"kind": "code", "field": "code", "codebook_id": "book", "codebook_version": "1"}
    )
    book = Codebook(
        schema_version="1.0",
        id="book",
        version="1",
        source=dict(SOURCE, kind="heuristic"),
        codes=("AA",),
    )
    out = RuleEngine([definition], codebooks=[book]).evaluate(
        record(field("code", "ZZ")), context()
    )[0]
    assert out.signal_class == "statistical_suspicion" and out.finding.signal_kind == "anomaly"
    assert any(
        e.reference_id == "book@1" and "fixture section 1" in e.message
        for e in out.finding.evidence
    )
    wrong = Codebook(schema_version="1.0", id="book", version="2", source=SOURCE, codes=("AA",))
    mismatch = RuleEngine([definition], codebooks=[wrong]).evaluate(
        record(field("code", "ZZ")), context()
    )[0]
    assert mismatch.status == "unavailable_reference"
