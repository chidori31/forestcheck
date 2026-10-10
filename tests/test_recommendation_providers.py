from uuid import UUID

import pytest
from forestcheck.growth import GrowthQuery, GrowthTableRegistry
from forestcheck.ml.pipeline import Prediction
from forestcheck.repair.providers import growth_source, model_source, stock_sources
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.models import EvaluationContext, RuleDefinition
from test_ml_data import plan, record


def test_real_empty_registry_never_supplies_prepared_answer():
    source, candidate = growth_source(GrowthTableRegistry(), GrowthQuery(schema_version="1.0"))
    assert candidate is None and source.status == "unavailable_reference"
    assert "no_registered_reference" in source.basis


def test_available_growth_requires_explicit_query_field_dependencies():
    from test_growth import query, registry

    source, candidate = growth_source(registry(), query(), allow_synthetic=True)
    assert source.status == "insufficient_data" and candidate is None
    bindings = {
        name: "field_" + name
        for name in ("region", "species", "origin", "stand_type", "bonitet", "conditions", "age")
    }
    source, candidate = growth_source(
        registry(), query(), field_bindings=bindings, allow_synthetic=True
    )
    assert candidate.value == 2 and source.status == "available"
    assert set(source.input_fields) == set(bindings.values())
    assert "synthetic-section" in source.basis


def test_model_support_uses_actual_prediction_and_common_training_root():
    p = plan()
    prediction = Prediction(
        schema_version="1.0",
        status="ok",
        value=12.34,
        lower=10.1,
        upper=16.4,
        unit="m",
        interval_status="calibrated_group_max",
    )
    source, candidate = model_source(
        prediction, model_id=UUID(int=123), plan=p, target=p.targets[0], method="conditional_model"
    )
    baseline, _ = model_source(
        prediction, model_id=UUID(int=123), plan=p, target=p.targets[0], method="baseline"
    )
    assert candidate.value == 12.34 and candidate.lower == 10.1
    assert source.dependency_roots == baseline.dependency_roots
    assert set(source.input_fields) == {"a", "s", "r"}
    assert "unverified" in candidate.uncertainty_basis
    refused, value = model_source(
        prediction.model_copy(update={"status": "abstained", "value": None}),
        model_id=UUID(int=123),
        plan=p,
        target=p.targets[0],
        method="conditional_model",
    )
    assert refused.status == "insufficient_data" and value is None


def test_model_unit_mismatch_is_not_rescaled():
    p = plan()
    pred = Prediction(
        schema_version="1.0", status="ok", value=1200.0, unit="cm", interval_status="uncalibrated"
    )
    source, candidate = model_source(
        pred, model_id=UUID(int=9), plan=p, target=p.targets[0], method="conditional_model"
    )
    assert source.status == "insufficient_data" and candidate is None
    assert "unit" in source.basis


def relation_record(area=2.0, unit="ha"):
    r = record()
    prototype = r.fields[0]
    fields = tuple(
        prototype.model_copy(
            update={
                "name": n,
                "original_value": v,
                "normalized_value": v,
                "unit": u,
                "source": prototype.source.model_copy(update={"column": n}),
            }
        )
        for n, v, u in (("per", 10.0, "m3/ha"), ("area", area, unit), ("total", 30.0, "m3"))
    )
    return r.model_copy(update={"fields": fields})


def stock_rule(**updates):
    check = dict(
        kind="stock_balance",
        per_ha="per",
        area="area",
        total="total",
        quantity_levels=("stand",) * 3,
        stock_kinds=("standing",) * 2,
        relationship_confirmed=True,
        tolerance_kind="explicit",
        absolute_tolerance="1",
    )
    check.update(updates)
    return RuleDefinition(
        schema_version="1.0",
        id="stock",
        version="1",
        record_levels=("stand",),
        dependencies=("per", "area", "total"),
        source=dict(reference_id="synthetic", locator="fixture", kind="synthetic"),
        severity="warning",
        priority=1,
        explanation="Synthetic stock relation",
        check=check,
    )


def test_stock_provider_only_confirmed_compatible_relation():
    context = EvaluationContext(
        analysis_run_id=UUID(int=99), record_level="stand", allow_synthetic=True
    )
    sources, candidates = stock_sources(
        relation_record(), "total", RuleEngine([stock_rule()]), context, data_root="synthetic"
    )
    assert len(candidates) == 1 and candidates[0].value == 20.0
    assert candidates[0].unit == "m3" and candidates[0].support.input_fields == ("per", "area")
    for row, rule in (
        (relation_record(unit="m2"), stock_rule()),
        (relation_record(), stock_rule(relationship_confirmed=False)),
    ):
        sources, candidates = stock_sources(
            row, "total", RuleEngine([rule]), context, data_root="synthetic"
        )
        assert not candidates and sources[0].status != "available"
    sources, candidates = stock_sources(
        relation_record(area=0.0), "per", RuleEngine([stock_rule()]), context, data_root="synthetic"
    )
    assert not candidates


@pytest.mark.parametrize("field", ["a", "s", "r"])
def test_no_invented_arithmetic_for_unrelated_field(field):
    context = EvaluationContext(
        analysis_run_id=UUID(int=99), record_level="stand", allow_synthetic=True
    )
    sources, candidates = stock_sources(
        record(), field, RuleEngine([stock_rule()]), context, data_root="synthetic"
    )
    assert not candidates and sources[0].status == "unavailable_reference"


def test_unknown_category_and_heuristic_relation_do_not_claim_documented_value():
    context = EvaluationContext(
        analysis_run_id=UUID(int=99), record_level="stand", allow_synthetic=True
    )
    rule = stock_rule().model_copy(update={"categories": ("forest",)})
    sources, candidates = stock_sources(
        relation_record(), "total", RuleEngine([rule]), context, data_root="synthetic"
    )
    assert not candidates and sources[0].status == "insufficient_data"
    for check in (
        stock_rule(tolerance_kind="heuristic"),
        stock_rule().model_copy(
            update={"source": stock_rule().source.model_copy(update={"kind": "heuristic"})}
        ),
    ):
        sources, candidates = stock_sources(
            relation_record(), "total", RuleEngine([check]), context, data_root="synthetic"
        )
        assert not candidates and sources[0].status == "unavailable_reference"
