"""Only independently constructed synthetic observations; no forest norms."""

import copy

import pytest
from forestcheck.ml.pipeline import (
    ConditionalPipeline,
    TargetSpec,
    eligible_training_indices,
    fit_pipeline,
    robust_deviation,
)
from pydantic import ValidationError


def spec(**changes):
    data = dict(
        schema_version="1.0",
        target="response",
        unit="m",
        numeric_features=("age_input",),
        categorical_features=("species",),
        context_fields=("region",),
        required_categories=("species",),
        min_train_rows=12,
        min_train_groups=3,
        min_conditional_rows=8,
    )
    return TargetSpec(**(data | changes))


def observations():
    rows = [
        {"age_input": i % 16, "species": "A" if i % 3 else "B", "region": "R"} for i in range(96)
    ]
    values = [float(2 * row["age_input"] + (row["species"] == "A")) for row in rows]
    return rows, values, [f"g{i // 8}" for i in range(96)]


@pytest.fixture(scope="module")
def trained():
    rows, values, groups = observations()
    return fit_pipeline(rows, values, groups, spec(), iterations=12)


def test_pipeline_predicts_without_target_and_does_not_mutate(trained):
    rows, _, _ = observations()
    before = copy.deepcopy(rows)
    candidate = trained.predict(rows[:4])
    baseline = trained.baseline.predict(rows[:4])
    assert rows == before
    assert all(p.status == "ok" and p.lower <= p.value <= p.upper for p in candidate)
    assert all(p.interval_status == "uncalibrated" for p in candidate)
    assert all(p.robust_scale is not None for p in baseline)


def test_native_prediction_uses_explicit_cpu_budget(trained, monkeypatch):
    original = trained._model.predict
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs.get("thread_count"))
        return original(*args, **kwargs)

    monkeypatch.setattr(trained._model, "predict", capture)
    trained.predict(observations()[0][:4])
    assert calls == [2]


def test_unknown_category_and_region_abstain(trained):
    row = observations()[0][0]
    for changes in ({"species": "NEW"}, {"region": "NEW"}, {"region": None}):
        p = trained.predict([row | changes])[0]
        assert p.status == "abstained" and p.value is None and p.lower is None


def test_numeric_range_and_missing_all_numeric_abstain(trained):
    row = observations()[0][0]
    for age in (1000, None):
        assert trained.predict([row | {"age_input": age}])[0].status == "abstained"
    assert trained.predict([row | {"age_input": 0}])[0].status == "ok"


def test_missing_optional_numeric_is_reproducible():
    rows, values, groups = observations()
    rows = [r | {"second": None if i % 2 else float(i % 7)} for i, r in enumerate(rows)]
    model = fit_pipeline(
        rows, values, groups, spec(numeric_features=("age_input", "second")), iterations=8
    )
    a, b = model.predict(rows[:6]), model.predict(rows[:6])
    assert a == b and all(p.status == "ok" for p in a)


def test_invalid_targets_excluded_and_no_target_is_failure():
    rows, values, groups = observations()
    values[:4] = [None, float("nan"), float("inf"), None]
    model = fit_pipeline(rows, values, groups, spec(), iterations=8)
    assert model.train_count == 92
    with pytest.raises(ValueError, match="insufficient"):
        fit_pipeline(rows, [None] * len(rows), groups, spec(), iterations=8)


def test_insufficient_groups_and_rows_fail():
    rows, values, groups = observations()
    for selected, selected_values, selected_groups in (
        (rows[:6], values[:6], groups[:6]),
        (rows, values, ["one"] * len(rows)),
    ):
        with pytest.raises(ValueError, match="insufficient"):
            fit_pipeline(selected, selected_values, selected_groups, spec(), iterations=8)


def test_rare_condition_uses_explicit_global_fallback():
    rows, values, groups = observations()
    rows[0]["species"] = "rare"
    model = fit_pipeline(rows, values, groups, spec(), iterations=8)
    prediction = model.baseline.predict([rows[0]])[0]
    assert prediction.status == "ok"
    assert prediction.baseline_level == "global"
    assert "small_conditional_group_global_fallback" in prediction.warnings


def test_baseline_uses_training_numeric_quantile_bins():
    rows, values, groups = observations()
    rows = [r | {"species": "A"} for r in rows]
    model = fit_pipeline(rows, values, groups, spec(), train_candidate=False)
    low, high = model.baseline.predict([rows[0] | {"age_input": 0}, rows[0] | {"age_input": 15}])
    assert low.baseline_level == high.baseline_level == "conditional"
    assert high.value - low.value > 10
    assert len(model.export_state()["numeric_condition_cuts"]) <= 4


def test_sparse_numeric_bins_fall_back_to_category():
    rows, values, groups = observations()
    rows = [r | {"species": "A"} for r in rows]
    model = fit_pipeline(rows, values, groups, spec(min_conditional_rows=30), train_candidate=False)
    prediction = model.baseline.predict(rows[:1])[0]
    assert prediction.baseline_level == "category"
    assert "small_numeric_bin_category_fallback" in prediction.warnings


def test_numeric_bins_are_unchanged_by_inference():
    rows, values, groups = observations()
    model = fit_pipeline(rows, values, groups, spec(), train_candidate=False)
    before = model.export_state()
    model.baseline.predict([rows[0] | {"age_input": 10000}])
    assert model.export_state() == before


def test_fit_eligibility_allows_optional_missing_numeric_and_filters_required_inputs():
    specification = spec(numeric_features=("age_input", "second"))
    base = {"age_input": 3, "second": None, "species": "A", "region": "R"}
    rows = [
        base,
        base | {"age_input": None, "second": 2},
        base | {"age_input": None},
        base | {"species": None},
        base | {"region": None},
        base,
        base,
        base,
    ]
    targets = [1, 2, 3, 4, 5, None, float("nan"), True]
    assert eligible_training_indices(rows, targets, specification) == (0, 1)


def test_fit_and_receipt_share_exact_eligible_indices():
    rows, values, groups = observations()
    rows = [r | {"second": None if i % 2 else i % 7} for i, r in enumerate(rows)]
    rows[0]["region"] = None
    rows[1]["species"] = None
    values[2] = None
    specification = spec(numeric_features=("age_input", "second"))
    indices = eligible_training_indices(rows, values, specification)
    model = fit_pipeline(rows, values, groups, specification, train_candidate=False)
    assert model.train_count == len(indices) == 93
    assert set(model.export_state()["train_groups"]) == {groups[i] for i in indices}


def test_fit_eligibility_rejects_malformed_excluded_rows():
    row = {"age_input": 1, "species": "A", "region": "R", "response": 5}
    with pytest.raises(ValueError, match="feature_keys"):
        eligible_training_indices([row], [None], spec())
    with pytest.raises(ValueError, match="fit_lengths"):
        eligible_training_indices(observations()[0], [1], spec())


def test_robust_deviation_is_a_score_not_a_probability():
    rows, values, groups = observations()
    model = fit_pipeline(rows, values, groups, spec(), train_candidate=False)
    prediction = model.baseline.predict(rows[:1])[0]
    value = prediction.value + 10 * prediction.robust_scale
    assert robust_deviation(value, prediction) == pytest.approx(10)
    assert robust_deviation(None, prediction) is None
    unavailable = model.predict(rows[:1])[0]
    assert robust_deviation(1, unavailable) is None


@pytest.mark.parametrize("role", ["tune", "calibration", "holdout", "test"])
def test_fit_rejects_roles_outside_train(role):
    rows, values, groups = observations()
    with pytest.raises(ValueError, match="fit_role"):
        fit_pipeline(rows, values, groups, spec(), role=role, iterations=8)


def test_target_and_extra_fields_cannot_be_predictors(trained):
    with pytest.raises(ValidationError):
        spec(numeric_features=("response",))
    row = observations()[0][0]
    for extra in ("response", "record_id", "labels"):
        with pytest.raises(ValueError, match="feature_keys"):
            trained.predict([row | {extra: 3}])


def test_typed_category_encoding_has_no_missing_token_collision():
    rows, values, groups = observations()
    rows = [r | {"optional": None if i % 2 else "__MISSING__"} for i, r in enumerate(rows)]
    model = fit_pipeline(
        rows, values, groups, spec(categorical_features=("species", "optional")), iterations=8
    )
    categories = model.export_state()["category_support"]["optional"]
    assert len(categories) == 2


def test_state_rejects_incompatible_schema_and_extra_fields(trained, tmp_path):
    path = tmp_path / "model.cbm"
    trained.save_catboost(path)
    state = trained.export_state()
    for changed in (state | {"schema_version": "9.0"}, state | {"extra": 1}):
        with pytest.raises(ValidationError):
            ConditionalPipeline.from_state(changed, path)


def test_native_roundtrip_predictions_reproduce(trained, tmp_path):
    path = tmp_path / "model.cbm"
    trained.save_catboost(path)
    restored = ConditionalPipeline.from_state(trained.export_state(), path)
    rows = observations()[0][:10]
    for old, new in zip(trained.predict(rows), restored.predict(rows), strict=True):
        assert new.value == pytest.approx(old.value, abs=1e-10)
        assert new.lower == pytest.approx(old.lower, abs=1e-10)


def test_group_calibration_is_separate_and_needs_enough_groups(trained):
    rows, values, _ = observations()
    with pytest.raises(ValueError, match="calibration_role"):
        trained.calibrate(rows, values, [f"c{i}" for i in range(96)], role="tune")
    metadata = trained.calibrate(rows, values, ["small"] * 96)
    assert metadata["status"] == "uncalibrated_insufficient_groups"
    groups = [f"cal{i // 4}" for i in range(96)]
    raw = trained.predict(rows[:4], calibrated=False)
    metadata = trained.calibrate(rows, values, groups)
    assert metadata["status"] == "calibrated_group_max" and metadata["groups"] == 24
    adjusted = trained.predict(rows[:4])
    for a, b in zip(raw, adjusted, strict=True):
        assert b.lower <= a.lower and b.upper >= a.upper
        assert b.value == a.value


def test_calibration_cannot_reuse_training_groups(trained):
    rows, values, groups = observations()
    with pytest.raises(ValueError, match="calibration_train_overlap"):
        trained.calibrate(rows, values, groups)


def test_baseline_remains_available_without_candidate():
    rows, values, groups = observations()
    model = fit_pipeline(rows, values, groups, spec(), train_candidate=False)
    assert model.baseline.predict(rows[:1])[0].status == "ok"
    unavailable = model.predict(rows[:1])[0]
    assert unavailable.status == "abstained"
    assert unavailable.warnings == ("candidate_unavailable",)


@pytest.mark.parametrize(
    "field,value", [("age_input", True), ("age_input", "12"), ("age_input", float("nan"))]
)
def test_invalid_numeric_predictor_never_silently_coerced(field, value):
    rows, values, groups = observations()
    rows[0][field] = value
    with pytest.raises(ValueError, match="invalid_numeric_predictor"):
        fit_pipeline(rows, values, groups, spec(), train_candidate=False)


def test_context_does_not_enter_estimator_features():
    assert spec().feature_names == ("age_input", "species")


def test_no_numeric_variation_refuses_candidate():
    rows, values, groups = observations()
    rows = [row | {"age_input": 1} for row in rows]
    with pytest.raises(ValueError, match="insufficient_numeric_feature_variation"):
        fit_pipeline(rows, values, groups, spec(), train_candidate=False)


def test_calibration_success_cannot_be_refitted(trained, tmp_path):
    path = tmp_path / "model.cbm"
    trained.save_catboost(path)
    fresh_state = trained.export_state()
    fresh_state["calibrations"]["catboost"] = {
        "schema_version": "1.0",
        "status": "uncalibrated",
        "groups": 0,
        "rows": 0,
        "nominal_coverage": 0.8,
        "adjustment": None,
        "reference": "unverified_observations",
    }
    model = ConditionalPipeline.from_state(fresh_state, path)
    rows, values, _ = observations()
    groups = [f"c{i // 4}" for i in range(96)]
    model.calibrate(rows, values, groups)
    with pytest.raises(ValueError, match="calibration_already_fitted"):
        model.calibrate(rows, values, groups)
