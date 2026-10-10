import json
import math

import pytest
from forestcheck.ml.isolation import (
    IsolationModel,
    eligible_isolation_indices,
    fit_isolation,
)


def sample_rows(count=120):
    return [
        {
            "age": float(10 + i % 40),
            "height": float(5 + i % 11),
            "diameter": None if i % 7 == 0 else float(8 + i % 17),
        }
        for i in range(count)
    ]


def fit(rows=None, **kwargs):
    rows = sample_rows() if rows is None else rows
    return fit_isolation(
        rows,
        groups=[f"group-{i % 6}" for i in range(len(rows))],
        feature_names=("age", "height", "diameter"),
        n_estimators=12,
        **kwargs,
    )


def test_data_only_constant_tree_inference():
    model = IsolationModel.model_validate(
        {
            "schema_version": "1.0",
            "sklearn_version": "synthetic-tree-fixture",
            "feature_names": ["x", "y"],
            "seed": 1,
            "fit_role": "train",
            "train_rows": 40,
            "train_groups": 3,
            "max_samples": 40,
            "medians": [2.0, 3.0],
            "minima": [2.0, 3.0],
            "maxima": [2.0, 3.0],
            "trees": [
                {
                    "children_left": [-1],
                    "children_right": [-1],
                    "features": [-2],
                    "thresholds": [-2.0],
                    "samples": [40],
                }
            ],
            "training_scores": [0.5] * 40,
        }
    )
    prediction = model.predict([{"x": 2.0, "y": 3.0}])[0]
    assert prediction.score == 0.5 and prediction.rank_quantile == 1.0


def test_shared_eligibility_matches_fitted_rows_and_preserves_indices():
    rows = sample_rows(60)
    rows.insert(0, {"age": 10.0, "height": None, "diameter": None})
    rows.insert(12, {"age": None, "height": None, "diameter": None})
    names = ("age", "height", "diameter")
    selected = eligible_isolation_indices(rows, names)
    assert selected == tuple(i for i in range(62) if i not in {0, 12})
    model = fit(rows)
    assert model.train_rows == len(selected)
    assert all(sum(rows[i][name] is not None for name in names) >= 2 for i in selected)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_shared_eligibility_rejects_invalid_values_in_excluded_rows(value):
    with pytest.raises(ValueError, match="invalid_numeric"):
        eligible_isolation_indices(
            [{"age": value, "height": None, "diameter": None}],
            ("age", "height", "diameter"),
        )


def test_shared_eligibility_requires_exact_schema():
    with pytest.raises(ValueError, match="feature_schema_mismatch"):
        eligible_isolation_indices([{"age": 1.0}], ("age", "height"))
    with pytest.raises(ValueError, match="invalid_feature_names"):
        eligible_isolation_indices([], ("age", "age"))


def test_export_matches_sklearn_score_samples():
    import numpy as np
    from sklearn.ensemble import IsolationForest

    rows = sample_rows()
    model = fit(rows)
    data = np.array(
        [
            [
                row[name] if row[name] is not None else model.medians[index]
                for index, name in enumerate(model.feature_names)
            ]
            + [float(row[name] is None) for name in model.feature_names]
            for row in rows
        ],
        dtype=np.float32,
    )
    estimator = IsolationForest(
        n_estimators=12,
        max_samples=len(rows),
        random_state=1701,
        n_jobs=2,
        contamination="auto",
    ).fit(data)
    expected = -estimator.score_samples(data)
    predictions = model.predict(rows)
    assert max(abs(p.score - expected[i]) for i, p in enumerate(predictions)) < 1e-10
    assert all(0 < p.rank_quantile <= 1 for p in predictions)


def test_missing_inputs_repeat_json_roundtrip():
    model = fit()
    rows = [{"age": None, "height": 15.0, "diameter": 25.0}]
    loaded = IsolationModel.model_validate_json(model.model_dump_json())
    assert loaded.predict(rows) == model.predict(rows)
    assert model.predict(rows)[0].status == "scored"
    assert "imputed_missing" in model.predict(rows)[0].warnings


def test_insufficient_observed_predictors_abstain():
    prediction = fit().predict([{"age": 20.0, "height": None, "diameter": None}])[0]
    assert prediction.status == "abstained"
    assert prediction.score is None and prediction.rank_quantile is None
    assert "insufficient_observed_features" in prediction.warnings


@pytest.mark.parametrize("role", ["tune", "calibration", "holdout", "test", "validation"])
def test_nontrain_role_rejected_before_fit(role):
    with pytest.raises(ValueError, match="fit_role_not_allowed"):
        fit(role=role)


def test_train_fold_is_accepted():
    assert fit(role="train_fold").fit_role == "train_fold"


def test_small_samples_or_groups_rejected():
    with pytest.raises(ValueError, match="insufficient_training_rows"):
        fit(sample_rows(20))
    with pytest.raises(ValueError, match="insufficient_training_groups"):
        fit_isolation(
            sample_rows(), groups=["only"] * 120, feature_names=("age", "height", "diameter")
        )


def test_rows_without_two_observations_not_training_answers():
    rows = [{"age": 1.0, "height": None, "diameter": None}] * 60
    with pytest.raises(ValueError, match="insufficient_training_rows"):
        fit(rows)


def test_entire_unobserved_feature_cannot_be_imputed_as_fact():
    rows = [{"age": 1.0, "height": 2.0, "diameter": None}] * 60
    with pytest.raises(ValueError, match="no_observed_training_feature"):
        fit(rows)


def test_prediction_cannot_refit_medians_or_rank_reference():
    model = fit()
    before = model.model_dump_json()
    model.predict([{"age": 900.0, "height": 900.0, "diameter": None}])
    assert model.model_dump_json() == before


def test_zero_and_rare_large_value_are_not_missing():
    model = fit()
    p = model.predict([{"age": 0.0, "height": 50000.0, "diameter": 1.0}])[0]
    assert p.status == "scored" and math.isfinite(p.score)
    assert "outside_training_range" in p.warnings
    assert "imputed_missing" not in p.warnings


@pytest.mark.parametrize("value", [True, "20", float("nan"), float("inf")])
def test_invalid_numeric_values_rejected(value):
    with pytest.raises(ValueError, match="invalid_numeric"):
        fit().predict([{"age": value, "height": 20.0, "diameter": 20.0}])


def test_unknown_fields_and_missing_feature_rejected():
    model = fit()
    with pytest.raises(ValueError, match="feature_schema_mismatch"):
        model.predict([{"age": 2.0, "height": 3.0}])
    with pytest.raises(ValueError, match="feature_schema_mismatch"):
        model.predict([{"age": 2.0, "height": 3.0, "diameter": 4.0, "label": 1.0}])


def test_unknown_version_and_extra_fields_rejected():
    content = json.loads(fit().model_dump_json())
    content["schema_version"] = "2.0"
    with pytest.raises(ValueError):
        IsolationModel.model_validate(content)
    content["schema_version"] = "1.0"
    content["unexpected"] = True
    with pytest.raises(ValueError):
        IsolationModel.model_validate(content)


def test_malformed_tree_cycle_and_feature_index_rejected():
    content = json.loads(fit().model_dump_json())
    content["trees"][0]["children_left"][0] = 0
    with pytest.raises(ValueError, match="invalid_tree"):
        IsolationModel.model_validate(content)
    content = json.loads(fit().model_dump_json())
    content["trees"][0]["features"][0] = 999
    with pytest.raises(ValueError, match="tree_feature"):
        IsolationModel.model_validate(content)


def test_missing_ml_dependency_does_not_affect_json_prediction(monkeypatch):
    import builtins

    model = fit()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("sklearn", "numpy")):
            raise ImportError("optional library unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert model.predict(sample_rows(3))[0].status == "scored"
    with pytest.raises(ValueError, match="optional_ml_dependency_unavailable"):
        fit()
