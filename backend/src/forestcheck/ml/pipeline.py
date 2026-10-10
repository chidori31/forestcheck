"""Conditional observation models with explicit train support and abstention.

Targets are observations, not expert truth. No error classifier is constructed.
File trust, mapping identity and frozen split membership are checked by the caller;
this module additionally rejects unsafe roles, unexpected fields and group reuse.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from forestcheck.domain.contracts import Contract, Text
from forestcheck.evaluation.leakage import FeatureColumn, validate_features
from pydantic import Field, StrictInt, model_validator

Count = Annotated[int, Field(strict=True, ge=1)]


class TargetSpec(Contract):
    target: Text
    unit: Text
    numeric_features: tuple[Text, ...]
    categorical_features: tuple[Text, ...] = ()
    context_fields: tuple[Text, ...] = ()
    required_categories: tuple[Text, ...] = ()
    min_numeric_features: Count = 1
    min_train_rows: Count = 40
    min_train_groups: Count = 3
    min_conditional_rows: Count = 10

    @model_validator(mode="after")
    def safe_features(self) -> Self:
        names = self.numeric_features + self.categorical_features + self.context_fields
        if not self.numeric_features or self.min_numeric_features > len(self.numeric_features):
            raise ValueError("insufficient_independent_features_spec")
        if self.target in names or len(set(names)) != len(names):
            raise ValueError("target_or_duplicate_feature")
        validate_features(
            tuple(
                FeatureColumn(schema_version="1.0", name=n, source_kind="observed_field")
                for n in names
            )
        )
        if not set(self.required_categories) <= set(self.categorical_features):
            raise ValueError("required_categories_not_features")
        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Context is support metadata and is never passed to an estimator."""
        return self.numeric_features + self.categorical_features


class Prediction(Contract):
    status: Literal["ok", "abstained"]
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    unit: Text
    warnings: tuple[str, ...] = ()
    interval_status: Literal["uncalibrated", "calibrated_group_max", "unavailable"]
    baseline_level: Literal["conditional", "category", "global"] | None = None
    robust_scale: float | None = None


class _Stats(Contract):
    count: Count
    median: float
    lower: float
    upper: float
    mad_scale: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.lower <= self.median <= self.upper:
            raise ValueError("unordered_baseline_interval")
        return self


class _Calibration(Contract):
    status: Literal["uncalibrated", "uncalibrated_insufficient_groups", "calibrated_group_max"]
    groups: Annotated[int, Field(strict=True, ge=0)] = 0
    rows: Annotated[int, Field(strict=True, ge=0)] = 0
    nominal_coverage: Literal[0.8] = 0.8
    adjustment: Annotated[float, Field(ge=0)] | None = None
    reference: Literal["unverified_observations"] = "unverified_observations"

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.status == "calibrated_group_max") != (self.adjustment is not None):
            raise ValueError("calibration_state_mismatch")
        return self


class _PipelineState(Contract):
    method: Literal["catboost_multi_quantile_0.1_0.5_0.9"]
    spec: TargetSpec
    seed: StrictInt
    train_count: Count
    train_groups: tuple[Text, ...]
    numeric_support: dict[str, tuple[float, float] | None]
    category_support: dict[str, tuple[str, ...]]
    context_support: dict[str, tuple[str, ...]]
    numeric_condition_cuts: tuple[float, ...]
    global_stats: _Stats
    conditional_stats: dict[str, _Stats]
    category_stats: dict[str, _Stats]
    calibrations: dict[Literal["catboost", "baseline"], _Calibration]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if set(self.numeric_support) != set(self.spec.numeric_features):
            raise ValueError("numeric_support_mismatch")
        if set(self.category_support) != set(self.spec.categorical_features):
            raise ValueError("category_support_mismatch")
        if set(self.context_support) != set(self.spec.context_fields):
            raise ValueError("context_support_mismatch")
        if len(set(self.train_groups)) != len(self.train_groups):
            raise ValueError("duplicate_training_group")
        if len(self.train_groups) < self.spec.min_train_groups:
            raise ValueError("insufficient_training_groups_state")
        if self.train_count < self.spec.min_train_rows:
            raise ValueError("insufficient_training_rows_state")
        for bounds in self.numeric_support.values():
            if bounds is not None and bounds[0] > bounds[1]:
                raise ValueError("invalid_numeric_support")
        if set(self.calibrations) != {"catboost", "baseline"}:
            raise ValueError("calibration_methods_mismatch")
        if tuple(sorted(set(self.numeric_condition_cuts))) != self.numeric_condition_cuts:
            raise ValueError("unordered_numeric_condition_cuts")
        if len(self.numeric_condition_cuts) > 4:
            raise ValueError("too_many_numeric_condition_cuts")
        return self


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def robust_deviation(observation, prediction: Prediction) -> float | None:
    """Absolute median/MAD deviation. Unbounded descriptive score, not error odds.

    For training rows the caller must supply out-of-fold baseline predictions.
    A zero MAD carries no stable scale and deliberately returns no score.
    """
    if (
        not _number(observation)
        or prediction.status != "ok"
        or prediction.robust_scale is None
        or prediction.robust_scale <= 0
    ):
        return None
    return abs(observation - prediction.value) / prediction.robust_scale


def _category(value) -> str:
    if value is None:
        return '["missing"]'
    if type(value) not in (str, int, float, bool):
        raise ValueError("unsupported_category_type")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite_category")
    return json.dumps(
        [type(value).__name__, value], ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _validated_rows(rows, spec: TargetSpec) -> list[dict]:
    names = set(spec.feature_names + spec.context_fields)
    checked = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != names:
            raise ValueError("feature_keys_mismatch")
        for name in spec.numeric_features:
            if row[name] is not None and not _number(row[name]):
                raise ValueError("invalid_numeric_predictor")
        for name in spec.categorical_features + spec.context_fields:
            _category(row[name])
        checked.append(dict(row))
    return checked


def eligible_training_indices(rows, targets, spec: TargetSpec) -> tuple[int, ...]:
    """Ordered indices used by fit and provenance receipts, without fitting state.

    Validate the entire supplied feature schema, including rejected target rows.
    Keep finite numeric observations with enough independent numeric inputs and
    all required category/context inputs. Optional numeric missingness is allowed.
    This is eligibility, not a declaration that retained observations are true.
    """
    checked = _validated_rows(rows, spec)
    if len(checked) != len(targets):
        raise ValueError("fit_lengths_mismatch")
    return tuple(
        i
        for i, (row, target) in enumerate(zip(checked, targets, strict=True))
        if _number(target)
        and sum(row[n] is not None for n in spec.numeric_features) >= spec.min_numeric_features
        and all(row[n] is not None for n in spec.required_categories + spec.context_fields)
    )


def _matrix(rows: list[dict], spec: TargetSpec) -> list[list]:
    return [
        [
            *(float(row[n]) if row[n] is not None else np.nan for n in spec.numeric_features),
            *(_category(row[n]) for n in spec.categorical_features),
        ]
        for row in rows
    ]


def _category_condition(row: dict, spec: TargetSpec) -> str:
    return json.dumps(
        [_category(row[n]) for n in spec.categorical_features],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _condition(row: dict, spec: TargetSpec, cuts: tuple[float, ...]) -> str:
    value = row[spec.numeric_features[0]]
    numeric_bin = None if value is None else int(np.searchsorted(cuts, value, side="right"))
    return json.dumps(
        [_category_condition(row, spec), numeric_bin], ensure_ascii=False, separators=(",", ":")
    )


def _stats(values: list[float]) -> _Stats:
    lower, median, upper = np.quantile(values, [0.1, 0.5, 0.9])
    # Normal-consistency scaling is descriptive, not a Gaussian truth claim.
    mad_scale = 1.4826 * np.median(np.abs(np.asarray(values) - median))
    return _Stats(
        schema_version="1.0",
        count=len(values),
        lower=float(lower),
        median=float(median),
        upper=float(upper),
        mad_scale=float(mad_scale),
    )


class _BaselineView:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, rows, *, calibrated=True) -> list[Prediction]:
        return self.pipeline.predict(rows, method="baseline", calibrated=calibrated)


class ConditionalPipeline:
    def __init__(self, state: _PipelineState, model):
        self._state = state
        self._model = model
        self.baseline = _BaselineView(self)

    @property
    def spec(self) -> TargetSpec:
        return self._state.spec

    @property
    def train_count(self) -> int:
        return self._state.train_count

    @property
    def calibration(self) -> dict:
        return self._state.calibrations["catboost"].model_dump(mode="json")

    def export_state(self) -> dict:
        return self._state.model_dump(mode="json")

    def save_catboost(self, path: Path) -> None:
        """Only the native CBM format; caller supplies a checked private destination."""
        if self._model is None:
            raise ValueError("candidate_unavailable")
        self._model.save_model(str(path), format="cbm")

    @classmethod
    def from_state(cls, state: dict, model_path: Path | None) -> Self:
        parsed = _PipelineState.model_validate(state)
        model = None
        if model_path is not None:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor()
            model.load_model(str(model_path), format="cbm")
            if tuple(model.feature_names_) != parsed.spec.feature_names:
                raise ValueError("native_model_feature_mismatch")
            if model.get_all_params().get("loss_function") != "MultiQuantile:alpha=0.1,0.5,0.9":
                raise ValueError("native_model_objective_mismatch")
        return cls(parsed, model)

    def _support(self, row: dict) -> tuple[bool, tuple[str, ...]]:
        warnings = []
        present = sum(row[n] is not None for n in self.spec.numeric_features)
        if present < self.spec.min_numeric_features:
            return False, ("insufficient_independent_numeric_inputs",)
        if present < len(self.spec.numeric_features):
            warnings.append("partial_missing_numeric_inputs")
        for name in self.spec.context_fields:
            if row[name] is None or _category(row[name]) not in self._state.context_support[name]:
                return False, ("unknown_or_missing_context:" + name,)
        for name in self.spec.categorical_features:
            if name in self.spec.required_categories and row[name] is None:
                return False, ("missing_required_category:" + name,)
            if _category(row[name]) not in self._state.category_support[name]:
                return False, ("unknown_category:" + name,)
        for name in self.spec.numeric_features:
            value, bounds = row[name], self._state.numeric_support[name]
            if value is not None and (bounds is None or not bounds[0] <= value <= bounds[1]):
                return False, ("outside_train_range:" + name,)
        stats = self._state.category_stats.get(_category_condition(row, self.spec))
        if stats is None or stats.count < self.spec.min_conditional_rows:
            warnings.append("rare_category_combination")
        return True, tuple(warnings)

    def predict(self, rows, *, method="catboost", calibrated=True) -> list[Prediction]:
        if method not in {"catboost", "baseline"}:
            raise ValueError("unknown_prediction_method")
        checked = _validated_rows(rows, self.spec)
        supports = [self._support(row) for row in checked]
        eligible = [i for i, (valid, _) in enumerate(supports) if valid]
        estimates = {}
        if method == "catboost" and eligible and self._model is not None:
            raw = self._model.predict(
                _matrix([checked[i] for i in eligible], self.spec), thread_count=2
            )
            estimates = dict(zip(eligible, np.sort(raw, axis=1), strict=True))
        output = []
        calibration = self._state.calibrations[method]
        adjustment = calibration.adjustment if calibrated else None
        for i, row in enumerate(checked):
            valid, warnings = supports[i]
            if not valid or (method == "catboost" and self._model is None):
                output.append(
                    Prediction(
                        schema_version="1.0",
                        status="abstained",
                        unit=self.spec.unit,
                        interval_status="unavailable",
                        warnings=warnings or ("candidate_unavailable",),
                    )
                )
                continue
            level, robust_scale = None, None
            if method == "baseline":
                stats = self._state.conditional_stats.get(
                    _condition(row, self.spec, self._state.numeric_condition_cuts)
                )
                if stats is None or stats.count < self.spec.min_conditional_rows:
                    stats = self._state.category_stats.get(_category_condition(row, self.spec))
                    if stats is None or stats.count < self.spec.min_conditional_rows:
                        stats, level = self._state.global_stats, "global"
                        warnings += ("small_conditional_group_global_fallback",)
                    else:
                        level = "category"
                        warnings += ("small_numeric_bin_category_fallback",)
                else:
                    level = "conditional"
                lower, value, upper = stats.lower, stats.median, stats.upper
                robust_scale = stats.mad_scale
                if robust_scale == 0:
                    warnings += ("zero_mad_no_standardized_deviation",)
            else:
                lower, value, upper = map(float, estimates[i])
            if adjustment is not None:
                lower, upper = lower - adjustment, upper + adjustment
            output.append(
                Prediction(
                    schema_version="1.0",
                    status="ok",
                    value=value,
                    lower=lower,
                    upper=upper,
                    unit=self.spec.unit,
                    warnings=warnings,
                    robust_scale=robust_scale,
                    baseline_level=level,
                    interval_status=(
                        "calibrated_group_max" if adjustment is not None else "uncalibrated"
                    ),
                )
            )
        return output

    def calibrate(
        self, rows, targets, groups, *, role="calibration", min_groups=20, method="catboost"
    ) -> dict:
        """CQR: one maximum score per group. Calibration is not its own evaluation.

        Coverage requires suitable exchangeability of independent groups, which
        must be justified separately. Only the 80% protocol is implemented.
        """
        if role != "calibration":
            raise ValueError("calibration_role_required")
        if method not in self._state.calibrations:
            raise ValueError("unknown_prediction_method")
        if type(min_groups) is not int or min_groups < 2:
            raise ValueError("invalid_calibration_minimum")
        if len(rows) != len(targets) or len(rows) != len(groups):
            raise ValueError("calibration_lengths_mismatch")
        if any(type(g) is not str or not g for g in groups):
            raise ValueError("invalid_calibration_groups")
        if set(groups) & set(self._state.train_groups):
            raise ValueError("calibration_train_overlap")
        if self._state.calibrations[method].status == "calibrated_group_max":
            raise ValueError("calibration_already_fitted")
        predictions = self.predict(rows, method=method, calibrated=False)
        scores = defaultdict(list)
        for prediction, target, group in zip(predictions, targets, groups, strict=True):
            if prediction.status == "ok" and _number(target):
                scores[group].append(max(prediction.lower - target, target - prediction.upper, 0.0))
        maxima = sorted(max(values) for values in scores.values())
        count = len(maxima)
        order = math.ceil((count + 1) * 0.8)
        eligible = count >= min_groups and order <= count
        cal = _Calibration(
            schema_version="1.0",
            groups=count,
            rows=sum(map(len, scores.values())),
            status=("calibrated_group_max" if eligible else "uncalibrated_insufficient_groups"),
            adjustment=float(maxima[order - 1]) if eligible else None,
        )
        updated = self.export_state()
        updated["calibrations"][method] = cal.model_dump(mode="json")
        self._state = _PipelineState.model_validate(updated)
        return cal.model_dump(mode="json")


def fit_pipeline(
    rows,
    targets,
    groups,
    spec: TargetSpec,
    *,
    role="train",
    seed=1701,
    iterations=120,
    thread_count=2,
    train_candidate=True,
) -> ConditionalPipeline:
    """Fit on an explicitly verified train cohort; target None/nonfinite is unknown.

    Root orchestration must verify UUID membership against the frozen split before
    invoking this low-level function; a role string is not a split authorization.
    """
    if role not in {"train", "train_fold"}:
        raise ValueError("fit_role_not_train")
    if type(iterations) is not int or not 1 <= iterations <= 120:
        raise ValueError("iterations_outside_cpu_budget")
    if type(thread_count) is not int or not 1 <= thread_count <= 2:
        raise ValueError("threads_outside_cpu_budget")
    if type(seed) is not int:
        raise ValueError("invalid_seed")
    checked = _validated_rows(rows, spec)
    if len(checked) != len(targets) or len(checked) != len(groups):
        raise ValueError("fit_lengths_mismatch")
    if any(type(g) is not str or not g for g in groups):
        raise ValueError("invalid_fit_groups")
    eligible = eligible_training_indices(checked, targets, spec)
    kept = [(checked[i], float(targets[i]), groups[i]) for i in eligible]
    train_groups = sorted({g for _, _, g in kept})
    if len(kept) < spec.min_train_rows or len(train_groups) < spec.min_train_groups:
        raise ValueError("insufficient_training_rows_or_groups")
    fit_rows, fit_targets, _ = zip(*kept, strict=True)
    fit_rows, fit_targets = list(fit_rows), list(fit_targets)
    if len(set(fit_targets)) < 2:
        raise ValueError("insufficient_target_variation")
    numeric_support = {}
    for name in spec.numeric_features:
        values = [float(r[name]) for r in fit_rows if r[name] is not None]
        numeric_support[name] = (min(values), max(values)) if values else None
    if not any(
        bounds is not None and bounds[0] != bounds[1] for bounds in numeric_support.values()
    ):
        raise ValueError("insufficient_numeric_feature_variation")
    bin_values = [
        r[spec.numeric_features[0]] for r in fit_rows if r[spec.numeric_features[0]] is not None
    ]
    cuts = (
        tuple(
            sorted(
                {
                    float(v)
                    for v in np.quantile(bin_values, [0.2, 0.4, 0.6, 0.8])
                    if min(bin_values) < v < max(bin_values)
                }
            )
        )
        if bin_values
        else ()
    )
    conditions, categories = defaultdict(list), defaultdict(list)
    for row, target in zip(fit_rows, fit_targets, strict=True):
        conditions[_condition(row, spec, cuts)].append(target)
        categories[_category_condition(row, spec)].append(target)
    state = _PipelineState(
        schema_version="1.0",
        method="catboost_multi_quantile_0.1_0.5_0.9",
        spec=spec,
        seed=seed,
        train_count=len(kept),
        train_groups=tuple(train_groups),
        numeric_support=numeric_support,
        category_support={
            n: tuple(sorted({_category(r[n]) for r in fit_rows})) for n in spec.categorical_features
        },
        context_support={
            n: tuple(sorted({_category(r[n]) for r in fit_rows})) for n in spec.context_fields
        },
        numeric_condition_cuts=cuts,
        global_stats=_stats(fit_targets),
        conditional_stats={key: _stats(values) for key, values in conditions.items()},
        category_stats={key: _stats(values) for key, values in categories.items()},
        calibrations={
            name: _Calibration(schema_version="1.0", status="uncalibrated")
            for name in ("catboost", "baseline")
        },
    )
    model = None
    if train_candidate:
        from catboost import CatBoostRegressor, Pool

        model = CatBoostRegressor(
            iterations=iterations,
            depth=5,
            learning_rate=0.06,
            loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
            random_seed=seed,
            thread_count=thread_count,
            task_type="CPU",
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            Pool(
                _matrix(fit_rows, spec),
                label=fit_targets,
                cat_features=list(range(len(spec.numeric_features), len(spec.feature_names))),
                feature_names=list(spec.feature_names),
            )
        )
    return ConditionalPipeline(state, model)
