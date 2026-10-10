"""Optional IsolationForest, exported as validated data-only JSON trees.

Scores and empirical training ranks measure unusualness, never error probability.
Split identity / OOF ownership and region/species checks belong to the caller.
Inference has no sklearn/numpy dependency and never loads executable serialization.
"""

import bisect
import math
import struct
from functools import lru_cache
from statistics import median
from typing import Annotated, Literal

from forestcheck.domain.contracts import Contract, Text, ValueObject
from pydantic import Field, StrictFloat, StrictInt, model_validator

Finite = StrictFloat
Index = StrictInt
MAX_TREES = 256
MAX_NODES = 8191
MAX_FLOAT32 = 3.4028234663852886e38


@lru_cache(maxsize=4097)
def _expected_path(sample_count: int) -> float:
    if sample_count <= 1:
        return 0.0
    if sample_count == 2:
        return 1.0
    return (
        2.0 * (math.log(sample_count - 1.0) + 0.5772156649015329)
        - 2.0 * (sample_count - 1.0) / sample_count
    )


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


class IsolationTree(ValueObject):
    children_left: Annotated[tuple[Index, ...], Field(min_length=1, max_length=MAX_NODES)]
    children_right: Annotated[tuple[Index, ...], Field(min_length=1, max_length=MAX_NODES)]
    features: Annotated[tuple[Index, ...], Field(min_length=1, max_length=MAX_NODES)]
    thresholds: Annotated[tuple[Finite, ...], Field(min_length=1, max_length=MAX_NODES)]
    samples: Annotated[tuple[Index, ...], Field(min_length=1, max_length=MAX_NODES)]

    @model_validator(mode="after")
    def valid_graph(self):
        size = len(self.features)
        if any(
            len(column) != size
            for column in (
                self.children_left,
                self.children_right,
                self.thresholds,
                self.samples,
            )
        ) or any(n < 1 for n in self.samples):
            raise ValueError("invalid_tree_array")
        seen, stack = set(), [0]
        while stack:
            node = stack.pop()
            if node < 0 or node >= size or node in seen:
                raise ValueError("invalid_tree_graph")
            seen.add(node)
            left, right = self.children_left[node], self.children_right[node]
            if left == -1 and right == -1:
                if self.features[node] != -2 or self.thresholds[node] != -2.0:
                    raise ValueError("invalid_tree_leaf")
                continue
            if (
                left < 0
                or right < 0
                or left >= size
                or right >= size
                or left == right
                or self.features[node] < 0
            ):
                raise ValueError("invalid_tree_branch")
            if self.samples[left] + self.samples[right] != self.samples[node]:
                raise ValueError("invalid_tree_sample_count")
            stack.extend((left, right))
        if len(seen) != size:
            raise ValueError("invalid_tree_unreachable_nodes")
        return self

    def path_length(self, row: tuple[float, ...]) -> float:
        node, depth = 0, 0
        while self.children_left[node] != -1:
            node = (
                self.children_left[node]
                if row[self.features[node]] <= self.thresholds[node]
                else self.children_right[node]
            )
            depth += 1
        return depth + _expected_path(self.samples[node])


class IsolationPrediction(ValueObject):
    status: Literal["scored", "abstained"]
    score: Annotated[Finite, Field(ge=0, le=1)] | None
    rank_quantile: Annotated[Finite, Field(ge=0, le=1)] | None
    warnings: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "scored") != (self.score is not None and self.rank_quantile is not None):
            raise ValueError("score_status_mismatch")
        if self.status == "abstained" and (
            self.score is not None or self.rank_quantile is not None
        ):
            raise ValueError("abstention_has_score")
        return self


def _validated_rows(rows, feature_names):
    result = []
    names = set(feature_names)
    for row in rows:
        if not isinstance(row, dict) or set(row) != names:
            raise ValueError("feature_schema_mismatch")
        values = []
        for name in feature_names:
            value = row[name]
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or abs(value) > MAX_FLOAT32
            ):
                raise ValueError("invalid_numeric_value")
            values.append(None if value is None else float(value))
        result.append(tuple(values))
    return result


def eligible_isolation_indices(rows, feature_names: tuple[str, ...]) -> tuple[int, ...]:
    """Indices used by IF fit, context support, and training-cohort receipts.

    Require at least two observed finite numeric fields. Missing is exactly None;
    malformed values and feature schemas fail even on a would-be excluded row.
    This eligibility does not attest measurement correctness or biological bounds.
    """
    if (
        not isinstance(feature_names, tuple)
        or not 2 <= len(feature_names) <= 32
        or any(not isinstance(n, str) or not n for n in feature_names)
        or len(set(feature_names)) != len(feature_names)
    ):
        raise ValueError("invalid_feature_names")
    validated = _validated_rows(rows, feature_names)
    return tuple(
        i for i, row in enumerate(validated) if sum(value is not None for value in row) >= 2
    )


class IsolationModel(Contract):
    method_version: Literal["isolation-forest-json-1.0"] = "isolation-forest-json-1.0"
    sklearn_version: Text
    feature_names: Annotated[tuple[Text, ...], Field(min_length=2, max_length=32)]
    seed: StrictInt
    fit_role: Literal["train", "train_fold"]
    train_rows: Annotated[StrictInt, Field(ge=40)]
    train_groups: Annotated[StrictInt, Field(ge=3)]
    max_samples: Annotated[StrictInt, Field(ge=2, le=4096)]
    medians: tuple[Finite, ...]
    minima: tuple[Finite, ...]
    maxima: tuple[Finite, ...]
    trees: Annotated[tuple[IsolationTree, ...], Field(min_length=1, max_length=MAX_TREES)]
    training_scores: Annotated[tuple[Finite, ...], Field(min_length=40, max_length=2_000_000)]

    @model_validator(mode="after")
    def consistent_model(self):
        width = len(self.feature_names)
        if len(set(self.feature_names)) != width:
            raise ValueError("duplicate_feature")
        if any(len(v) != width for v in (self.medians, self.minima, self.maxima)):
            raise ValueError("transform_width_mismatch")
        if any(
            not low <= center <= high or max(abs(low), abs(high)) > MAX_FLOAT32
            for low, center, high in zip(self.minima, self.medians, self.maxima, strict=True)
        ):
            raise ValueError("invalid_transform_range")
        if self.max_samples > self.train_rows or self.train_groups > self.train_rows:
            raise ValueError("training_counts_mismatch")
        for tree in self.trees:
            if tree.samples[0] != self.max_samples:
                raise ValueError("tree_sample_count_mismatch")
            if any(feature >= 2 * width for feature in tree.features):
                raise ValueError("tree_feature_out_of_bounds")
        if (
            len(self.training_scores) != self.train_rows
            or any(not 0 <= score <= 1 for score in self.training_scores)
            or tuple(sorted(self.training_scores)) != self.training_scores
        ):
            raise ValueError("invalid_training_score_distribution")
        return self

    def _score(self, row):
        transformed = tuple(
            _float32(self.medians[i] if value is None else value) for i, value in enumerate(row)
        )
        transformed += tuple(float(value is None) for value in row)
        depth = sum(tree.path_length(transformed) for tree in self.trees)
        return 2.0 ** (-depth / (len(self.trees) * _expected_path(self.max_samples)))

    def predict(self, rows) -> tuple[IsolationPrediction, ...]:
        validated = _validated_rows(rows, self.feature_names)
        predictions = []
        for row in validated:
            if sum(value is not None for value in row) < 2:
                predictions.append(
                    IsolationPrediction(
                        status="abstained",
                        score=None,
                        rank_quantile=None,
                        warnings=("insufficient_observed_features",),
                    )
                )
                continue
            warnings = []
            if any(value is None for value in row):
                warnings.append("imputed_missing")
            if any(
                value is not None and not self.minima[i] <= value <= self.maxima[i]
                for i, value in enumerate(row)
            ):
                warnings.append("outside_training_range")
            score = self._score(row)
            rank = bisect.bisect_right(self.training_scores, score) / len(self.training_scores)
            predictions.append(
                IsolationPrediction(
                    status="scored",
                    score=score,
                    rank_quantile=rank,
                    warnings=tuple(warnings),
                )
            )
        return tuple(predictions)


def fit_isolation(
    rows,
    *,
    groups,
    feature_names: tuple[str, ...],
    seed: int = 1701,
    n_estimators: int = 80,
    max_samples: int = 256,
    role: str = "train",
) -> IsolationModel:
    """Fit numeric-only train data. Caller must bind rows to the frozen manifest.

    At least two observed fields are required per fit/predict row. Medians,
    missing indicators, ranges and rank reference are fitted on eligible train
    rows only. No contamination threshold or binary error label is produced.
    """
    if role not in {"train", "train_fold"}:
        raise ValueError("fit_role_not_allowed")
    if (
        type(seed) is not int
        or not 0 <= seed <= 2**32 - 1
        or type(n_estimators) is not int
        or not 1 <= n_estimators <= MAX_TREES
        or type(max_samples) is not int
        or not 2 <= max_samples <= 4096
    ):
        raise ValueError("invalid_forest_configuration")
    rows, groups = list(rows), list(groups)
    if len(rows) != len(groups) or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("training_group_mismatch")
    eligible_indices = eligible_isolation_indices(rows, feature_names)
    validated = _validated_rows(rows, feature_names)
    eligible = [(validated[i], groups[i]) for i in eligible_indices]
    if len(eligible) < 40:
        raise ValueError("insufficient_training_rows")
    if len({group for _, group in eligible}) < 3:
        raise ValueError("insufficient_training_groups")
    observed = [
        [row[i] for row, _ in eligible if row[i] is not None] for i in range(len(feature_names))
    ]
    if any(not column for column in observed):
        raise ValueError("no_observed_training_feature")
    medians = tuple(float(median(column)) for column in observed)
    try:
        import numpy as np
        import sklearn
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise ValueError("optional_ml_dependency_unavailable") from exc
    data = np.array(
        [
            [medians[i] if value is None else value for i, value in enumerate(row)]
            + [float(value is None) for value in row]
            for row, _ in eligible
        ],
        dtype=np.float32,
    )
    effective_samples = min(max_samples, len(eligible))
    estimator = IsolationForest(
        n_estimators=n_estimators,
        max_samples=effective_samples,
        random_state=seed,
        n_jobs=2,
        contamination="auto",
    ).fit(data)
    trees = tuple(
        IsolationTree(
            children_left=tuple(int(v) for v in tree.tree_.children_left),
            children_right=tuple(int(v) for v in tree.tree_.children_right),
            features=tuple(int(v) for v in tree.tree_.feature),
            thresholds=tuple(float(v) for v in tree.tree_.threshold),
            samples=tuple(int(v) for v in tree.tree_.n_node_samples),
        )
        for tree in estimator.estimators_
    )
    # Use the exact exported inference arithmetic for rank tie reproducibility.
    denominator = len(trees) * _expected_path(effective_samples)
    scores = sorted(
        2.0 ** (-sum(tree.path_length(row) for tree in trees) / denominator)
        for row in data.tolist()
    )
    return IsolationModel(
        schema_version="1.0",
        sklearn_version=sklearn.__version__,
        feature_names=feature_names,
        seed=seed,
        fit_role=role,
        train_rows=len(eligible),
        train_groups=len({group for _, group in eligible}),
        max_samples=effective_samples,
        medians=medians,
        minima=tuple(min(column) for column in observed),
        maxima=tuple(max(column) for column in observed),
        trees=trees,
        training_scores=tuple(scores),
    )
