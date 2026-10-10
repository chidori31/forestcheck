"""Allowlist and fit boundary for future transformations. No estimator is trained here."""

import re
from typing import Literal
from uuid import UUID

from forestcheck.domain.contracts import Contract, Digest, Text
from forestcheck.evaluation.split import SplitManifest, hash_value


class FeatureColumn(Contract):
    name: Text
    source_kind: Literal["observed_field", "label", "provenance", "identifier"]


class FitReceipt(Contract):
    split_sha256: Digest
    feature_sha256: Digest
    record_ids: tuple[UUID, ...]
    group_ids: tuple[Digest, ...]
    validation_fold: int | None


RESERVED = {
    "is_error", "expected_value", "original_value", "label", "labels", "target",
    "split", "fold", "group_id", "record_id", "dataset_id", "error_assessment",
    "correction_assessment", "suggestion_decision", "review_status", "truth",
}


def validate_features(columns: tuple[FeatureColumn, ...]) -> None:
    if not columns or len({c.name for c in columns}) != len(columns):
        raise ValueError("empty_or_duplicate_features")
    for column in columns:
        if column.source_kind != "observed_field":
            raise ValueError("non_predictor_feature")
        name = re.sub(r"[^a-z0-9]+", "_", column.name.casefold()).strip("_")
        if name in RESERVED or name.startswith(("corruption_", "synthetic_", "label_", "target_")):
            raise ValueError("reserved_feature")


def fit_transformer(transformer, samples: dict[UUID, dict], *, manifest: SplitManifest,
                    columns: tuple[FeatureColumn, ...], validation_fold: int | None = None):
    """All checks run before calling fit. Caller must declare honest field lineage;
    no name-based guard can detect a renamed target hidden in observed values.
    """
    validate_features(columns)
    if not samples:
        raise ValueError("empty_fit")
    if manifest.strategy == "group_cv":
        if type(validation_fold) is not int or not 0 <= validation_fold < manifest.folds:
            raise ValueError("cv_validation_fold_required")
    elif validation_fold is not None:
        raise ValueError("unexpected_validation_fold")
    assignments = {a.record_id: a for a in manifest.assignments}
    selected, names = [], {c.name for c in columns}
    for rid, row in samples.items():
        if rid not in assignments:
            raise ValueError("record_not_in_split")
        a = assignments[rid]
        if manifest.strategy == "group_cv":
            if a.fold == validation_fold:
                raise ValueError("cv_validation_leakage")
        elif a.role != "train":
            raise ValueError("role_not_allowed")
        if set(row) != names:
            raise ValueError("feature_keys_mismatch")
        selected.append(a)
    ordered = sorted(samples, key=str)
    transformer.fit([{c.name: samples[rid][c.name] for c in columns} for rid in ordered])
    return FitReceipt(
        schema_version="1.0", split_sha256=manifest.digest,
        feature_sha256=hash_value([c.model_dump(mode="json") for c in columns]),
        record_ids=tuple(ordered), group_ids=tuple(sorted({a.group_id for a in selected})),
        validation_fold=validation_fold,
    )
