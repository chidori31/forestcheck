"""Comparable observations, never expert truth or independent model confirmation.

Selection uses exact region, species, age and record level. The caller supplies a
verified mapping and frozen source/split hashes. Full raw-row fingerprints must
include business keys and all raw cells, excluding technical row UUIDs; matching
age/height numbers alone are not evidence of a copied record. Fingerprints only
exclude copies and are not search predictors. This module reads or writes no files.
"""

from collections import Counter, defaultdict
from statistics import median
from types import MappingProxyType
from typing import Annotated, Literal
from uuid import UUID

from forestcheck.domain.contracts import Contract, Digest, Text
from forestcheck.evaluation.split import Role, SplitIdentity, SplitManifest, hash_value
from pydantic import Field, StrictFloat, StrictInt, model_validator

Number = StrictInt | StrictFloat
State = Literal["observed", "missing", "parse_error", "unknown", "not_applicable"]
Count = Annotated[int, Field(strict=True, ge=0)]


class ReferenceQuery(SplitIdentity):
    """Query identity/context, deliberately without a target value or error label."""

    role: Role
    record_level: Text
    region: Text | None
    species: Text | None
    age: Number | None
    age_unit: Text | None
    target_field: Text
    target_unit: Text | None
    raw_record_sha256: Digest


class ReferenceObservation(ReferenceQuery):
    target_value: Number | None
    target_state: State
    source_snapshot_sha256: Digest

    @model_validator(mode="after")
    def observed_target(self):
        if (self.target_state == "observed") != (self.target_value is not None):
            raise ValueError("only_observed_target_has_value")
        return self


class ComparableResult(Contract):
    procedure_version: Literal["comparable-exact-groups-1.0.0"] = "comparable-exact-groups-1.0.0"
    status: Literal["available", "insufficient_data", "unavailable_reference"]
    reason: Text
    target_field: Text
    point: Number | None
    lower: Number | None
    upper: Number | None
    unit: Text
    interval_kind: Literal["descriptive_group_medians_uncalibrated"] = (
        "descriptive_group_medians_uncalibrated"
    )
    n_rows: Count
    n_groups: Count
    n_matching_context: Count
    minimum_rows: Annotated[int, Field(strict=True, ge=1)]
    minimum_groups: Annotated[int, Field(strict=True, ge=2)]
    reference_ids: tuple[UUID, ...]
    reference_ids_sha256: Digest
    excluded_counts: dict[Text, Count]
    dataset_id: UUID
    split_sha256: Digest
    source_snapshot_sha256: Digest
    mapping_version: Text
    mapping_sha256: Digest
    dependency_family: Text
    selection_basis: Text
    limitations: tuple[Text, ...]

    @model_validator(mode="after")
    def result_consistency(self):
        if len(self.reference_ids) != self.n_rows or len(set(self.reference_ids)) != self.n_rows:
            raise ValueError("reference_id_count_mismatch")
        if self.n_groups > self.n_rows:
            raise ValueError("reference_group_count_mismatch")
        if self.reference_ids_sha256 != hash_value(sorted(str(r) for r in self.reference_ids)):
            raise ValueError("reference_ids_digest_mismatch")
        if self.status == "available":
            if (
                self.point is None
                or self.lower is None
                or self.upper is None
                or not self.lower <= self.point <= self.upper
                or self.n_rows < self.minimum_rows
                or self.n_groups < self.minimum_groups
            ):
                raise ValueError("available_reference_requires_values_and_support")
        elif any(value is not None for value in (self.point, self.lower, self.upper)):
            raise ValueError("unavailable_reference_cannot_supply_value")
        return self


def _identity_checked(row: ReferenceQuery, manifest: SplitManifest):
    assignment = manifest.assert_allowed(row.record_id, roles={row.role})
    for key in ("dataset_id", "group_id", "parent_id", "duplicate_id", "role"):
        if getattr(row, key) != getattr(assignment, key):
            raise ValueError("split_identity_mismatch")


def _context(row):
    return row.record_level, row.region, row.species, row.age


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def _copy_components(rows):
    """Join declared groups only by full-raw equality, never scalar coincidence."""
    parents = {r.group_id: r.group_id for r in rows}

    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    first = {}
    for row in rows:
        previous = first.setdefault(row.raw_record_sha256, row.group_id)
        left, right = root(previous), root(row.group_id)
        if left != right:
            parents[max(left, right)] = min(left, right)
    groups = {group: root(group) for group in parents}
    raw = {digest: groups[group] for digest, group in first.items()}
    return MappingProxyType(groups), MappingProxyType(raw)


class ReferencePool:
    """Frozen train-only observations. Query protection is enforced, not advisory.

    Statistics give each surviving declared group component one vote: take its median,
    then the median and linearly interpolated 0.1/0.9 quantiles across those medians.
    Groups linked by exact full raw copies form one component. Query copies exclude
    that whole component, including related rows that differ in context or value.
    This is a descriptive central interval, not a prediction interval or calibrated
    confidence. Minimum support is an explicit engineering policy, not a guarantee.
    """

    def __init__(
        self,
        observations,
        *,
        manifest: SplitManifest,
        dataset_id: UUID,
        target_field: str,
        target_unit: str,
        age_unit: str,
        snapshot_input_key: str,
        snapshot_sha256: str,
        mapping_input_key: str,
        mapping_version: str,
        mapping_sha256: str,
        train_folds=None,
        minimum_rows: int = 20,
        minimum_groups: int = 5,
    ):
        if manifest.strategy == "group_cv":
            raise ValueError("reference_group_cv_executor_not_implemented")
        if manifest.inputs.get(snapshot_input_key) != snapshot_sha256:
            raise ValueError("source_snapshot_not_bound_to_split")
        if manifest.inputs.get(mapping_input_key) != mapping_sha256:
            raise ValueError("mapping_not_bound_to_split")
        if not all(
            isinstance(v, str) and v
            for v in (
                target_field,
                target_unit,
                age_unit,
                mapping_version,
            )
        ):
            raise ValueError("target_units_and_mapping_required")
        if (
            type(minimum_rows) is not int
            or minimum_rows < 1
            or type(minimum_groups) is not int
            or minimum_groups < 2
        ):
            raise ValueError("minimum_reference_support_invalid")
        rows = tuple(ReferenceObservation.model_validate(r.model_dump()) for r in observations)
        if len({r.record_id for r in rows}) != len(rows):
            raise ValueError("duplicate_reference_record")
        for row in rows:
            if row.role != "train":
                raise ValueError("reference_role_not_train")
            if row.dataset_id != dataset_id:
                raise ValueError("reference_dataset_mismatch")
            if row.target_field != target_field:
                raise ValueError("reference_target_field_mismatch")
            _identity_checked(row, manifest)
            if row.source_snapshot_sha256 != snapshot_sha256:
                raise ValueError("reference_snapshot_mismatch")
            if row.target_unit != target_unit or row.age_unit != age_unit:
                raise ValueError("reference_unit_mismatch")
        folds = dict(train_folds or {})
        if folds:
            train_groups = {a.group_id for a in manifest.assignments if a.role == "train"}
            if (
                set(folds) != train_groups
                or any(type(v) is not int or v < 0 for v in folds.values())
                or len(set(folds.values())) < 2
            ):
                raise ValueError("frozen_train_folds_incomplete_or_invalid")
        index = defaultdict(list)
        for row in sorted(rows, key=lambda r: str(r.record_id)):
            index[_context(row)].append(row)
        self._index = MappingProxyType({key: tuple(value) for key, value in index.items()})
        self._group_components, self._raw_components = _copy_components(rows)
        self._folds = MappingProxyType(folds)
        self._manifest = manifest
        self._metadata = MappingProxyType(
            dict(
                target_field=target_field,
                unit=target_unit,
                dataset_id=dataset_id,
                split_sha256=manifest.digest,
                source_snapshot_sha256=snapshot_sha256,
                mapping_version=mapping_version,
                mapping_sha256=mapping_sha256,
                dependency_family="source_snapshot:" + snapshot_sha256,
                minimum_rows=minimum_rows,
                minimum_groups=minimum_groups,
                selection_basis="Exact record_level + region + species + age; no target matching; "
                "join groups linked by full raw copies; equal weight per surviving "
                "group-component median; q10/q90 across group-component medians",
                limitations=(
                    "not_expert_verified",
                    "descriptive_interval_not_calibrated",
                    "minimum_support_is_engineering_policy_not_reliability_guarantee",
                    "same_snapshot_models_and_references_are_dependent_evidence",
                    "declared_group_parent_duplicate_links_and_full_raw_copies_only",
                    "unknown_cross_code_spatial_or_temporal_links_not_resolved",
                    "measurement_procedure_and_true_precision_require_confirmation",
                ),
            )
        )
        self._age_unit = age_unit

    def query(self, query: ReferenceQuery, *, excluded_groups=()) -> ComparableResult:
        # Revalidate even model_copy-created instances; it bypasses Pydantic validators.
        q = ReferenceQuery.model_validate(query.model_dump())
        if q.role not in {"train", "tune"}:
            raise ValueError("query_role_not_allowed")
        return self._query_checked(q, excluded_groups=excluded_groups)

    def query_final(self, query: ReferenceQuery, *, release: dict) -> ComparableResult:
        """Explicit stage-10 route; caller verifies build freeze and one-shot release first."""
        q = ReferenceQuery.model_validate(query.model_dump())
        if (
            release.get("schema_version") != "1.0"
            or release.get("split") != self._metadata["split_sha256"]
            or q.role != "holdout"
            or str(q.record_id) not in release.get("cohort", ())
            or q.group_id not in release.get("groups", ())
            or q.group_id in self._manifest.prior_exposed_groups
        ):
            raise ValueError("final_reference_release_mismatch")
        return self._query_checked(q)

    def _query_checked(self, q: ReferenceQuery, *, excluded_groups=()) -> ComparableResult:
        if q.dataset_id != self._metadata["dataset_id"]:
            raise ValueError("query_dataset_mismatch")
        if q.target_field != self._metadata["target_field"]:
            raise ValueError("query_target_field_mismatch")
        _identity_checked(q, self._manifest)
        excluded = set(excluded_groups)
        if q.role == "train":
            if not self._folds:
                raise ValueError("train_query_requires_frozen_folds")
            fold = self._folds[q.group_id]
            expected = {g for g, f in self._folds.items() if f == fold}
            if not expected <= excluded:
                raise ValueError("complete_validation_fold_exclusion_required")
        if q.target_unit != self._metadata["unit"]:
            return self._result("unavailable_reference", "target_unit_mismatch")
        if q.age_unit != self._age_unit:
            return self._result("unavailable_reference", "age_unit_mismatch")
        if any(v is None for v in (q.age, q.region, q.species)):
            return self._result("insufficient_data", "missing_context")
        matching = self._index.get(_context(q), ())
        if not matching:
            return self._result("unavailable_reference", "no_exact_comparable_context")
        blocked_components = {
            self._group_components[group]
            for group in excluded | {q.group_id}
            if group in self._group_components
        }
        if q.raw_record_sha256 in self._raw_components:
            blocked_components.add(self._raw_components[q.raw_record_sha256])
        selected, counts, fingerprints, families = [], Counter(), set(), set()
        for row in matching:
            if row.group_id in excluded:
                reason = "excluded_fold_group"
            elif row.record_id == q.record_id:
                reason = "current_record"
            elif row.group_id == q.group_id:
                reason = "related_group"
            elif row.parent_id == q.parent_id:
                reason = "related_parent"
            elif row.duplicate_id is not None and row.duplicate_id == q.duplicate_id:
                reason = "query_duplicate_family"
            elif row.raw_record_sha256 == q.raw_record_sha256:
                reason = "query_copy"
            elif self._group_components[row.group_id] in blocked_components:
                reason = "copy_related_group"
            elif row.target_state != "observed":
                reason = "unknown_target"
            elif row.raw_record_sha256 in fingerprints:
                reason = "repeated_raw_copy"
            elif row.duplicate_id is not None and row.duplicate_id in families:
                reason = "repeated_duplicate_family"
            else:
                fingerprints.add(row.raw_record_sha256)
                if row.duplicate_id is not None:
                    families.add(row.duplicate_id)
                selected.append(row)
                continue
            counts[reason] += 1
        groups = {self._group_components[r.group_id] for r in selected}
        if (
            len(selected) < self._metadata["minimum_rows"]
            or len(groups) < self._metadata["minimum_groups"]
        ):
            return self._result(
                "insufficient_data",
                "insufficient_reference_groups_or_rows",
                selected,
                counts,
                len(matching),
            )
        return self._result(
            "available", "descriptive_reference_available", selected, counts, len(matching)
        )

    def _result(self, status, reason, rows=(), counts=None, matching=0):
        groups = defaultdict(list)
        for row in rows:
            groups[self._group_components[row.group_id]].append(row.target_value)
        point = lower = upper = None
        if status == "available":
            values = [median(values) for values in groups.values()]
            point, lower, upper = median(values), _quantile(values, 0.1), _quantile(values, 0.9)
        ids = tuple(sorted((r.record_id for r in rows), key=str))
        return ComparableResult(
            schema_version="1.0",
            status=status,
            reason=reason,
            point=point,
            lower=lower,
            upper=upper,
            n_rows=len(ids),
            n_groups=len(groups),
            n_matching_context=matching,
            reference_ids=ids,
            reference_ids_sha256=hash_value([str(r) for r in ids]),
            excluded_counts=dict(counts or {}),
            **self._metadata,
        )

