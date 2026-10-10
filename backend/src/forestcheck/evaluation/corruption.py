"""Controlled copies after a frozen split, with evaluation-only target sidecars."""

from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Annotated, Literal, Self
from uuid import UUID, uuid5

from pydantic import Field, StrictBool, model_validator

from forestcheck.domain.contracts import Contract, Digest, FieldValue, Record, Scalar, Text
from forestcheck.evaluation.split import SplitManifest, canonical

Mode = Literal["masked_observation", "synthetic_stress"]
Kind = Literal[
    "missing", "decimal_shift", "wrong_unit", "field_swap", "invalid_code", "multi_field"
]


def _same_scalar(left: Scalar, right: Scalar) -> bool:
    return type(left) is type(right) and left == right


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("invalid_decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid_decimal") from exc
    if not result.is_finite():
        raise ValueError("invalid_decimal")
    return result


class EligibleField(Contract):
    schema_version: Literal["1.0"] = "1.0"
    name: Text
    data_type: Literal["number", "integer", "text", "code"]
    unit: Text | None

    @model_validator(mode="after")
    def numeric_units_known(self) -> Self:
        if self.data_type in {"number", "integer"} and self.unit is None:
            raise ValueError("numeric_unit_required")
        return self


class CorruptionSpec(Contract):
    id: Text
    version: Text
    kind: Kind
    fields: Annotated[tuple[EligibleField, ...], Field(min_length=1)]
    source: Text
    factor: Text | None = None
    replacement_unit: Text | None = None
    allowed_codes: tuple[Text, ...] = ()
    replacement_code: Text | None = None
    codebook_version: Text | None = None
    relation: Literal["product"] | None = None
    tolerance: Text | None = None
    relation_scope: Text | None = None
    relation_units_confirmed: StrictBool = False

    @model_validator(mode="after")
    def operation_contract(self) -> Self:
        count = {"field_swap": 2, "multi_field": 3}.get(self.kind, 1)
        if len(self.fields) != count or len({item.name for item in self.fields}) != count:
            raise ValueError("wrong_field_set")
        permitted = {
            "missing": set(),
            "decimal_shift": {"factor"},
            "wrong_unit": {"replacement_unit"},
            "field_swap": set(),
            "invalid_code": {"allowed_codes", "replacement_code", "codebook_version"},
            "multi_field": {
                "factor",
                "relation",
                "tolerance",
                "relation_scope",
                "relation_units_confirmed",
            },
        }[self.kind]
        for name in (
            "factor",
            "replacement_unit",
            "allowed_codes",
            "replacement_code",
            "codebook_version",
            "relation",
            "tolerance",
            "relation_scope",
            "relation_units_confirmed",
        ):
            if name not in permitted and getattr(self, name) not in (None, (), False):
                raise ValueError("irrelevant_operation_parameter")
        if self.kind in {"decimal_shift", "multi_field"}:
            if self.factor is None or _decimal(self.factor) <= 0 or _decimal(self.factor) == 1:
                raise ValueError("nontrivial_positive_factor_required")
        if self.kind == "decimal_shift":
            digits = _decimal(self.factor).as_tuple()
            exponent = digits.exponent + len(digits.digits) - 1
            # Operational cap on perturbation size, not a forestry validity bound.
            if (
                digits.digits[0] != 1
                or any(digits.digits[1:])
                or exponent == 0
                or abs(exponent) > 12
            ):
                raise ValueError("bounded_power_of_ten_required")
        if self.kind == "wrong_unit":
            if self.replacement_unit is None or self.replacement_unit == self.fields[0].unit:
                raise ValueError("different_replacement_unit_required")
        if self.kind in {"decimal_shift", "wrong_unit", "multi_field"}:
            if any(item.data_type not in {"number", "integer"} for item in self.fields):
                raise ValueError("numeric_contract_required")
        if self.kind == "invalid_code":
            if (
                self.fields[0].data_type != "code"
                or not self.allowed_codes
                or len(set(self.allowed_codes)) != len(self.allowed_codes)
                or self.replacement_code is None
                or self.codebook_version is None
                or self.replacement_code in self.allowed_codes
            ):
                raise ValueError("explicit_codebook_and_invalid_replacement_required")
        if self.kind == "multi_field":
            if (
                self.relation is None
                or self.relation_scope is None
                or self.tolerance is None
                or not self.relation_units_confirmed
            ):
                raise ValueError("explicit_relation_required")
            if _decimal(self.tolerance) < 0:
                raise ValueError("nonnegative_tolerance_required")
        return self


class CorruptionLabel(Contract):
    case_id: UUID
    record_id: UUID
    modified_revision: Annotated[int, Field(strict=True, ge=2)]
    field_name: Text
    expected_value: Scalar
    expected_unit: Text | None
    is_error: StrictBool | None
    target_basis: Literal["unverified_source_observation", "synthetic_contract"]

    @model_validator(mode="after")
    def separate_observation_from_truth(self) -> Self:
        if self.target_basis == "unverified_source_observation" and self.is_error is not None:
            raise ValueError("unverified_observation_is_not_error_truth")
        if self.target_basis == "synthetic_contract" and self.is_error is not True:
            raise ValueError("synthetic_corruption_label_required")
        return self


class CorruptionLineage(Contract):
    case_id: UUID
    mode: Mode
    source_kind: Literal["synthetic", "real_observation"]
    split_digest: Digest
    group_id: Digest
    role: Literal["train", "tune", "calibration", "development_cv"]
    validation_fold: Annotated[int, Field(strict=True, ge=0)] | None = None
    cv_partition: Literal["training", "validation"] | None = None
    seed: Text
    spec_id: Text
    spec_version: Text
    spec_digest: Digest
    kind: Kind
    source: Text

    @model_validator(mode="after")
    def explicit_fold_context(self) -> Self:
        if self.role == "development_cv":
            if self.validation_fold is None or self.cv_partition is None:
                raise ValueError("explicit_cv_partition_required")
        elif self.validation_fold is not None or self.cv_partition is not None:
            raise ValueError("unexpected_cv_parameters")
        return self


class CorruptionCase(Contract):
    """Persist these four payloads separately; only modified fields can be inputs."""

    original: Record
    modified: Record
    labels: tuple[CorruptionLabel, ...]
    lineage: CorruptionLineage

    @model_validator(mode="after")
    def copy_and_target_integrity(self) -> Self:
        original, modified = self.original, self.modified
        if (
            original.id != modified.id
            or original.dataset_id != modified.dataset_id
            or original.dataset_revision != modified.dataset_revision
            or original.business_key != modified.business_key
            or modified.revision != original.revision + 1
        ):
            raise ValueError("modified_identity_or_revision_mismatch")
        by_name = {item.name: item for item in original.fields}
        if [item.name for item in original.fields] != [item.name for item in modified.fields]:
            raise ValueError("modified_field_set_mismatch")
        changed = {
            item.name
            for item in modified.fields
            if item.model_dump_json() != by_name[item.name].model_dump_json()
        }
        if (
            not changed
            or len(self.labels) != len(changed)
            or {label.field_name for label in self.labels} != changed
        ):
            raise ValueError("changed_field_labels_mismatch")
        expected_basis = (
            "synthetic_contract"
            if self.lineage.mode == "synthetic_stress"
            else "unverified_source_observation"
        )
        if self.lineage.mode == "synthetic_stress" and self.lineage.source_kind != "synthetic":
            raise ValueError("synthetic_source_required")
        if self.lineage.mode == "masked_observation" and self.lineage.kind != "missing":
            raise ValueError("masking_requires_missing_operation")
        for item in modified.fields:
            if item.name not in changed:
                continue
            if not _same_scalar(item.original_value, item.normalized_value):
                raise ValueError("modified_raw_target_leak_or_mismatch")
            if self.lineage.kind == "missing" and (
                item.state != "missing" or item.normalized_value is not None
            ):
                raise ValueError("masked_value_not_hidden")
        for label in self.labels:
            target = by_name[label.field_name]
            if (
                label.case_id != self.lineage.case_id
                or label.record_id != original.id
                or label.modified_revision != modified.revision
                or not _same_scalar(label.expected_value, target.normalized_value)
                or label.expected_unit != target.unit
                or label.target_basis != expected_basis
            ):
                raise ValueError("corruption_target_mismatch")
        return self


def _eligible(field: FieldValue, spec: EligibleField) -> None:
    value = field.normalized_value
    if field.state != "observed" or field.unit != spec.unit:
        raise ValueError("ineligible_state_or_unit")
    if spec.data_type == "number" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError("ineligible_type")
    if spec.data_type == "integer" and type(value) is not int:
        raise ValueError("ineligible_type")
    if spec.data_type in {"text", "code"} and not isinstance(value, str):
        raise ValueError("ineligible_type")


def _number(value: Decimal) -> int | float:
    if not value.is_finite():
        raise ValueError("nonfinite_corruption")
    if value == value.to_integral_value():
        return int(value)
    result = float(value)
    if not _decimal(result).is_finite():
        raise ValueError("nonfinite_corruption")
    return result


def _replace(field: FieldValue, value: Scalar, *, unit: str | None = None) -> FieldValue:
    return FieldValue.model_validate(
        field.model_dump()
        | {
            "original_value": value,
            "normalized_value": value,
            "state": "missing" if value is None else "observed",
            "unit": unit if unit is not None else field.unit,
        }
    )


def generate_case(
    record: Record,
    spec: CorruptionSpec,
    *,
    manifest: SplitManifest,
    seed: str,
    mode: Mode,
    source_kind: Literal["synthetic", "real_observation"],
    validation_fold: int | None = None,
    cv_partition: Literal["training", "validation"] | None = None,
) -> CorruptionCase:
    """No holdout access or original mutation; no inferred biological validity.

    A synthetic source declaration is required for fully synthetic stress tests.
    Masked real observations are reconstruction targets, never expert truth.
    """
    if not seed:
        raise ValueError("seed_required")
    if mode not in {"masked_observation", "synthetic_stress"}:
        raise ValueError("unknown_evaluation_mode")
    if source_kind not in {"synthetic", "real_observation"}:
        raise ValueError("unknown_source_kind")
    if mode == "synthetic_stress" and source_kind != "synthetic":
        raise ValueError("synthetic_source_required")
    if mode == "masked_observation" and spec.kind != "missing":
        raise ValueError("masking_requires_missing_operation")
    assignment = manifest.assert_allowed(
        record.id, roles={"train", "tune", "calibration", "development_cv"}
    )
    if assignment.role == "development_cv":
        if (
            type(validation_fold) is not int
            or validation_fold < 0
            or validation_fold >= manifest.folds
            or cv_partition not in {"training", "validation"}
        ):
            raise ValueError("explicit_cv_partition_required")
        if (assignment.fold == validation_fold) != (cv_partition == "validation"):
            raise ValueError("cv_partition_mismatch")
    elif validation_fold is not None or cv_partition is not None:
        raise ValueError("unexpected_cv_parameters")
    if assignment.dataset_id != record.dataset_id:
        raise ValueError("dataset_not_in_assignment")
    by_name = {item.name: item for item in record.fields}
    chosen = []
    for item in spec.fields:
        if item.name not in by_name:
            raise ValueError("ineligible_absent_field")
        current = by_name[item.name]
        _eligible(current, item)
        chosen.append(current)
    changes: dict[str, FieldValue] = {}
    first = chosen[0]
    if spec.kind == "missing":
        changes[first.name] = _replace(first, None)
    elif spec.kind == "decimal_shift":
        changes[first.name] = _replace(
            first, _number(_decimal(first.normalized_value) * _decimal(spec.factor))
        )
    elif spec.kind == "wrong_unit":
        changes[first.name] = _replace(first, first.normalized_value, unit=spec.replacement_unit)
    elif spec.kind == "field_swap":
        second = chosen[1]
        changes[first.name] = _replace(first, second.normalized_value)
        changes[second.name] = _replace(second, first.normalized_value)
    elif spec.kind == "invalid_code":
        if first.normalized_value not in spec.allowed_codes:
            raise ValueError("ineligible_unknown_original_code")
        changes[first.name] = _replace(first, spec.replacement_code)
    elif spec.kind == "multi_field":
        left, right, total = (_decimal(item.normalized_value) for item in chosen)
        tolerance = _decimal(spec.tolerance)
        if abs(left * right - total) > tolerance:
            raise ValueError("original_relation_not_satisfied")
        factor = _decimal(spec.factor)
        if abs(left * factor * right * factor - total) <= tolerance:
            raise ValueError("no_contradiction_created")
        for item in chosen[:2]:
            changes[item.name] = _replace(item, _number(_decimal(item.normalized_value) * factor))
    if any(
        (changed.normalized_value, changed.unit, changed.state)
        == (by_name[name].normalized_value, by_name[name].unit, by_name[name].state)
        for name, changed in changes.items()
    ):
        raise ValueError("no_change")
    modified = Record.model_validate(
        record.model_dump()
        | {
            "revision": record.revision + 1,
            "fields": [changes.get(item.name, item) for item in record.fields],
        }
    )
    split_digest = manifest.digest
    spec_digest = sha256(spec.model_dump_json().encode()).hexdigest()
    case_id = uuid5(
        record.id,
        f"{record.revision}:{split_digest}:{spec_digest}:{seed}:{mode}:{source_kind}:{validation_fold}:{cv_partition}",
    )
    basis = "synthetic_contract" if mode == "synthetic_stress" else "unverified_source_observation"
    labels = tuple(
        CorruptionLabel(
            schema_version="1.0",
            case_id=case_id,
            record_id=record.id,
            modified_revision=modified.revision,
            field_name=name,
            expected_value=by_name[name].normalized_value,
            expected_unit=by_name[name].unit,
            is_error=True if mode == "synthetic_stress" else None,
            target_basis=basis,
        )
        for name in changes
    )
    return CorruptionCase(
        schema_version="1.0",
        original=record,
        modified=modified,
        labels=labels,
        lineage=CorruptionLineage(
            schema_version="1.0",
            case_id=case_id,
            mode=mode,
            source_kind=source_kind,
            split_digest=split_digest,
            group_id=assignment.group_id,
            role=assignment.role,
            validation_fold=validation_fold,
            cv_partition=cv_partition,
            seed=seed,
            spec_id=spec.id,
            spec_version=spec.version,
            spec_digest=spec_digest,
            kind=spec.kind,
            source=spec.source,
        ),
    )


def case_payloads(case: CorruptionCase) -> dict[str, str]:
    """Independent payloads for a caller enforcing private-directory persistence.

    This function performs no I/O. Never append the label/lineage payloads to a
    feature table. The original source snapshot is not present in modified.json.
    """
    return {
        "original.json": canonical(case.original.model_dump(mode="json")),
        "modified.json": canonical(case.modified.model_dump(mode="json")),
        "labels.json": canonical([label.model_dump(mode="json") for label in case.labels]),
        "lineage.json": canonical(case.lineage.model_dump(mode="json")),
    }
