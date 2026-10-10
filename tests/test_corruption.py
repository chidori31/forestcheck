"""Synthetic, independent corruption parameters; no biological bounds."""

from hashlib import sha256
from uuid import UUID

import pytest
from forestcheck.domain.contracts import BusinessKey, FieldValue, Record, SourceCoordinates
from forestcheck.evaluation.corruption import (
    CorruptionCase,
    CorruptionSpec,
    case_payloads,
    generate_case,
)
from forestcheck.evaluation.split import SplitIdentity, freeze_split
from pydantic import ValidationError


def field(name, value, unit=None, state="observed"):
    return FieldValue(
        schema_version="1.0",
        name=name,
        original_value=value,
        normalized_value=value,
        unit=unit,
        state=state,
        source=SourceCoordinates(artifact_id=UUID(int=1), table="synthetic", row=1, column=name),
    )


def record(*fields):
    return Record(
        schema_version="1.0",
        id=UUID(int=2),
        dataset_id=UUID(int=3),
        business_key=BusinessKey(stand="0001", layer="2"),
        fields=fields,
    )


def manifest_for(row, wanted="train"):
    ids = [
        SplitIdentity(
            schema_version="1.0",
            record_id=UUID(int=2 + i),
            dataset_id=row.dataset_id,
            group_id=sha256(f"group-{i}".encode()).hexdigest(),
            parent_id=sha256(f"parent-{i}".encode()).hexdigest(),
        )
        for i in range(100)
    ]
    for seed in range(100):
        result = freeze_split(
            ids,
            seed=str(seed),
            grouping_basis="synthetic independent objects",
            inputs={"synthetic": "a" * 64},
        )
        if (
            result.assert_allowed(row.id, roles={"train", "tune", "calibration", "holdout"}).role
            == wanted
        ):
            return result
    raise AssertionError("fixture could not allocate requested role")


def spec(kind, fields, **kwargs):
    return CorruptionSpec.model_validate(
        {
            "schema_version": "1.0",
            "id": "synthetic-change",
            "version": "1.0",
            "kind": kind,
            "fields": fields,
            "source": "Synthetic algebra/declared units, fixture section 1",
            **kwargs,
        }
    )


def eligible(name, data_type="number", unit="m"):
    return {"name": name, "data_type": data_type, "unit": unit}


def generate(row, definition, **kwargs):
    return generate_case(
        row,
        definition,
        manifest=manifest_for(row),
        seed="independent-test-17",
        mode="synthetic_stress",
        source_kind="synthetic",
        **kwargs,
    )


@pytest.mark.parametrize(
    "kind,parameters,expected_value,expected_unit",
    [
        ("missing", {}, None, "m"),
        ("decimal_shift", {"factor": "0.01"}, 0.37, "m"),
        ("wrong_unit", {"replacement_unit": "cm"}, 37, "cm"),
    ],
)
def test_numeric_corruptions_copy_original_and_separate_targets(
    kind,
    parameters,
    expected_value,
    expected_unit,
):
    row = record(field("x", 37, "m"))
    before = row.model_dump_json()
    definition = spec(kind, [eligible("x")], **parameters)
    result = generate(row, definition)
    changed = result.modified.fields[0]
    assert result.original == row and row.model_dump_json() == before
    assert result.modified.id == row.id and result.modified.revision == row.revision + 1
    assert changed.normalized_value == expected_value and changed.unit == expected_unit
    assert changed.original_value == expected_value
    assert result.labels[0].expected_value == 37
    assert result.labels[0].target_basis == "synthetic_contract"
    assert (
        result.lineage.group_id
        == manifest_for(row).assert_allowed(row.id, roles={"train"}).group_id
    )
    assert generate(row, definition) == result
    assert [f.name for f in result.modified.fields] == ["x"]


def test_masked_observation_is_not_claimed_expert_truth():
    row = record(field("x", 37, "m"))
    result = generate_case(
        row,
        spec("missing", [eligible("x")]),
        manifest=manifest_for(row),
        seed="test",
        mode="masked_observation",
        source_kind="real_observation",
    )
    assert result.labels[0].is_error is None
    assert result.labels[0].target_basis == "unverified_source_observation"
    assert result.labels[0].expected_value == 37
    assert result.modified.fields[0].original_value is None


def test_swap_and_invalid_code_have_explicit_eligibility():
    row = record(field("x", 4, "m"), field("y", 9, "m"), field("code", "A"))
    swap = generate(row, spec("field_swap", [eligible("x"), eligible("y")]))
    assert [f.normalized_value for f in swap.modified.fields[:2]] == [9, 4]
    assert len(swap.labels) == 2
    changed = generate(
        row,
        spec(
            "invalid_code",
            [eligible("code", "code", None)],
            allowed_codes=["A", "B"],
            replacement_code="UNKNOWN",
            codebook_version="synthetic-1",
        ),
    )
    assert changed.modified.fields[2].normalized_value == "UNKNOWN"
    with pytest.raises(ValidationError):
        spec(
            "invalid_code",
            [eligible("code", "code", None)],
            allowed_codes=["A"],
            replacement_code="A",
            codebook_version="synthetic-1",
        )


def test_multi_field_contradiction_requires_confirmed_satisfied_relation():
    row = record(field("x", 3, "m"), field("y", 7, "m"), field("total", 21, "m2"))
    definition = spec(
        "multi_field",
        [eligible("x"), eligible("y"), eligible("total", unit="m2")],
        factor="2",
        relation="product",
        tolerance="0",
        relation_scope="same object",
        relation_units_confirmed=True,
    )
    result = generate(row, definition)
    assert [f.normalized_value for f in result.modified.fields] == [6, 14, 21]
    assert {label.field_name for label in result.labels} == {"x", "y"}
    inconsistent = record(field("x", 3, "m"), field("y", 7, "m"), field("total", 22, "m2"))
    with pytest.raises(ValueError, match="original_relation_not_satisfied"):
        generate(inconsistent, definition)


def test_multi_field_relation_cannot_silently_assume_compatible_units():
    with pytest.raises(ValidationError, match="explicit_relation_required"):
        spec(
            "multi_field",
            [eligible("x"), eligible("y"), eligible("total", unit="m2")],
            factor="2",
            relation="product",
            tolerance="0",
            relation_scope="same object",
        )


@pytest.mark.parametrize("factor", ["3", "0.5", "1", "1e1000000", "NaN"])
def test_decimal_shift_requires_bounded_nonzero_base10_exponent(factor):
    with pytest.raises(ValidationError):
        spec("decimal_shift", [eligible("x")], factor=factor)


def test_case_contract_rejects_target_mismatch_and_changed_identity():
    row = record(field("x", 7, "m"))
    case = generate(row, spec("missing", [eligible("x")]))
    payload = case.model_dump(mode="json")
    payload["modified"]["id"] = str(UUID(int=999))
    with pytest.raises(ValidationError):
        CorruptionCase.model_validate(payload)
    payload = case.model_dump(mode="json")
    payload["labels"][0]["expected_value"] = 900
    with pytest.raises(ValidationError):
        CorruptionCase.model_validate(payload)


def test_canonical_split_digest_dataset_identity_and_separate_payloads():
    row = record(field("x", 7139, "m"))
    manifest = manifest_for(row)
    result = generate_case(
        row,
        spec("missing", [eligible("x")]),
        manifest=manifest,
        seed="artifact",
        mode="masked_observation",
        source_kind="real_observation",
    )
    assert result.lineage.split_digest == manifest.digest
    payloads = case_payloads(result)
    assert set(payloads) == {"original.json", "modified.json", "labels.json", "lineage.json"}
    assert "7139" in payloads["original.json"] and "7139" in payloads["labels.json"]
    assert "7139" not in payloads["modified.json"]
    assert "expected_value" not in payloads["modified.json"]
    wrong_dataset = row.model_copy(update={"dataset_id": UUID(int=999)})
    with pytest.raises(ValueError, match="dataset_not_in_assignment"):
        generate_case(
            wrong_dataset,
            spec("missing", [eligible("x")]),
            manifest=manifest,
            seed="artifact",
            mode="synthetic_stress",
            source_kind="synthetic",
        )


@pytest.mark.parametrize("original,target", [(1, True), (0, False), (1, 1.0)])
def test_roundtrip_rejects_type_changed_target(original, target):
    row = record(field("x", original, "m"))
    result = generate(row, spec("missing", [eligible("x")]))
    payload = result.model_dump(mode="json")
    payload["labels"][0]["expected_value"] = target
    with pytest.raises(ValidationError, match="corruption_target_mismatch"):
        CorruptionCase.model_validate(payload)


@pytest.mark.parametrize("kind,parameters", [("missing", {}), ("decimal_shift", {"factor": "10"})])
def test_roundtrip_rejects_hidden_original_raw_in_modified(kind, parameters):
    row = record(field("x", 1, "m"))
    result = generate(row, spec(kind, [eligible("x")], **parameters))
    payload = result.model_dump(mode="json")
    payload["modified"]["fields"][0]["original_value"] = 1
    with pytest.raises(ValidationError, match="modified_raw_target_leak_or_mismatch"):
        CorruptionCase.model_validate(payload)


def cv_manifest_for(row):
    identities = [
        SplitIdentity(
            schema_version="1.0",
            record_id=UUID(int=2 + i),
            dataset_id=row.dataset_id,
            group_id=sha256(f"cv-group-{i}".encode()).hexdigest(),
            parent_id=sha256(f"cv-parent-{i}".encode()).hexdigest(),
        )
        for i in range(9)
    ]
    return freeze_split(
        identities,
        seed="small-independent-fixture",
        grouping_basis="synthetic",
        inputs={"synthetic": "a" * 64},
    )


def test_cv_generation_requires_explicit_fold_and_partition():
    row = record(field("x", 7, "m"))
    manifest = cv_manifest_for(row)
    assignment = manifest.assert_allowed(row.id, roles={"development_cv"})
    kwargs = {
        "manifest": manifest,
        "seed": "fold",
        "mode": "synthetic_stress",
        "source_kind": "synthetic",
    }
    with pytest.raises(ValueError, match="explicit_cv_partition_required"):
        generate_case(row, spec("missing", [eligible("x")]), **kwargs)
    case = generate_case(
        row,
        spec("missing", [eligible("x")]),
        **kwargs,
        validation_fold=assignment.fold,
        cv_partition="validation",
    )
    assert case.lineage.role == "development_cv"
    assert case.lineage.validation_fold == assignment.fold
    assert case.lineage.cv_partition == "validation"
    assert case == CorruptionCase.model_validate_json(case.model_dump_json())
    with pytest.raises(ValueError, match="cv_partition_mismatch"):
        generate_case(
            row,
            spec("missing", [eligible("x")]),
            **kwargs,
            validation_fold=assignment.fold,
            cv_partition="training",
        )
    other_fold = (assignment.fold + 1) % manifest.folds
    training = generate_case(
        row,
        spec("missing", [eligible("x")]),
        **kwargs,
        validation_fold=other_fold,
        cv_partition="training",
    )
    assert training.lineage.cv_partition == "training"
    with pytest.raises(ValueError, match="unexpected_cv_parameters"):
        generate_case(
            row,
            spec("missing", [eligible("x")]),
            manifest=manifest_for(row),
            seed="fold",
            mode="synthetic_stress",
            source_kind="synthetic",
            validation_fold=0,
            cv_partition="validation",
        )


@pytest.mark.parametrize(
    "bad",
    [
        field("x", None, "m", "missing"),
        field("x", None, "m", "not_applicable"),
        field("x", 2, "unknown"),
        field("x", True, "m"),
        field("x", "2", "m"),
    ],
)
def test_ineligible_fields_are_rejected_without_fabricating_label(bad):
    with pytest.raises(ValueError, match="ineligible"):
        generate(record(bad), spec("decimal_shift", [eligible("x")], factor="10"))


def test_zero_and_rare_valid_large_value_are_not_biological_errors():
    row = record(field("x", 0, "m"), field("rare", 90000, "m"))
    with pytest.raises(ValueError, match="no_change"):
        generate(row, spec("decimal_shift", [eligible("x")], factor="10"))
    masked = generate(row, spec("missing", [eligible("x")]))
    assert masked.labels[0].expected_value == 0
    assert masked.modified.fields[1] == row.fields[1]


def test_split_membership_and_holdout_are_checked_before_generation():
    row = record(field("x", 37, "m"))
    with pytest.raises(ValueError):
        generate_case(
            row,
            spec("missing", [eligible("x")]),
            manifest=manifest_for(row, "holdout"),
            seed="test",
            mode="synthetic_stress",
            source_kind="synthetic",
        )
    unrelated = row.model_copy(update={"id": UUID(int=99999)})
    with pytest.raises(ValueError):
        generate_case(
            unrelated,
            spec("missing", [eligible("x")]),
            manifest=manifest_for(row),
            seed="test",
            mode="synthetic_stress",
            source_kind="synthetic",
        )


def test_real_records_cannot_be_declared_fully_synthetic_and_mask_mode_requires_hiding():
    row = record(field("x", 37, "m"))
    with pytest.raises(ValueError, match="synthetic_source_required"):
        generate_case(
            row,
            spec("missing", [eligible("x")]),
            manifest=manifest_for(row),
            seed="test",
            mode="synthetic_stress",
            source_kind="real_observation",
        )
    with pytest.raises(ValueError, match="masking_requires_missing"):
        generate_case(
            row,
            spec("decimal_shift", [eligible("x")], factor="10"),
            manifest=manifest_for(row),
            seed="test",
            mode="masked_observation",
            source_kind="real_observation",
        )


@pytest.mark.parametrize("update", [{"schema_version": "2.0"}, {"hidden_answer": True}])
def test_unknown_contracts_are_rejected(update):
    value = spec("missing", [eligible("x")]).model_dump(mode="json") | update
    with pytest.raises(ValidationError):
        CorruptionSpec.model_validate(value)
