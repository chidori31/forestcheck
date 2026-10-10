from uuid import UUID

import pytest
from forestcheck.domain.contracts import BusinessKey, FieldValue, Record, SourceCoordinates
from forestcheck.ml.data import (
    FeatureBinding,
    MLPlan,
    TargetBinding,
    extract_sample,
    freeze_subsets,
)
from pydantic import ValidationError
from test_evaluation_split import split


def plan(**updates):
    data = dict(
        schema_version="1.0",
        plan_version="ml-plan-1.0.0",
        seed=1701,
        table="invented",
        dataset_id=UUID(int=9),
        mapping_version="synthetic-1",
        mapping_sha256="a" * 64,
        split_sha256="b" * 64,
        snapshot_sha256="c" * 64,
        source_sha256="d" * 64,
        source_kind="real_observation",
        features=(
            FeatureBinding(
                schema_version="1.0",
                name="age",
                source_field="a",
                kind="numeric",
                unit="year",
                provenance="Synthetic measurement",
                dependency_status="unverified_measurement_procedure",
            ),
            FeatureBinding(
                schema_version="1.0",
                name="species",
                source_field="s",
                kind="categorical",
                unit=None,
                provenance="Synthetic category",
                dependency_status="unverified_measurement_procedure",
            ),
            FeatureBinding(
                schema_version="1.0",
                name="region",
                source_field="r",
                kind="context",
                unit=None,
                provenance="Synthetic context",
                dependency_status="unverified_measurement_procedure",
            ),
        ),
        targets=(
            TargetBinding(
                schema_version="1.0",
                name="height",
                source_field="h",
                unit="m",
                provenance="Synthetic measurement",
            ),
            TargetBinding(
                schema_version="1.0",
                name="diameter",
                source_field="d",
                unit="cm",
                provenance="Synthetic measurement",
            ),
        ),
        excluded_fields=("h_copy", "age_class"),
        exclusion_rationale="Copies and derived categories excluded",
        train_folds={"e" * 64: 0, "f" * 64: 1, "0" * 64: 2},
        calibration_fit_groups=("1" * 64,),
        calibration_check_groups=("2" * 64,),
        limitations=("Observations are not expert truth",),
    )
    return MLPlan(**(data | updates))


def record(height_state="observed", height=0):
    source = SourceCoordinates(artifact_id=UUID(int=8), table="invented", row=1, column="any")
    return Record(
        schema_version="1.0",
        id=UUID(int=2),
        dataset_id=UUID(int=9),
        business_key=BusinessKey(territory="01", compartment="003", stand="005"),
        fields=tuple(
            FieldValue(
                schema_version="1.0",
                name=n,
                original_value=v,
                normalized_value=v if state == "observed" else None,
                unit=u,
                state=state,
                source=source.model_copy(update={"column": n}),
            )
            for n, v, u, state in (
                ("a", 30, "year", "observed"),
                ("s", "001", None, "observed"),
                ("r", "01", None, "observed"),
                ("h", height, "m", height_state),
                ("d", 10, "cm", "observed"),
                ("h_copy", 999, "m", "observed"),
            )
        ),
    )


def test_extract_separates_target_and_context_preserves_zero():
    sample = extract_sample(record(), plan(), group_id="e" * 64, role="train")
    assert sample.features == {"age": 30, "species": "001", "region": "01"}
    assert sample.targets == {"height": 0, "diameter": 10}
    assert "h_copy" not in sample.features and "height" not in sample.features


@pytest.mark.parametrize("state", ["missing", "parse_error", "unknown", "not_applicable"])
def test_nonobserved_target_never_known_response(state):
    sample = extract_sample(record(state, "invalid"), plan(), group_id="e" * 64, role="train")
    assert sample.targets["height"] is None


def test_target_copy_as_feature_or_excluded_field_rejected():
    p = plan()
    for source in ("h", "h_copy"):
        features = (p.features[0].model_copy(update={"source_field": source}), *p.features[1:])
        with pytest.raises(ValidationError, match="target_or_excluded_source"):
            plan(features=features)
    with pytest.raises(ValidationError):
        plan(schema_version="2.0")


def test_holdout_schema_and_unit_refused():
    with pytest.raises(ValueError, match="holdout_forbidden"):
        extract_sample(record(), plan(), group_id="e" * 64, role="holdout")
    with pytest.raises(ValueError, match="dataset_mismatch"):
        extract_sample(
            record().model_copy(update={"dataset_id": UUID(int=999)}),
            plan(),
            group_id="e" * 64,
            role="train",
        )
    row = record()
    row = Record.model_validate(
        row.model_dump()
        | {
            "fields": [
                f.model_dump() | {"unit": "cm"} if f.name == "h" else f.model_dump()
                for f in row.fields
            ]
        }
    )
    with pytest.raises(ValueError, match="unit_mismatch"):
        extract_sample(row, plan(), group_id="e" * 64, role="train")


def test_subsets_only_train_and_calibration_before_values():
    manifest = split()
    folds, fit, check = freeze_subsets(manifest, seed=1701)
    assert set(folds.values()) == {0, 1, 2}
    assert set(folds) == {a.group_id for a in manifest.assignments if a.role == "train"}
    assert not set(fit) & set(check)
    assert set(fit) | set(check) == {
        a.group_id for a in manifest.assignments if a.role == "calibration"
    }
