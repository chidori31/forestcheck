"""Invented growth fixtures verify mechanics, never forestry norms or real tables."""

from copy import deepcopy

import pytest
from forestcheck.growth import (
    GrowthQuery,
    GrowthTable,
    GrowthTableRegistry,
)
from pydantic import ValidationError


def table_payload(**metadata_changes):
    metadata = {
        "schema_version": "1.0",
        "table_id": "synthetic-growth",
        "table_version": "fixture-1",
        "source": {
            "source_id": "invented-reference",
            "title": "Invented fixture, not a forestry reference",
            "section": "synthetic-section",
            "page": "1",
        },
        "region": "synthetic-region-a",
        "species": "synthetic-species",
        "origin": "synthetic-origin",
        "stand_type": "synthetic-stand",
        "bonitet": "synthetic-class",
        "conditions": "synthetic-conditions",
        "indicator": "synthetic-height",
        "indicator_meaning": "Height of invented test objects",
        "unit": "synthetic-length",
        "age_range": {"minimum": 10, "maximum": 30, "unit": "synthetic-age"},
        "interpolation": {"method": "none", "authorization": None},
        "synthetic": True,
    }
    metadata.update(metadata_changes)
    return {
        "schema_version": "1.0",
        "metadata": metadata,
        "points": [{"age": 10, "value": 2}, {"age": 30, "value": 8}],
    }


def query(**changes):
    payload = {
        "schema_version": "1.0",
        "region": "synthetic-region-a",
        "species": "synthetic-species",
        "origin": "synthetic-origin",
        "stand_type": "synthetic-stand",
        "bonitet": "synthetic-class",
        "conditions": "synthetic-conditions",
        "indicator": "synthetic-height",
        "unit": "synthetic-length",
        "age_unit": "synthetic-age",
        "age": 10,
    }
    payload.update(changes)
    return GrowthQuery.model_validate(payload)


def registry(**metadata_changes):
    return GrowthTableRegistry((GrowthTable.model_validate(table_payload(**metadata_changes)),))


@pytest.mark.parametrize("location", ["table", "metadata", "query"])
@pytest.mark.parametrize("version", ["2.0", "1.1", 1, None])
def test_contracts_reject_incompatible_versions(location, version):
    if location == "query":
        with pytest.raises(ValidationError):
            query(schema_version=version)
        return
    payload = table_payload()
    target = payload if location == "table" else payload["metadata"]
    target["schema_version"] = version
    with pytest.raises(ValidationError):
        GrowthTable.model_validate(payload)


@pytest.mark.parametrize("location", ["table", "metadata", "source", "range", "policy", "point"])
def test_unknown_fields_are_rejected_at_every_table_boundary(location):
    payload = table_payload()
    targets = {
        "table": payload,
        "metadata": payload["metadata"],
        "source": payload["metadata"]["source"],
        "range": payload["metadata"]["age_range"],
        "policy": payload["metadata"]["interpolation"],
        "point": payload["points"][0],
    }
    targets[location]["future_field"] = "unsupported"
    with pytest.raises(ValidationError):
        GrowthTable.model_validate(payload)


def test_contracts_preserve_roundtrip_and_freeze_nested_provenance():
    table = GrowthTable.model_validate(table_payload())
    assert GrowthTable.model_validate_json(table.model_dump_json()) == table
    with pytest.raises(ValidationError):
        table.metadata.region = "other"
    with pytest.raises(ValidationError):
        table.metadata.source.page = "other"
    with pytest.raises(ValidationError):
        query(stocking=0.5)
    with pytest.raises(ValidationError):
        query(composition=0.5)


@pytest.mark.parametrize("field", ["schema_version", "source", "synthetic", "interpolation"])
def test_required_metadata_is_not_inferred(field):
    payload = table_payload()
    del payload["metadata"][field]
    with pytest.raises(ValidationError):
        GrowthTable.model_validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"bonitet": None, "conditions": None},
        {"region": " "},
        {"synthetic": "false"},
        {"indicator_meaning": ""},
    ],
)
def test_unconfirmed_metadata_is_rejected(changes):
    with pytest.raises(ValidationError):
        GrowthTable.model_validate(table_payload(**changes))


def test_source_needs_section_or_page():
    payload = table_payload()
    payload["metadata"]["source"].update(section=None, page=None)
    with pytest.raises(ValidationError):
        GrowthTable.model_validate(payload)


@pytest.mark.parametrize(
    "bad_points",
    [
        [],
        [{"age": 10, "value": 2}, {"age": 10, "value": 8}],
        [{"age": 30, "value": 8}, {"age": 10, "value": 2}],
        [{"age": 10, "value": 2}, {"age": 40, "value": 8}],
        [{"age": 10, "value": 2}, {"age": 20, "value": 8}],
        [{"age": "10", "value": 2}, {"age": 30, "value": 8}],
        [{"age": True, "value": 2}, {"age": 30, "value": 8}],
        [{"age": 10, "value": float("nan")}, {"age": 30, "value": 8}],
    ],
)
def test_unreliable_point_ranges_are_rejected(bad_points):
    payload = table_payload()
    payload["points"] = bad_points
    with pytest.raises(ValidationError):
        GrowthTable.model_validate(payload)


def test_no_reference_does_not_claim_a_successful_comparison():
    result = GrowthTableRegistry().lookup(query())
    assert result.status == "unavailable_reference"
    assert result.reason == "no_registered_reference"
    assert result.value is None and result.metadata is None


def test_empty_registry_reports_missing_reference_before_missing_query_dimensions():
    result = GrowthTableRegistry().lookup(query(region=None, age=None, bonitet=None))
    assert result.status == "unavailable_reference"
    assert result.reason == "no_registered_reference"
    assert set(result.missing_fields) == {"region", "age"}
    assert result.value is None and result.metadata is None


def test_synthetic_reference_requires_explicit_test_flag():
    result = registry().lookup(query())
    assert result.status == "unavailable_reference"
    assert any(item.field == "synthetic" for item in result.mismatches)
    result = registry().lookup(query(), allow_synthetic=True)
    assert result.status == "available" and result.value == 2
    assert result.metadata.synthetic is True
    assert result.method == "exact" and result.unit == "synthetic-length"
    assert result.metadata.source.page == "1"
    assert len(result.supporting_points) == 1
    with pytest.raises(ValidationError):
        result.value = 10


@pytest.mark.parametrize(
    "field",
    [
        "region",
        "species",
        "origin",
        "stand_type",
        "bonitet",
        "conditions",
        "indicator",
        "unit",
        "age_unit",
    ],
)
def test_applicability_requires_exact_match_without_nearest_fallback(field):
    result = registry().lookup(query(**{field: "other"}), allow_synthetic=True)
    assert result.status == "unavailable_reference" and result.value is None
    mismatch = next(item for item in result.mismatches if item.field == field)
    assert mismatch.actual == "other" and mismatch.expected != "other"


@pytest.mark.parametrize(
    "field",
    [
        "region",
        "species",
        "origin",
        "stand_type",
        "bonitet",
        "conditions",
        "indicator",
        "unit",
        "age_unit",
        "age",
    ],
)
def test_missing_applicability_is_insufficient_data(field):
    result = registry().lookup(query(**{field: None}), allow_synthetic=True)
    assert result.status == "insufficient_data" and result.value is None
    assert field in result.missing_fields


def test_bonitet_or_conditions_can_be_explicitly_not_used_by_reference():
    result = registry(bonitet=None).lookup(query(bonitet=None), allow_synthetic=True)
    assert result.status == "available"
    result = registry(conditions=None).lookup(query(conditions=None), allow_synthetic=True)
    assert result.status == "available"
    result = registry().lookup(query(bonitet=None, conditions=None), allow_synthetic=True)
    assert result.status == "insufficient_data"


def test_ambiguous_matches_are_never_selected_by_insertion_order():
    first = GrowthTable.model_validate(table_payload())
    second = GrowthTable.model_validate(table_payload(table_id="second-synthetic-table"))
    for tables in [(first, second), (second, first)]:
        result = GrowthTableRegistry(tables).lookup(query(), allow_synthetic=True)
        assert result.status == "unavailable_reference"
        assert result.reason == "ambiguous_reference" and result.value is None
        assert set(result.candidate_ids) == {"synthetic-growth", "second-synthetic-table"}


def test_duplicate_table_identity_is_rejected():
    table = GrowthTable.model_validate(table_payload())
    with pytest.raises(ValueError, match="duplicate_table_version"):
        GrowthTableRegistry((table, table))


@pytest.mark.parametrize("age", [0, 9.9, 30.1, 100])
def test_extrapolation_is_unavailable_even_when_interpolation_allowed(age):
    result = registry().lookup(query(age=age, allow_interpolation=True), allow_synthetic=True)
    assert result.status == "unavailable_reference" and result.value is None
    assert any(item.field == "age" for item in result.mismatches)


def test_interpolation_requires_both_reference_authority_and_query_opt_in():
    result = registry().lookup(query(age=20, allow_interpolation=True), allow_synthetic=True)
    assert result.status == "unavailable_reference" and result.value is None
    policy = {"method": "linear", "authorization": table_payload()["metadata"]["source"]}
    result = registry(interpolation=policy).lookup(query(age=20), allow_synthetic=True)
    assert result.status == "unavailable_reference"
    result = registry(interpolation=policy).lookup(
        query(age=20, allow_interpolation=True), allow_synthetic=True
    )
    assert result.status == "available" and result.value == 5
    assert result.method == "linear" and len(result.supporting_points) == 2


def test_linear_policy_without_authority_is_rejected():
    with pytest.raises(ValidationError):
        GrowthTable.model_validate(
            table_payload(
                interpolation={
                    "method": "linear",
                    "authorization": None,
                }
            )
        )


def test_lookup_does_not_modify_input_or_apply_scaling():
    payload = table_payload()
    snapshot = deepcopy(payload)
    table = GrowthTable.model_validate(payload)
    result = GrowthTableRegistry((table,)).lookup(query(age=30), allow_synthetic=True)
    assert result.value == 8
    assert payload == snapshot and table.model_dump(mode="json") == snapshot


@pytest.mark.parametrize("value", ["10", True, -1, float("inf")])
def test_query_rejects_invalid_age(value):
    with pytest.raises(ValidationError):
        query(age=value)


def test_synthetic_flag_must_be_literal_boolean():
    with pytest.raises(TypeError, match="allow_synthetic_must_be_bool"):
        registry().lookup(query(), allow_synthetic="false")


def test_missing_reference_dimension_precedes_interpolation_decision():
    result = registry().lookup(query(bonitet=None, age=20), allow_synthetic=True)
    assert result.status == "insufficient_data"
    assert "bonitet" in result.missing_fields


def test_zero_age_and_value_are_valid_exact_reference_points():
    payload = table_payload(age_range={"minimum": 0, "maximum": 0, "unit": "synthetic-age"})
    payload["points"] = [{"age": 0, "value": 0}]
    table = GrowthTable.model_validate(payload)
    result = GrowthTableRegistry((table,)).lookup(query(age=0), allow_synthetic=True)
    assert result.status == "available" and result.value == 0 and result.method == "exact"
