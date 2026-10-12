"""Synthetic adapter inputs, independent of competition workbook schemas."""

from importlib import import_module, util
from uuid import UUID, uuid5

import pytest
from forestcheck.domain.contracts import FieldValue, Record, SourceCoordinates
from pydantic import ValidationError


def api():
    assert util.find_spec("forestcheck.ingest.expert_workbook") is not None, "adapter missing"
    return import_module("forestcheck.ingest.expert_workbook")


def record(number, sheet, layer="L1", species="AA", layer_stock=10):
    values = (
        ("layer_id", layer, None),
        ("species", species, None),
        ("layer_stock", layer_stock, "m3"),
        ("height_a", 12, "m"),
        ("height_b", 1.0, "m"),
    )
    return Record(
        schema_version="1.0",
        id=UUID(int=number),
        dataset_id=UUID(int=100),
        fields=tuple(
            FieldValue(
                schema_version="1.0",
                name=name,
                original_value=value,
                normalized_value=value,
                unit=unit,
                state="missing" if value is None else "observed",
                source=SourceCoordinates(
                    artifact_id=UUID(int=200),
                    table=sheet,
                    row=number + 1,
                    column=f"column_{position:04d}",
                ),
            )
            for position, (name, value, unit) in enumerate(values, 1)
        ),
    )


def binding(**updates):
    return api().WorkbookHierarchyBinding.model_validate(
        {
            "schema_version": "1.0",
            "version": "test-1",
            "root_sheet": "stands",
            "element_sheet": "elements",
            "regeneration_sheet": "regen",
            "layer_id_field": "layer_id",
            "layer_fields": ["layer_stock"],
            "element_identity_field": "species",
            "source": {
                "reference_id": "invented-binding",
                "locator": "Synthetic definition",
                "kind": "synthetic",
            },
            "sheet_completeness": {"stands": True, "elements": True, "regen": True},
        }
        | updates
    )


def link(number, parent=1, sheet="elements", status="matched"):
    return {
        "child_record_id": str(UUID(int=number)),
        "child_sheet": sheet,
        "parent_sheet": "stands",
        "parent_record_id": str(UUID(int=parent)) if parent else None,
        "status": status,
    }


def build(elements=(), *, relations=None, overrides=None, **binding_changes):
    records = {"stands": (record(1, "stands"),), "elements": tuple(elements), "regen": ()}
    records.update(overrides or {})
    links = tuple(link(r.id.int) for r in elements) if relations is None else relations
    return api().build_hierarchy(records, links, binding(**binding_changes))


def state(snapshot, parent=1, child="layer"):
    return next(
        c.status
        for c in snapshot.coverage
        if c.parent_record_id == UUID(int=parent) and c.child_level == child
    )


def test_stable_virtual_layer_preserves_physical_multiplicity_raw_and_both_heights():
    a, b = record(2, "elements"), record(3, "elements")
    before = (a.model_dump_json(), b.model_dump_json())
    first, second = build((a, b)), build((b, a))
    assert first == second
    layers = [n for n in first.nodes if n.level == "layer"]
    assert len(layers) == 1
    assert layers[0].record.id == uuid5(UUID(int=1), "test-1:layer:L1")
    assert [f.name for f in layers[0].record.fields] == ["layer_stock"]
    elements = [n for n in first.nodes if n.level == "element"]
    assert {n.record.id for n in elements} == {a.id, b.id}
    assert all(n.parent_record_id == layers[0].record.id for n in elements)
    assert state(first) == "complete"
    assert state(first, layers[0].record.id.int, "element") == "complete"
    assert (a.model_dump_json(), b.model_dump_json()) == before
    assert [(f.name, f.normalized_value) for f in elements[0].record.fields][-2:] == [
        ("height_a", 12),
        ("height_b", 1.0),
    ]


def test_same_layer_id_under_different_parents_creates_distinct_layers():
    a, b = record(2, "elements"), record(3, "elements")
    snap = build(
        (a, b),
        relations=(link(2), link(3, 4)),
        overrides={"stands": (record(1, "stands"), record(4, "stands"))},
    )
    assert len([n for n in snap.nodes if n.level == "layer"]) == 2


def test_empty_complete_element_sheet_can_prove_no_layer_but_missing_sheet_cannot():
    assert state(build()) == "complete"
    records = {"stands": (record(1, "stands"),), "regen": ()}
    snap = api().build_hierarchy(records, (), binding())
    assert state(snap) == "missing_sheet"
    with pytest.raises(ValueError, match="root_sheet_missing"):
        api().build_hierarchy({"elements": ()}, (), binding())


def test_missing_species_identity_preserves_row_as_other_and_blocks_empty_species_claim():
    row = record(2, "elements", species=None)
    snap = build((row,))
    layer = next(n for n in snap.nodes if n.level == "layer")
    physical = next(n for n in snap.nodes if n.record.id == row.id)
    assert physical.level == "other" and physical.record == row
    assert physical.parent_record_id == layer.record.id
    assert state(snap, layer.record.id.int, "element") == "incomplete"


@pytest.mark.parametrize("layer", [None, "", 1])
def test_unconfirmed_layer_id_does_not_invent_a_layer(layer):
    row = record(2, "elements", layer=layer)
    snap = build((row,))
    assert not any(n.level == "layer" for n in snap.nodes)
    assert any(n.record == row for n in snap.nodes)
    assert state(snap) == "incomplete"


def test_explicit_sheet_completeness_is_required_even_after_successful_read():
    snap = build((record(2, "elements"),), sheet_completeness={})
    assert state(snap) == "incomplete"
    layer = next(n for n in snap.nodes if n.level == "layer")
    assert state(snap, layer.record.id.int, "element") == "incomplete"


@pytest.mark.parametrize(
    "relations",
    [(link(2, None, status="ambiguous_parent"),), (link(2), link(2)), (), (link(2, 99),)],
)
def test_ambiguous_missing_or_unresolved_join_preserves_row_and_blocks_coverage(relations):
    row = record(2, "elements")
    snap = build((row,), relations=relations)
    assert not any(n.level == "layer" for n in snap.nodes)
    assert any(n.record == row for n in snap.nodes)
    assert state(snap) == "unresolved_links"


def test_contradictory_repeated_layer_attributes_are_not_taken_from_first_row():
    a, b = record(2, "elements", layer_stock=10), record(3, "elements", layer_stock=20)
    snap = build((a, b))
    layer = next(n for n in snap.nodes if n.level == "layer")
    assert layer.record.fields == ()
    assert state(snap) == "unresolved_links"
    assert state(snap, layer.record.id.int, "element") == "unresolved_links"
    assert {n.record.id for n in snap.nodes if n.level == "element"} == {a.id, b.id}


def test_regeneration_and_unmapped_rows_remain_separate():
    regen, other = record(4, "regen"), record(5, "extra")
    snap = build(
        (), relations=(link(4, sheet="regen"),), overrides={"regen": (regen,), "extra": (other,)}
    )
    assert next(n for n in snap.nodes if n.record.id == regen.id).level == "regeneration"
    assert next(n for n in snap.nodes if n.record.id == other.id).level == "other"
    assert state(snap, child="regeneration") == "complete"


def test_binding_and_input_identity_are_strict():
    for changes in (
        {"schema_version": "2.0"},
        {"unknown": True},
        {"sheet_completeness": {"elements": 1}},
        {"element_sheet": "stands"},
        {"layer_fields": ["height_a", "height_a"]},
    ):
        with pytest.raises(ValidationError):
            binding(**changes)
    row = record(2, "elements")
    with pytest.raises(ValueError, match="duplicate_physical_record"):
        build((row, row))


def test_unconfirmed_binding_cannot_prove_absence_from_empty_export():
    source = {"reference_id": "guess", "locator": "unconfirmed", "kind": "heuristic"}
    assert state(build(source=source)) == "unresolved_links"


def test_relation_to_unloaded_physical_row_blocks_empty_population_claim():
    assert state(build(relations=(link(99),))) == "incomplete"


def test_layer_attributes_compare_scalar_types_without_boolean_numeric_coercion():
    a, b = record(2, "elements", layer_stock=1), record(3, "elements", layer_stock=True)
    snap = build((a, b))
    layer = next(n for n in snap.nodes if n.level == "layer")
    assert layer.record.fields == () and state(snap) == "unresolved_links"
