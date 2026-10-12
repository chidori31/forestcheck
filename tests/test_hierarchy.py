"""Invented records only; expert clarification supplies applicability, not forestry ranges."""

from importlib import import_module, util
from uuid import UUID

import pytest
from forestcheck.domain.contracts import FieldValue, Record, SourceCoordinates
from pydantic import ValidationError

SOURCE = {
    "reference_id": "owner_supplied_expert_response",
    "locator": "answer 1",
    "kind": "explicit_configuration",
}
SYNTHETIC = {
    "reference_id": "invented_dictionary",
    "locator": "test definition",
    "kind": "synthetic",
}
RUN = UUID(int=99)


def api():
    assert util.find_spec("forestcheck.rules.hierarchy") is not None, "hierarchy checks missing"
    return import_module("forestcheck.rules.hierarchy")


def value(name, data, *, state="observed", unit=None):
    return FieldValue(
        schema_version="1.0",
        name=name,
        original_value=data,
        normalized_value=data if state == "observed" else None,
        state=state,
        unit=unit,
        source=SourceCoordinates(artifact_id=UUID(int=90), table="invented", row=2, column=name),
    )


def node(number, level="stand", parent=None, *, fields=None, link_status=None):
    record = Record(
        schema_version="1.0",
        id=UUID(int=number),
        dataset_id=UUID(int=91),
        fields=fields
        if fields is not None
        else (value("area", 0, unit="ha"), value("inventory", "0007"), value("land", "NAT")),
    )
    return {
        "record": record,
        "level": level,
        "parent_record_id": UUID(int=parent) if parent else None,
        "link_status": link_status or ("matched" if parent else "root"),
    }


def coverage(parent, child="layer", status="complete"):
    return {"parent_record_id": UUID(int=parent), "child_level": child, "status": status}


def snapshot(nodes, cover=()):
    return api().HierarchySnapshot(
        schema_version="1.0", nodes=nodes, coverage=cover, allow_synthetic=True
    )


def policy(**updates):
    payload = {
        "schema_version": "1.0",
        "id": "expert.structure",
        "version": "1",
        "fields": {"area": "area", "inventory_number": "inventory", "land_kind": "land"},
        "source": SOURCE,
        "land_categories": [
            {"value": "NAT", "natural_origin": True, "version": "invented-v1", "source": SYNTHETIC},
            {
                "value": "ROAD",
                "natural_origin": False,
                "version": "invented-v1",
                "source": SYNTHETIC,
            },
        ],
    }
    return api().HierarchyPolicy.model_validate(payload | updates)


def evaluate(snap, **updates):
    return api().evaluate_hierarchy(snap, policy(**updates), RUN)


def required(outcomes, role="area"):
    return next(o for o in outcomes if o.rule_id.endswith(".required." + role))


@pytest.mark.parametrize(
    ("fields", "status", "reason", "signal"),
    [
        ((), "violation", "required_field_absent", "missing"),
        ((value("area", "", state="missing"),), "violation", "required_value_blank", "missing"),
        ((value("area", "  "),), "violation", "required_value_blank", "missing"),
        (
            (value("area", "bad", state="parse_error"),),
            "violation",
            "required_value_parse_error",
            "parse_error",
        ),
        (
            (value("area", "?", state="unknown"),),
            "insufficient_data",
            "required_value_unknown",
            None,
        ),
        (
            (value("area", "-", state="not_applicable"),),
            "violation",
            "required_value_not_applicable",
            "inconsistency",
        ),
        ((value("area", 0, unit="ha"),), "passed", "required_value_observed", None),
    ],
)
def test_required_states_are_distinct_and_zero_is_present(fields, status, reason, signal):
    snap = snapshot([node(1, fields=fields)])
    before = snap.model_dump_json()
    outcome = required(evaluate(snap))
    assert (outcome.status, outcome.reason) == (status, reason)
    assert (outcome.finding.signal_kind if outcome.finding else None) == signal
    if outcome.finding:
        assert outcome.finding.field_name == "area"
        assert {e.code for e in outcome.finding.evidence} >= {
            "rule_source",
            "field_observation",
            "applicability",
        }
    assert snap.model_dump_json() == before


def test_all_three_stand_fields_required_but_not_on_regeneration():
    result = evaluate(snapshot([node(1, fields=()), node(2, "regeneration", 1, fields=())]))
    missing = [o for o in result if ".required." in o.rule_id]
    assert len(missing) == 3 and all(o.record_id == UUID(int=1) for o in missing)
    assert {o.finding.field_name for o in missing} == {"area", "inventory", "land"}


@pytest.mark.parametrize("state", [None, "missing_sheet", "incomplete", "unresolved_links"])
def test_missing_layer_cannot_be_proved_from_incomplete_or_unavailable_relation(state):
    snap = snapshot([node(1)], [] if state is None else [coverage(1, status=state)])
    out = next(o for o in evaluate(snap) if o.rule_id.endswith(".layers"))
    assert out.status == "insufficient_data" and out.finding is None


def test_natural_origin_layer_required_only_for_confirmed_category():
    snap = snapshot([node(1)], [coverage(1)])
    out = next(o for o in evaluate(snap) if o.rule_id.endswith(".layers"))
    assert out.status == "violation" and out.signal_class == "confirmed_contradiction"
    unknown = evaluate(snap, land_categories=[])
    assert next(o for o in unknown if o.rule_id.endswith(".layers")).status == "insufficient_data"
    road = snapshot([node(1, fields=(value("land", "ROAD"),))], [coverage(1)])
    assert (
        next(o for o in evaluate(road) if o.rule_id.endswith(".layers")).status == "not_applicable"
    )


def test_ambiguous_layer_join_is_not_missing_layer_even_if_coverage_claims_complete():
    snap = snapshot([node(1), node(2, "layer", 1, link_status="ambiguous_parent")], [coverage(1)])
    out = next(o for o in evaluate(snap) if o.rule_id.endswith(".layers"))
    assert out.status == "insufficient_data"


def test_ambiguous_parent_hint_cannot_prove_absence_for_other_candidate_parent():
    snap = snapshot(
        [node(1), node(10), node(2, "layer", 10, link_status="ambiguous_parent")],
        [coverage(1), coverage(10)],
    )
    layers = [o for o in evaluate(snap) if o.rule_id.endswith(".layers")]
    assert all(o.status == "insufficient_data" for o in layers)


def test_species_requirement_is_tentative_advisory_and_requires_complete_coverage():
    nodes = [node(1), node(2, "layer", 1), node(3, "regeneration", 1)]
    snap = snapshot(nodes, [coverage(1), coverage(2, "element")])
    out = next(o for o in evaluate(snap) if o.rule_id.endswith(".species"))
    assert out.record_id == UUID(int=2) and out.status == "violation"
    assert out.signal_class == "statistical_suspicion" and out.finding.signal_kind == "anomaly"
    assert "tentative" in out.reason
    incomplete = snapshot(nodes, [coverage(1), coverage(2, "element", "incomplete")])
    out = next(o for o in evaluate(incomplete) if o.rule_id.endswith(".species"))
    assert out.status == "insufficient_data" and out.finding is None


def test_element_multiplicity_is_preserved_and_regeneration_is_not_a_layer():
    nodes = [
        node(1),
        node(2, "layer", 1),
        node(3, "element", 2),
        node(4, "element", 2),
        node(5, "regeneration", 1),
    ]
    snap = snapshot(nodes, [coverage(1), coverage(2, "element")])
    original = snap.model_dump_json()
    assert all(o.status == "passed" for o in evaluate(snap))
    assert len(snap.nodes) == 5 and snap.model_dump_json() == original
    regen_only = snapshot([node(1), node(5, "regeneration", 1)], [coverage(1)])
    assert (
        next(o for o in evaluate(regen_only) if o.rule_id.endswith(".layers")).status == "violation"
    )


def quantity(record_id, **updates):
    return {
        "record_id": UUID(int=record_id),
        "field": "stock",
        "unit": "m3",
        "quantity_kind": "living",
        "basis": "total",
        "organism_group": "tree",
        "observation_scope": "invented-one-event",
        "definition_confirmed": True,
        "source": SYNTHETIC,
    } | updates


def stock_check(**updates):
    return api().StockSumCheck.model_validate(
        {
            "schema_version": "1.0",
            "id": "sum.stock",
            "version": "1",
            "total": quantity(1),
            "components": [quantity(2), quantity(3)],
            "component_level": "layer",
            "relationship_confirmed": True,
            "absolute_tolerance": "0.1",
            "tolerance_source": SYNTHETIC,
            "source": SYNTHETIC,
        }
        | updates
    )


def stock_snapshot(total=10, *, cover="complete", unit="m3"):
    return snapshot(
        [
            node(1, fields=(value("stock", total, unit="m3"),)),
            node(2, "layer", 1, fields=(value("stock", 4, unit=unit),)),
            node(3, "layer", 1, fields=(value("stock", 6, unit="m3"),)),
        ],
        [coverage(1, status=cover)],
    )


def stock(snap, **updates):
    return api().evaluate_stock_sum(snap, stock_check(**updates), RUN)


def test_complete_compatible_stock_sum_is_checked_once_with_tolerance_and_raw():
    snap = stock_snapshot(10.05)
    before = snap.model_dump_json()
    assert stock(snap).status == "passed"
    assert snap.model_dump_json() == before
    out = stock(stock_snapshot(11))
    assert out.status == "violation" and out.finding.field_name == "stock"
    assert out.signal_class == "confirmed_contradiction"
    assert "11" in next(e.message for e in out.finding.evidence if e.code == "field_observation")


@pytest.mark.parametrize(
    "changes",
    [
        {"unit": "m3/ha", "basis": "per_ha"},
        {"quantity_kind": "dead"},
        {"organism_group": "shrub"},
        {"organism_group": "unknown"},
        {"observation_scope": "another-year"},
        {"definition_confirmed": False},
    ],
)
def test_stock_incompatible_semantics_abstain(changes):
    out = stock(stock_snapshot(), components=[quantity(2, **changes), quantity(3)])
    assert out.status in {"insufficient_data", "unavailable_reference"} and out.finding is None


def test_stock_incomplete_population_units_and_missing_tolerance_abstain():
    assert stock(stock_snapshot(cover="incomplete")).status == "insufficient_data"
    assert stock(stock_snapshot(unit="m3/ha")).status == "insufficient_data"
    assert stock(stock_snapshot(), components=[quantity(2)]).status == "insufficient_data"
    assert (
        stock(stock_snapshot(), absolute_tolerance=None, tolerance_source=None).status
        == "unavailable_reference"
    )
    assert stock(stock_snapshot(), relationship_confirmed=False).status == "unavailable_reference"


def test_stock_definition_must_match_total_vs_per_hectare_unit():
    out = stock(
        stock_snapshot(),
        total=quantity(1, basis="per_ha"),
        components=[quantity(2, basis="per_ha"), quantity(3, basis="per_ha")],
    )
    assert out.status == "insufficient_data" and out.finding is None


def test_synthetic_or_heuristic_dictionary_cannot_enable_real_category_rule():
    real = api().HierarchySnapshot.model_validate(
        snapshot([node(1)], [coverage(1)]).model_dump() | {"allow_synthetic": False}
    )
    assert (
        next(o for o in evaluate(real) if o.rule_id.endswith(".layers")).status
        == "insufficient_data"
    )
    category = policy().land_categories[0].model_dump()
    category["source"] = dict(SOURCE, kind="heuristic")
    assert (
        next(
            o for o in evaluate(real, land_categories=[category]) if o.rule_id.endswith(".layers")
        ).status
        == "insufficient_data"
    )


def test_missing_parent_and_cross_snapshot_child_cannot_prove_structure():
    broken = node(2, "layer", 88)
    snap = snapshot([node(1), broken], [coverage(1)])
    assert (
        next(o for o in evaluate(snap) if o.rule_id.endswith(".layers")).status
        == "insufficient_data"
    )
    different = node(2, "layer", 1)
    different["record"] = different["record"].model_copy(update={"dataset_id": UUID(int=777)})
    snap = snapshot([node(1), different], [coverage(1)])
    assert (
        next(o for o in evaluate(snap) if o.rule_id.endswith(".layers")).status
        == "insufficient_data"
    )


def test_stock_parent_cannot_be_counted_as_component_and_repeated_ids_rejected():
    for components in ([quantity(2), quantity(2)], [quantity(1), quantity(2)]):
        with pytest.raises(ValidationError):
            stock_check(components=components)


def test_sidecars_reject_unknown_versions_fields_duplicate_nodes_and_categories():
    h = api()
    payload = snapshot([node(1)]).model_dump()
    for update in ({"schema_version": "2.0"}, {"invented": True}, {"nodes": [node(1), node(1)]}):
        with pytest.raises(ValidationError):
            h.HierarchySnapshot.model_validate(payload | update)
    with pytest.raises(ValidationError):
        policy(fields={"area": "area", "inventory_number": "area", "land_kind": "land"})
    with pytest.raises(ValidationError):
        policy(land_categories=[policy().land_categories[0]] * 2)
    with pytest.raises(ValidationError):
        h.StockSumCheck.model_validate(stock_check().model_dump() | {"schema_version": "2.0"})


def test_category_subtype_binding_does_not_replace_required_broad_land_kind():
    fields = (
        value("area", 1, unit="ha"),
        value("inventory", "0007"),
        value("land", "NONFOREST"),
        value("forest_subtype", None, state="missing"),
    )
    snap = snapshot([node(1, fields=fields)], [coverage(1)])
    outcomes = evaluate(snap, category_field="forest_subtype")
    assert required(outcomes, "land_kind").status == "passed"
    assert next(o for o in outcomes if o.rule_id.endswith(".layers")).status == "insufficient_data"
    fields = (*fields[:2], value("land", "FOREST"), value("forest_subtype", "NAT"))
    snap = snapshot([node(1, fields=fields)], [coverage(1)])
    outcome = next(
        o for o in evaluate(snap, category_field="forest_subtype") if o.rule_id.endswith(".layers")
    )
    assert outcome.status == "violation" and outcome.finding.field_name == "forest_subtype"


def test_invalid_stand_root_is_rejected_before_absence_claim():
    with pytest.raises(ValidationError, match="stand_root"):
        snapshot([node(1, parent=777, link_status="missing_parent")], [coverage(1)])


def test_cross_snapshot_relation_for_another_parent_blocks_absence_claim():
    child = node(3, "layer", 2)
    child["record"] = child["record"].model_copy(update={"dataset_id": UUID(int=777)})
    snap = snapshot([node(1), node(2), child], [coverage(1), coverage(2)])
    outcomes = [o for o in evaluate(snap) if o.rule_id.endswith(".layers")]
    assert all(o.status == "insufficient_data" for o in outcomes)
