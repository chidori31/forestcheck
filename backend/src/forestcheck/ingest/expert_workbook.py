"""Build explicit hierarchical sidecars without flattening or changing physical records."""

from collections import defaultdict
from typing import Literal, Self
from uuid import UUID, uuid5

from pydantic import StrictBool, model_validator

from forestcheck.domain.contracts import Contract, Record, Text, ValueObject
from forestcheck.rules.hierarchy import HierarchyNode, HierarchySnapshot, RelationCoverage
from forestcheck.rules.models import RuleSource


class WorkbookHierarchyBinding(Contract):
    version: Text
    root_sheet: Text
    element_sheet: Text
    regeneration_sheet: Text | None = None
    layer_id_field: Text
    layer_fields: tuple[Text, ...]
    element_identity_field: Text
    source: RuleSource
    sheet_completeness: dict[Text, StrictBool] = {}

    @model_validator(mode="after")
    def explicit_distinct_bindings(self) -> Self:
        sheets = [self.root_sheet, self.element_sheet]
        if self.regeneration_sheet is not None:
            sheets.append(self.regeneration_sheet)
        if len(set(sheets)) != len(sheets):
            raise ValueError("hierarchy_sheets_must_be_distinct")
        if len(set(self.layer_fields)) != len(self.layer_fields):
            raise ValueError("duplicate_layer_attribute_binding")
        if not set(self.sheet_completeness) <= set(sheets):
            raise ValueError("completeness_for_unbound_sheet")
        return self


class _Relation(ValueObject):
    child_record_id: UUID
    child_sheet: Text
    parent_sheet: Text
    parent_record_id: UUID | None
    status: Literal["matched", "missing_parent", "ambiguous_parent"]


def _observed_identifier(record, name):
    field = next((f for f in record.fields if f.name == name), None)
    if field is None or field.state != "observed":
        return None
    value = field.normalized_value
    # Identifier parsing belongs to the positional mapping; never stringify a numeric guess.
    return value if isinstance(value, str) and value.strip() else None


def _attributes(records, names):
    """Ignore coordinates for equality, retain raw/state/unit, then select stable provenance."""
    selected = []
    for name in names:
        values = [next((f for f in r.fields if f.name == name), None) for r in records]
        if any(f is None for f in values):
            return (), "incomplete"
        representations = [f.model_dump_json(exclude={"source", "schema_version"}) for f in values]
        if any(v != representations[0] for v in representations[1:]):
            return (), "unresolved_links"
        selected.append(values[0])  # records already sorted; equality was checked on every row.
    return tuple(selected), "complete"


def build_hierarchy(
    records_by_sheet: dict[str, tuple[Record, ...]],
    relations: tuple[dict, ...],
    binding: WorkbookHierarchyBinding,
) -> HierarchySnapshot:
    """Adapt the old workbook sidecars; completeness is an explicit external assertion.

    Root absence is a loading failure, not an empty successful hierarchy. Physical rows
    remain unchanged. Virtual layer attributes are admitted only after group consistency.
    """
    if binding.root_sheet not in records_by_sheet:
        raise ValueError("root_sheet_missing")
    physical = [r for rows in records_by_sheet.values() for r in rows]
    if len({r.id for r in physical}) != len(physical):
        raise ValueError("duplicate_physical_record")
    for sheet, records in records_by_sheet.items():
        if any(f.source.table != sheet for r in records for f in r.fields):
            raise ValueError("record_source_sheet_mismatch")
    roots = {r.id: r for r in records_by_sheet[binding.root_sheet]}
    by_child = defaultdict(list)
    for payload in relations:
        relation = _Relation.model_validate(payload)
        by_child[relation.child_record_id].append(relation)
    nodes = [HierarchyNode(record=r, level="stand", link_status="root") for r in roots.values()]
    coverage = {}

    def initial(sheet):
        if binding.source.kind == "heuristic":
            return "unresolved_links"
        if sheet not in records_by_sheet:
            return "missing_sheet"
        if not binding.sheet_completeness.get(sheet, False) or not binding.sheet_completeness.get(
            binding.root_sheet, False
        ):
            return "incomplete"
        return "complete"

    def degrade(parent_id, level, status):
        key = (parent_id, level)
        ranking = {"complete": 0, "incomplete": 1, "missing_sheet": 2, "unresolved_links": 3}
        if ranking[status] > ranking[coverage[key]]:
            coverage[key] = status

    def parent_of(record, sheet):
        links = by_child[record.id]
        if binding.source.kind == "heuristic" or len(links) != 1:
            return None
        link = links[0]
        parent = roots.get(link.parent_record_id)
        if (
            link.status != "matched"
            or parent is None
            or link.child_sheet != sheet
            or link.parent_sheet != binding.root_sheet
            or (record.dataset_id, record.dataset_revision)
            != (parent.dataset_id, parent.dataset_revision)
        ):
            return None
        return parent

    for parent_id in roots:
        coverage[(parent_id, "layer")] = initial(binding.element_sheet)
        if binding.regeneration_sheet is not None:
            coverage[(parent_id, "regeneration")] = initial(binding.regeneration_sheet)

    expected_levels = {binding.element_sheet: "layer"}
    if binding.regeneration_sheet is not None:
        expected_levels[binding.regeneration_sheet] = "regeneration"
    loaded_ids = {sheet: {r.id for r in records} for sheet, records in records_by_sheet.items()}
    for links in by_child.values():
        for relation in links:
            if (
                relation.child_sheet in expected_levels
                and relation.child_record_id not in loaded_ids.get(relation.child_sheet, set())
            ):
                for parent_id in roots:
                    degrade(parent_id, expected_levels[relation.child_sheet], "incomplete")

    groups = defaultdict(list)
    for record in records_by_sheet.get(binding.element_sheet, ()):
        parent = parent_of(record, binding.element_sheet)
        if parent is None:
            nodes.append(HierarchyNode(record=record, level="other", link_status="unresolved"))
            for parent_id in roots:
                degrade(parent_id, "layer", "unresolved_links")
            continue
        layer_id = _observed_identifier(record, binding.layer_id_field)
        if layer_id is None:
            nodes.append(
                HierarchyNode(
                    record=record, level="other", parent_record_id=parent.id, link_status="matched"
                )
            )
            degrade(parent.id, "layer", "incomplete")
            continue
        groups[(parent.id, layer_id)].append(record)

    for (parent_id, layer_id), records in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        parent = roots[parent_id]
        records.sort(key=lambda r: str(r.id))
        fields, attributes_status = _attributes(records, binding.layer_fields)
        layer_uuid = uuid5(parent_id, f"{binding.version}:layer:{layer_id}")
        layer = Record(
            schema_version="1.0",
            id=layer_uuid,
            revision=parent.revision,
            dataset_id=parent.dataset_id,
            dataset_revision=parent.dataset_revision,
            fields=fields,
        )
        nodes.append(
            HierarchyNode(
                record=layer, level="layer", parent_record_id=parent_id, link_status="matched"
            )
        )
        coverage[(layer_uuid, "element")] = initial(binding.element_sheet)
        degrade(parent_id, "layer", attributes_status)
        degrade(layer_uuid, "element", attributes_status)
        for record in records:
            identified = _observed_identifier(record, binding.element_identity_field) is not None
            nodes.append(
                HierarchyNode(
                    record=record,
                    level="element" if identified else "other",
                    parent_record_id=layer_uuid,
                    link_status="matched",
                )
            )
            if not identified:
                degrade(layer_uuid, "element", "incomplete")

    if binding.regeneration_sheet is not None:
        for record in records_by_sheet.get(binding.regeneration_sheet, ()):
            parent = parent_of(record, binding.regeneration_sheet)
            nodes.append(
                HierarchyNode(
                    record=record,
                    level="regeneration",
                    parent_record_id=parent.id if parent else None,
                    link_status="matched" if parent else "unresolved",
                )
            )
            if parent is None:
                for parent_id in roots:
                    degrade(parent_id, "regeneration", "unresolved_links")

    bound_sheets = {binding.root_sheet, binding.element_sheet, binding.regeneration_sheet}
    for sheet, records in records_by_sheet.items():
        if sheet not in bound_sheets:
            nodes.extend(
                HierarchyNode(record=r, level="other", link_status="unresolved") for r in records
            )
    return HierarchySnapshot(
        schema_version="1.0",
        nodes=tuple(sorted(nodes, key=lambda n: str(n.record.id))),
        coverage=tuple(
            RelationCoverage(parent_record_id=parent_id, child_level=level, status=status)
            for (parent_id, level), status in sorted(
                coverage.items(), key=lambda item: (str(item[0][0]), item[0][1])
            )
        ),
        allow_synthetic=binding.source.kind == "synthetic",
    )
