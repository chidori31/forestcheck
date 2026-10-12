"""Explicit hierarchy sidecars and expert checks; never infer joins or forestry codes."""

import json
from typing import Annotated, Literal, Self
from uuid import UUID, uuid5

from pydantic import Field, StrictBool, model_validator

from forestcheck.domain.contracts import Contract, Evidence, Finding, Record, Text, ValueObject
from forestcheck.repair.engine import _aggregate
from forestcheck.repair.models import AggregateRelation
from forestcheck.rules.models import DecimalText, EvaluationContext, RuleOutcome, RuleSource

Level = Literal["stand", "layer", "element", "regeneration", "other"]


class HierarchyNode(ValueObject):
    record: Record
    level: Level
    parent_record_id: UUID | None = None
    link_status: Literal["root", "matched", "missing_parent", "ambiguous_parent", "unresolved"]


class RelationCoverage(ValueObject):
    parent_record_id: UUID
    child_level: Level
    status: Literal["complete", "missing_sheet", "incomplete", "unresolved_links"]


class HierarchySnapshot(Contract):
    nodes: tuple[HierarchyNode, ...]
    coverage: tuple[RelationCoverage, ...] = ()
    allow_synthetic: StrictBool = False

    @model_validator(mode="after")
    def unambiguous_ids(self) -> Self:
        if any(
            n.level == "stand" and (n.parent_record_id is not None or n.link_status != "root")
            for n in self.nodes
        ):
            raise ValueError("stand_root_requires_no_parent_and_root_status")
        ids = [n.record.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate_technical_record_id")
        scopes = [(c.parent_record_id, c.child_level) for c in self.coverage]
        if len(set(scopes)) != len(scopes):
            raise ValueError("duplicate_relation_coverage")
        if any(c.parent_record_id not in ids for c in self.coverage):
            raise ValueError("coverage_parent_unavailable")
        return self


class StandFieldBindings(ValueObject):
    area: Text
    inventory_number: Text
    land_kind: Text

    @model_validator(mode="after")
    def distinct_fields(self) -> Self:
        if len(set(self.model_dump().values())) != 3:
            raise ValueError("distinct_required_field_bindings")
        return self


class LandCategoryBinding(ValueObject):
    """A caller-verified dictionary value, not an inferred meaning of a code."""

    value: Text
    natural_origin: StrictBool
    version: Text
    source: RuleSource


class HierarchyPolicy(Contract):
    id: Text
    version: Text
    fields: StandFieldBindings
    category_field: Text | None = None
    source: RuleSource
    land_categories: tuple[LandCategoryBinding, ...] = ()

    @model_validator(mode="after")
    def unique_categories(self) -> Self:
        if len({c.value for c in self.land_categories}) != len(self.land_categories):
            raise ValueError("duplicate_category_binding")
        return self


class StockQuantityBinding(ValueObject):
    record_id: UUID
    field: Text
    unit: Text
    quantity_kind: Text
    basis: Literal["per_ha", "total"]
    organism_group: Literal["tree", "shrub", "unknown"]
    observation_scope: Text
    definition_confirmed: StrictBool
    source: RuleSource


class StockSumCheck(Contract):
    id: Text
    version: Text
    total: StockQuantityBinding
    components: Annotated[tuple[StockQuantityBinding, ...], Field(min_length=1)]
    component_level: Literal["layer", "element"]
    relationship_confirmed: StrictBool
    absolute_tolerance: DecimalText | None = None
    tolerance_source: RuleSource | None = None
    source: RuleSource

    @model_validator(mode="after")
    def distinct_population(self) -> Self:
        ids = [c.record_id for c in self.components]
        if len(set(ids)) != len(ids) or self.total.record_id in ids:
            raise ValueError("aggregate_duplicate_or_parent_component")
        if len({c.field for c in self.components}) != 1:
            raise ValueError("aggregate_component_field_must_be_uniform")
        if (self.absolute_tolerance is None) != (self.tolerance_source is None):
            raise ValueError("aggregate_tolerance_requires_source")
        return self


def _field(record, name):
    return next((f for f in record.fields if f.name == name), None)


def _outcome(
    record,
    run_id,
    rule_id,
    version,
    source,
    status,
    reason,
    *,
    field=None,
    signal="inconsistency",
    classification="confirmed_contradiction",
    context="",
    extra=(),
):
    finding = None
    if status == "violation":
        value = _field(record, field) if field else None
        if source.kind == "heuristic":
            classification, signal = "statistical_suspicion", "anomaly"
        observation = {
            "field": field,
            "present": value is not None,
            "original_value": value.original_value if value else None,
            "state": value.state if value else None,
            "unit": value.unit if value else None,
        }
        finding = Finding(
            schema_version="1.0",
            id=uuid5(run_id, f"{record.id}:{record.revision}:{rule_id}@{version}"),
            record_id=record.id,
            record_revision=record.revision,
            analysis_run_id=run_id,
            field_name=field,
            signal_kind=signal,
            severity="warning" if classification == "statistical_suspicion" else "error",
            priority=2,
            review_status="unreviewed",
            evidence=(
                Evidence(
                    code=reason,
                    message=reason,
                    reference_id=f"{rule_id}@{version}",
                    source=value.source if value else None,
                ),
                Evidence(
                    code="rule_source",
                    message=f"{source.reference_id}; {source.locator}",
                    reference_id=source.reference_id,
                ),
                Evidence(
                    code="field_observation", message=json.dumps(observation, ensure_ascii=False)
                ),
                Evidence(code="applicability", message=context),
                *extra,
            ),
        )
    return RuleOutcome(
        schema_version="1.0",
        record_id=record.id,
        rule_id=rule_id,
        rule_version=version,
        status=status,
        reason=reason,
        finding=finding,
        signal_class=classification if finding else None,
    )


def _source_available(source, snapshot):
    return source.kind != "heuristic" and (source.kind != "synthetic" or snapshot.allow_synthetic)


def _children(snapshot, parent_id, child_level):
    """Return only proven direct children; absence needs explicit complete coverage."""
    coverage = next(
        (
            c
            for c in snapshot.coverage
            if c.parent_record_id == parent_id and c.child_level == child_level
        ),
        None,
    )
    if coverage is None or coverage.status != "complete":
        return None
    nodes = {n.record.id: n for n in snapshot.nodes}
    parent = nodes.get(parent_id)
    if parent is None:
        return None
    expected = {"layer": "stand", "element": "layer", "regeneration": "stand"}
    if expected.get(child_level) != parent.level:
        return None
    children = []
    for node in snapshot.nodes:
        if node.level != child_level:
            continue
        if node.link_status != "matched":
            return None
        linked = nodes.get(node.parent_record_id)
        # An unassigned/invalid relation could belong here; don't turn it into an empty set.
        if linked is None or linked.level != expected[child_level]:
            return None
        if (node.record.dataset_id, node.record.dataset_revision) != (
            linked.record.dataset_id,
            linked.record.dataset_revision,
        ):
            return None
        if node.parent_record_id != parent_id:
            continue
        if (node.record.dataset_id, node.record.dataset_revision) != (
            parent.record.dataset_id,
            parent.record.dataset_revision,
        ):
            return None
        children.append(node)
    return tuple(children)


def _required(record, field):
    value = _field(record, field)
    if value is None:
        return "violation", "required_field_absent", "missing", "confirmed_contradiction"
    if value.state == "parse_error":
        return "violation", "required_value_parse_error", "parse_error", "format_error"
    if value.state == "unknown":
        return "insufficient_data", "required_value_unknown", None, None
    if value.state == "not_applicable":
        return (
            "violation",
            "required_value_not_applicable",
            "inconsistency",
            "confirmed_contradiction",
        )
    if value.state == "missing" or (
        isinstance(value.normalized_value, str) and not value.normalized_value.strip()
    ):
        return "violation", "required_value_blank", "missing", "confirmed_contradiction"
    return "passed", "required_value_observed", None, None


def evaluate_hierarchy(
    snapshot: HierarchySnapshot, policy: HierarchyPolicy, analysis_run_id: UUID
) -> tuple[RuleOutcome, ...]:
    """Evaluate supplied records without joining, excluding, or mutating any row."""
    outcomes = []
    for stand in (n for n in snapshot.nodes if n.level == "stand"):
        record = stand.record

        def result(suffix, status, reason, **kwargs):
            return _outcome(
                record,
                analysis_run_id,
                policy.id + suffix,
                policy.version,
                policy.source,
                status,
                reason,
                **kwargs,
            )

        for role, field in policy.fields.model_dump().items():
            status, reason, signal, classification = _required(record, field)
            if not _source_available(policy.source, snapshot):
                status, reason = "unavailable_reference", "required_field_source_unavailable"
            outcomes.append(
                result(
                    ".required." + role,
                    status,
                    reason,
                    field=field,
                    signal=signal,
                    classification=classification,
                    context="Expert answer 1: required field of stand record; no numeric range inferred",
                )
            )
        category_field = policy.category_field or policy.fields.land_kind
        land = _field(record, category_field)
        category = next(
            (
                c
                for c in policy.land_categories
                if land is not None
                and land.state == "observed"
                and land.normalized_value == c.value
            ),
            None,
        )
        if category is None or not _source_available(category.source, snapshot):
            outcomes.append(
                result(".layers", "insufficient_data", "land_category_binding_unconfirmed")
            )
            continue
        if not category.natural_origin:
            outcomes.append(
                result(".layers", "not_applicable", "not_confirmed_natural_origin_category")
            )
            continue
        context = f"Natural-origin category: {category.value}; {category.source.reference_id}@{category.version}; {category.source.locator}"
        if not _source_available(policy.source, snapshot):
            outcomes.append(
                result(".layers", "unavailable_reference", "hierarchy_rule_source_unavailable")
            )
            continue
        layers = _children(snapshot, record.id, "layer")
        if layers is None:
            outcomes.append(
                result(".layers", "insufficient_data", "layer_relation_coverage_unconfirmed")
            )
            continue
        outcomes.append(
            result(
                ".layers",
                "passed" if layers else "violation",
                "layer_present" if layers else "natural_origin_layer_missing",
                field=category_field,
                signal="missing",
                context=context,
            )
        )
        for layer in layers:
            elements = _children(snapshot, layer.record.id, "element")
            status = (
                "insufficient_data" if elements is None else "passed" if elements else "violation"
            )
            reason = (
                "element_relation_coverage_unconfirmed"
                if elements is None
                else "species_records_present"
                if elements
                else "tentative_expert_species_records_missing"
            )
            outcomes.append(
                _outcome(
                    layer.record,
                    analysis_run_id,
                    policy.id + ".species",
                    policy.version,
                    policy.source,
                    status,
                    reason,
                    signal="anomaly",
                    classification="statistical_suspicion",
                    context=context + "; expert wording is tentative, not a verified norm",
                )
            )
    return tuple(outcomes)


def evaluate_stock_sum(
    snapshot: HierarchySnapshot, check: StockSumCheck, analysis_run_id: UUID
) -> RuleOutcome:
    """Admit a complete compatible direct-child sum to the existing aggregate engine."""
    nodes = {n.record.id: n for n in snapshot.nodes}
    total = nodes.get(check.total.record_id)
    if total is None:
        return RuleOutcome(
            schema_version="1.0",
            record_id=check.total.record_id,
            rule_id=check.id,
            rule_version=check.version,
            status="insufficient_data",
            reason="stock_total_record_unavailable",
        )

    def result(status, reason):
        return _outcome(
            total.record,
            analysis_run_id,
            check.id,
            check.version,
            check.source,
            status,
            reason,
            field=check.total.field,
            context=f"Complete {check.component_level} children of {total.level}; "
            f"{check.total.quantity_kind}; {check.total.basis}; "
            f"{check.total.organism_group}; {check.total.observation_scope}",
            extra=(
                Evidence(code="stock_semantics_and_tolerance", message=check.model_dump_json()),
            ),
        )

    quantities = (check.total, *check.components)
    if (
        not check.relationship_confirmed
        or check.absolute_tolerance is None
        or any(
            not _source_available(source, snapshot)
            for source in (check.source, check.tolerance_source)
        )
    ):
        return result("unavailable_reference", "stock_relationship_or_tolerance_unconfirmed")
    if any(
        not q.definition_confirmed
        or not _source_available(q.source, snapshot)
        or q.organism_group == "unknown"
        for q in quantities
    ):
        return result("unavailable_reference", "stock_quantity_definition_unconfirmed")
    if (
        len(
            {
                (q.unit, q.quantity_kind, q.basis, q.organism_group, q.observation_scope)
                for q in quantities
            }
        )
        != 1
    ):
        return result("insufficient_data", "stock_quantity_semantics_incompatible")
    if any(q.unit != {"total": "m3", "per_ha": "m3/ha"}[q.basis] for q in quantities):
        return result("insufficient_data", "stock_unit_and_basis_incompatible")
    children = _children(snapshot, total.record.id, check.component_level)
    if children is None or {c.record.id for c in children} != {
        c.record_id for c in check.components
    }:
        return result("insufficient_data", "stock_relation_population_incomplete")
    relation = AggregateRelation(
        schema_version="1.0",
        id=check.id,
        version=check.version,
        component_record_ids=tuple(c.record_id for c in check.components),
        component_field=check.components[0].field,
        total_record_id=total.record.id,
        total_field=check.total.field,
        unit=check.total.unit,
        quantity_kind=check.total.quantity_kind,
        component_level=check.component_level,
        total_level=total.level,
        relationship_confirmed=True,
        absolute_tolerance=check.absolute_tolerance,
        source=check.source,
    )
    contexts = {
        id_: EvaluationContext(
            analysis_run_id=analysis_run_id,
            record_level=node.level,
            allow_synthetic=snapshot.allow_synthetic,
        )
        for id_, node in nodes.items()
    }
    outcome = _aggregate(
        relation, {id_: n.record for id_, n in nodes.items()}, contexts, contexts[total.record.id]
    )
    return result(outcome.status, outcome.reason)
