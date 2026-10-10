"""Conservative proposal composition and copy-only validation. No apply operation."""

import hashlib
import json
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal, localcontext
from itertools import combinations
from uuid import UUID, uuid5

from forestcheck.domain.contracts import FieldValue, Finding, Record
from forestcheck.repair.models import (
    AggregateOutcome,
    AggregateRelation,
    Candidate,
    CandidateValidation,
    DatasetBinding,
    Explanation,
    NumericRange,
    PatchOperation,
    PrecisionPolicy,
    ProposedPatch,
    RecommendationResult,
    SourceSupport,
    SuggestionV2,
    Uncertainty,
    UnitHypothesis,
)
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.models import EvaluationContext, RuleOutcome


def canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def verify_patch(patch: ProposedPatch, record: Record, binding: DatasetBinding) -> None:
    """Verify reviewable before/after against a caller-verified snapshot; never applies it."""
    ProposedPatch.model_validate(patch.model_dump(mode="json"))
    if patch.binding != binding:
        raise ValueError("patch_snapshot_binding_mismatch")
    if (record.dataset_id, record.dataset_revision) != (
        binding.dataset_id,
        binding.dataset_revision,
    ):
        raise ValueError("patch_dataset_binding_mismatch")
    if (
        record.id != patch.record_id
        or record.revision != patch.record_revision
        or canonical_digest(record.model_dump(mode="json")) != patch.record_sha256
    ):
        raise ValueError("patch_record_version_mismatch")
    current = next((f for f in record.fields if f.name == patch.operation.field_name), None)
    if current != patch.operation.before:
        raise ValueError("patch_record_before_mismatch")


def _families(sources):
    groups = []
    for source in sources:
        roots, names = set(source.dependency_roots), {source.source_id}
        remaining = []
        for prior_roots, prior_names in groups:
            if roots & prior_roots:
                roots |= prior_roots
                names |= prior_names
            else:
                remaining.append((prior_roots, prior_names))
        groups = [*remaining, (roots, names)]
    return tuple(sorted(tuple(sorted(names)) for _, names in groups))


def _rounded(candidate, step):
    with localcontext() as ctx:
        ctx.prec = 2048
        increment = Decimal(step)

        def number(value, direction):
            if value is None:
                return None
            result = (Decimal(str(value)) / increment).to_integral_value(
                rounding=direction
            ) * increment
            return int(result) if result == result.to_integral_value() else float(result)

        return Candidate.model_validate(
            {
                **candidate.model_dump(),
                "value": number(candidate.value, ROUND_HALF_UP),
                "lower": number(candidate.lower, ROUND_FLOOR),
                "upper": number(candidate.upper, ROUND_CEILING),
            }
        )


def _conflict(left, right):
    lo1 = left.lower if left.lower is not None else left.value
    hi1 = left.upper if left.upper is not None else left.value
    lo2 = right.lower if right.lower is not None else right.value
    hi2 = right.upper if right.upper is not None else right.value
    if hi1 < lo2 or hi2 < lo1:
        return True
    return (left.value is not None and not lo2 <= left.value <= hi2) or (
        right.value is not None and not lo1 <= right.value <= hi1
    )


def _aggregate(relation, records, contexts, context):
    def result(status, reason):
        return AggregateOutcome(
            relation_id=relation.id, relation_version=relation.version, status=status, reason=reason
        )

    if not relation.relationship_confirmed or relation.source.kind == "heuristic":
        return result("unavailable_reference", "aggregate_relationship_not_confirmed")
    if relation.source.kind == "synthetic" and not context.allow_synthetic:
        return result("unavailable_reference", "synthetic_reference_disabled")
    ids = (*relation.component_record_ids, relation.total_record_id)
    selected = [records.get(id_) for id_ in ids]
    if any(r is None for r in selected):
        return result("insufficient_data", "aggregate_population_incomplete")
    if any(id_ not in contexts for id_ in ids):
        return result("insufficient_data", "aggregate_record_level_unavailable")
    if (
        any(
            contexts[id_].record_level != relation.component_level
            for id_ in relation.component_record_ids
        )
        or contexts[relation.total_record_id].record_level != relation.total_level
    ):
        return result("unavailable_reference", "aggregate_record_level_mismatch")
    if len({(r.dataset_id, r.dataset_revision) for r in selected}) != 1:
        return result("unavailable_reference", "aggregate_snapshot_mismatch")
    values = []
    for r, name in zip(
        selected, [relation.component_field] * (len(selected) - 1) + [relation.total_field]
    ):
        value = next((f for f in r.fields if f.name == name), None)
        if value is None or value.state != "observed":
            return result("insufficient_data", "aggregate_input_not_observed")
        if value.unit != relation.unit or type(value.normalized_value) not in (int, float):
            return result("insufficient_data", "aggregate_units_or_types_incompatible")
        values.append(Decimal(str(value.normalized_value)))
    with localcontext() as ctx:
        ctx.prec = 2048
        consistent = abs(sum(values[:-1]) - values[-1]) <= Decimal(relation.absolute_tolerance)
    return result(
        "passed" if consistent else "violation",
        "aggregate_within_declared_tolerance"
        if consistent
        else "aggregate_outside_declared_tolerance",
    )


def _validate(record, after, rule_engine, context, related, relations, related_contexts):
    records = [record, *related]
    if len({r.id for r in records}) != len(records):
        raise ValueError("duplicate_related_record")
    contexts = dict(related_contexts or {})
    if set(contexts) - {r.id for r in related}:
        raise ValueError("unexpected_related_context")
    contexts[record.id] = context

    def evaluate(population):
        groups, results = {}, []
        for row in population:
            row_context = contexts.get(row.id)
            if row_context is None:
                results.extend(
                    RuleOutcome(
                        schema_version="1.0",
                        record_id=row.id,
                        rule_id=rule.id,
                        rule_version=rule.version,
                        status="insufficient_data",
                        reason="related_record_context_unavailable",
                    )
                    for rule in rule_engine.rules
                )
                continue
            key = row_context.model_dump_json()
            groups.setdefault(key, (row_context, []))[1].append(row)
        for group_context, rows in groups.values():
            results.extend(rule_engine.evaluate_many(rows, group_context))
        return tuple(results)

    before_outcomes = evaluate(records)
    copied = [after, *(r.model_copy(deep=True) for r in related)]
    after_outcomes = evaluate(copied)
    previous = {(o.record_id, o.rule_id, o.rule_version): o for o in before_outcomes}
    new, degraded, unresolved = [], [], []
    for outcome in after_outcomes:
        key = (outcome.record_id, outcome.rule_id, outcome.rule_version)
        old = previous[key]
        label = f"{outcome.record_id}:{outcome.rule_id}@{outcome.rule_version}"
        if outcome.status == "violation":
            (unresolved if old.status == "violation" else new).append(label)
        if old.status in ("passed", "violation") and outcome.status not in ("passed", "violation"):
            degraded.append(label)
    before_aggregates = tuple(
        _aggregate(r, {v.id: v for v in records}, contexts, context) for r in relations
    )
    after_aggregates = tuple(
        _aggregate(r, {v.id: v for v in copied}, contexts, context) for r in relations
    )
    limits = []
    if any(r.id not in contexts for r in related):
        limits.append("related_record_context_unavailable")
    if not relations:
        limits.append("linked_aggregate_coverage_unavailable")
    if not before_outcomes:
        limits.append("rule_coverage_unavailable")
    if any(o.status in ("unavailable_reference", "insufficient_data") for o in after_outcomes):
        limits.append("some_rules_could_not_be_checked")
    if unresolved:
        limits.append("existing_violations_remain")
    return CandidateValidation(
        schema_version="1.0",
        before=before_outcomes,
        after=after_outcomes,
        aggregate_before=before_aggregates,
        aggregate_after=after_aggregates,
        new_violations=tuple(new),
        degraded_checks=tuple(degraded),
        unresolved_violations=tuple(unresolved),
        limitations=tuple(limits),
    )


class RecommendationEngine:
    def __init__(self, allowed_numeric_fields: dict[str, str]):
        # This allowlist is a checked mapping decision, not an inference from an opaque column ID.
        self.allowed_numeric_fields = dict(allowed_numeric_fields)

    def recommend(
        self,
        record: Record,
        finding: Finding,
        field_name: str,
        *,
        candidates: tuple[Candidate, ...],
        sources: tuple[SourceSupport, ...] = (),
        precision: PrecisionPolicy | None,
        binding: DatasetBinding,
        rule_engine: RuleEngine,
        context: EvaluationContext,
        related_records: tuple[Record, ...] = (),
        related_contexts: dict[UUID, EvaluationContext] | None = None,
        suspect_fields: tuple[str, ...] = (),
        aggregate_relations: tuple[AggregateRelation, ...] = (),
    ) -> RecommendationResult:
        if (record.dataset_id, record.dataset_revision) != (
            binding.dataset_id,
            binding.dataset_revision,
        ):
            raise ValueError("dataset_binding_mismatch")
        if (finding.record_id, finding.record_revision) != (record.id, record.revision):
            raise ValueError("finding_record_mismatch")
        if finding.field_name not in (None, field_name):
            raise ValueError("finding_field_mismatch")
        before = next((f for f in record.fields if f.name == field_name), None)
        all_sources = tuple(c.support for c in candidates) + tuple(sources)
        if len({s.source_id for s in all_sources}) != len(all_sources):
            raise ValueError("duplicate_evidence_source_id")
        available = tuple(s for s in all_sources if s.status == "available")
        families = _families(available)
        limitations = ["unverified_proposal_not_expert_label", "independence_not_proven"]
        limitations.extend(item for source in all_sources for item in source.limitations)
        validation = CandidateValidation(
            schema_version="1.0", limitations=("candidate_not_validated",)
        )
        status, reason, chosen = "refused", None, None
        alternatives = ()
        protected = {
            "id",
            "record_id",
            "dataset_id",
            "region",
            "territory",
            "compartment",
            "stand",
            "layer",
            "element",
            "species",
            "composition",
            "business_key",
        }
        if field_name.casefold() in protected:
            reason = "protected_field"
        elif field_name not in self.allowed_numeric_fields:
            reason = "numeric_field_not_allowlisted"
        elif before is None:
            reason = "field_absent"
        elif before.state in ("not_applicable", "unknown"):
            reason = "field_" + before.state
        elif before.unit != self.allowed_numeric_fields[field_name] or any(
            c.unit != before.unit for c in candidates
        ):
            reason = "unit_mismatch_hypothesis"
        elif precision is None or precision.source.kind in ("heuristic", "explicit_configuration"):
            reason = "measurement_precision_unavailable"
        elif precision.source.kind == "synthetic" and not context.allow_synthetic:
            reason = "measurement_precision_unavailable"
        elif precision.field_name != field_name or precision.unit != before.unit:
            reason = "precision_field_or_unit_mismatch"
        elif not candidates:
            reason = (
                "unavailable_reference"
                if any(s.status == "unavailable_reference" for s in sources)
                else "insufficient_data"
            )
        else:
            suspect = set(suspect_fields)
            affected = set().union(
                *(suspect.intersection(c.support.input_fields) for c in candidates)
            )
            limitations.extend("suspect_input:" + f for f in sorted(affected))
            # Refuse even single-input dependencies for a numeric patch; alternatives remain reviewable.
            alternatives = tuple(_rounded(c, precision.step) for c in candidates)
            if any(field_name in c.support.input_fields for c in candidates):
                reason = "target_dependency_leakage"
            elif len(affected) > 1:
                reason = "multiple_suspect_inputs"
            elif affected:
                status, reason = "manual_review", "suspect_input_requires_review"
            elif any(_conflict(a, b) for a, b in combinations(candidates, 2)) or any(
                _conflict(a, b) for a, b in combinations(alternatives, 2)
            ):
                reason = "conflicting_sources"
            else:
                methods = {
                    name: n
                    for n, name in enumerate(
                        ("arithmetic", "growth", "conditional_model", "baseline", "comparable")
                    )
                }
                choices = sorted(
                    (c for c in alternatives if c.value is not None),
                    key=lambda c: (methods[c.support.method], c.support.source_id),
                )
                if not choices:
                    status, reason = "manual_review", "range_without_supported_point"
                else:
                    chosen = choices[0]
                    after_field = FieldValue.model_validate(
                        {
                            **before.model_dump(),
                            "state": "observed",
                            "normalized_value": chosen.value,
                        }
                    )
                    after = Record.model_validate(
                        {
                            **record.model_dump(),
                            "fields": tuple(
                                after_field if f.name == field_name else f.model_copy(deep=True)
                                for f in record.fields
                            ),
                        }
                    )
                    validation = _validate(
                        record,
                        after,
                        rule_engine,
                        context,
                        related_records,
                        aggregate_relations,
                        related_contexts,
                    )
                    limitations.extend(validation.limitations)
                    if validation.new_violations or validation.degraded_checks:
                        status, reason = "manual_review", "candidate_creates_or_hides_violation"
                    elif any(
                        outcome.record_id == record.id
                        and outcome.status == "violation"
                        and field_name in definition.dependencies
                        for outcome in validation.after
                        for definition in rule_engine.rules
                        if outcome.rule_id == definition.id
                    ):
                        status, reason = "manual_review", "target_violation_unresolved"
                    elif "related_record_context_unavailable" in validation.limitations:
                        status, reason = "manual_review", "related_record_context_unavailable"
                    elif any(a.status == "violation" for a in validation.aggregate_after):
                        status, reason = "manual_review", "linked_aggregate_requires_coupled_review"
                    elif any(a.status != "passed" for a in validation.aggregate_after):
                        status, reason = "manual_review", "linked_aggregate_cannot_be_validated"
                    elif before.state == "observed" and chosen.value == before.normalized_value:
                        status = "no_change"
                    else:
                        status = "proposed"
        selected = chosen if status in ("proposed", "no_change") else None
        if alternatives and any(
            c.support.method in ("conditional_model", "baseline", "comparable")
            for c in alternatives
        ):
            limitations.append("rounded_filtered_intervals_not_recalibrated")
        if not aggregate_relations:
            limitations.append("linked_aggregate_coverage_unavailable")
        if any(s.status == "unavailable_reference" for s in all_sources):
            limitations.append("unavailable_reference")
        missing = tuple(s.basis for s in all_sources if s.status != "available") + (
            (reason,) if reason else ()
        )
        explanation = Explanation(
            observed="Поле отсутствует"
            if before is None
            else f"Состояние: {before.state}; raw: {before.original_value!r}; единица: {before.unit}",
            trigger=tuple(e.message for e in finding.evidence),
            comparisons=tuple(
                f"{s.method}: {s.source_id}@{s.version}; {s.basis}" for s in all_sources
            ),
            applicability=tuple(s.basis for s in available),
            considered_range="Диапазон не выдан"
            if selected is None or selected.lower is None
            else f"{selected.lower}..{selected.upper} {selected.unit}",
            missing=missing,
            specialist_actions=(
                "Проверить исходное измерение, единицу и применимость источника.",
                "Отдельно решить, есть ли ошибка и допустимо ли предложенное значение.",
                "Проверить связанные показатели; исходник автоматически не изменяется.",
            ),
        )
        rationale = (
            "Подозрение и численная оценка не подтверждают ошибку; решение принимает специалист."
        )
        if "unavailable_reference" in limitations or reason == "unavailable_reference":
            rationale += " Нет подходящего эталона для недоступного источника."
        if reason:
            rationale += f" Причина отказа от числа: {reason}."
        else:
            rationale += " Оценка согласована с доступными проверками на копии записи."
        uncertainty = Uncertainty(
            category="unavailable"
            if selected is None
            else "deterministic_relation"
            if selected.support.method == "arithmetic"
            else "uncalibrated_estimate",
            basis="Число не выбрано."
            if selected is None
            else selected.uncertainty_basis
            + " Округление и отбор рекомендаций не прошли отдельную калибровку.",
        )
        identity = canonical_digest(
            {
                "binding": binding.model_dump(mode="json"),
                "record": record.model_dump(mode="json"),
                "field": field_name,
                "finding": finding.model_dump(mode="json"),
                "sources": [s.model_dump(mode="json") for s in all_sources],
                "candidates": [c.model_dump(mode="json") for c in candidates],
                "precision": precision.model_dump(mode="json") if precision else None,
                "status": status,
                "reason": reason,
                "validation": validation.model_dump(mode="json"),
                "suspect_fields": sorted(set(suspect_fields)),
            }
        )
        suggestion = SuggestionV2(
            schema_version="2.0",
            id=uuid5(finding.id, f"recommendation-v2:{identity}"),
            finding_id=finding.id,
            record_id=record.id,
            record_revision=record.revision,
            field_name=field_name,
            proposed_value=selected.value if selected else None,
            unit=before.unit if before else None,
            rationale=rationale,
            before=before,
            binding=binding,
            method=selected.support.method if selected else None,
            evidence=all_sources,
            candidate_range=NumericRange(lower=selected.lower, upper=selected.upper)
            if selected and selected.lower is not None
            else None,
            alternatives=alternatives,
            unit_hypotheses=tuple(
                UnitHypothesis(
                    source_id=c.support.source_id, observed_unit=before.unit, candidate_unit=c.unit
                )
                for c in candidates
                if before is not None and c.unit != before.unit
            ),
            status=status,
            refusal_reason=reason,
            limitations=tuple(dict.fromkeys(limitations)),
            uncertainty=uncertainty,
            support_families=families,
            explanation=explanation,
        )
        patch = None
        if status == "proposed":
            payload = {
                "schema_version": "1.0",
                "id": str(uuid5(suggestion.id, "patch-v1")),
                "status": "proposed_only",
                "suggestion_id": str(suggestion.id),
                "binding": binding.model_dump(mode="json"),
                "record_id": str(record.id),
                "record_revision": record.revision,
                "record_sha256": canonical_digest(record.model_dump(mode="json")),
                "operation": PatchOperation(
                    field_name=field_name, before=before, after=after_field
                ).model_dump(mode="json"),
                "provenance": [s.model_dump(mode="json") for s in all_sources],
                "validation_sha256": canonical_digest(validation.model_dump(mode="json")),
            }
            patch = ProposedPatch.model_validate(
                {**payload, "integrity_sha256": canonical_digest(payload)}
            )
        return RecommendationResult(
            schema_version="1.0", suggestion=suggestion, patch=patch, validation=validation
        )
