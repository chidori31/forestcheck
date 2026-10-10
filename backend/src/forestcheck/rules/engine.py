"""Pure evaluation. Applicability and unavailable evidence are never passes."""

from collections import Counter
from decimal import Decimal, localcontext
from uuid import uuid5

from forestcheck.domain.contracts import Evidence, Finding, Record
from forestcheck.rules.models import (
    Codebook,
    CodeCheck,
    EvaluationContext,
    FieldContractCheck,
    RequiredCheck,
    RoundingCheck,
    RuleDefinition,
    RuleOutcome,
    StockBalanceCheck,
    UniqueKeyCheck,
)


class RuleEngine:
    def __init__(self, rules: list[RuleDefinition], *, codebooks: list[Codebook] | None = None):
        self.rules = tuple(rules)
        if len({r.id for r in self.rules}) != len(self.rules):
            raise ValueError("duplicate_rule_id")
        books = codebooks or []
        if len({b.id for b in books}) != len(books):
            raise ValueError("duplicate_codebook_id")
        self.codebooks = {b.id: b for b in books}

    def evaluate(self, record: Record, context: EvaluationContext) -> tuple[RuleOutcome, ...]:
        return tuple(self._evaluate(rule, record, context, None) for rule in self.rules)

    def evaluate_many(
        self, records: list[Record], context: EvaluationContext
    ) -> tuple[RuleOutcome, ...]:
        if len({r.id for r in records}) != len(records):
            raise ValueError("duplicate_technical_record_id")
        indices = {}
        for rule in self.rules:
            if isinstance(rule.check, UniqueKeyCheck):
                keys = [self._key(r, rule.check) for r in records]
                indices[rule.id] = Counter(k for k in keys if k is not None)
        return tuple(
            self._evaluate(rule, record, context, indices.get(rule.id))
            for record in records
            for rule in self.rules
        )

    @staticmethod
    def _key(record, check):
        fields = {f.name: f for f in record.fields}
        if any(name not in fields or fields[name].state != "observed" for name in check.fields):
            return None
        # Namespace by dataset/revision; equality across unrelated snapshots is not a duplicate.
        return (
            record.dataset_id,
            record.dataset_revision,
            tuple(
                (type(fields[n].normalized_value).__name__, fields[n].normalized_value)
                for n in check.fields
            ),
        )

    def _evaluate(self, rule, record, context, counts):
        fields = {f.name: f for f in record.fields}

        def result(status, reason, signal_class=None, signal_kind=None, extra_evidence=()):
            finding = None
            if status == "violation":
                if rule.source.kind == "heuristic":
                    signal_class, signal_kind = "statistical_suspicion", "anomaly"
                ref = f"{rule.id}@{rule.version}"
                coordinates = next(
                    (fields[n].source for n in rule.dependencies if n in fields), None
                )
                finding = Finding(
                    schema_version="1.0",
                    id=uuid5(context.analysis_run_id, f"{record.id}:{record.revision}:{ref}"),
                    record_id=record.id,
                    record_revision=record.revision,
                    analysis_run_id=context.analysis_run_id,
                    field_name=rule.dependencies[0] if len(rule.dependencies) == 1 else None,
                    signal_kind=signal_kind,
                    severity=rule.severity,
                    priority=rule.priority,
                    evidence=(
                        Evidence(
                            code=reason,
                            message=rule.explanation,
                            source=coordinates,
                            reference_id=ref,
                        ),
                        Evidence(
                            code="rule_source",
                            message=f"{rule.source.reference_id}; {rule.source.locator}",
                            reference_id=rule.source.reference_id,
                        ),
                        *extra_evidence,
                    ),
                    review_status="unreviewed",
                )
            return RuleOutcome(
                schema_version="1.0",
                record_id=record.id,
                rule_id=rule.id,
                rule_version=rule.version,
                status=status,
                reason=reason,
                signal_class=signal_class,
                finding=finding,
            )

        if context.record_level not in rule.record_levels:
            return result("not_applicable", "record_level_not_applicable")
        if rule.categories:
            if context.category is None:
                return result("insufficient_data", "object_category_unavailable")
            if context.category not in rule.categories:
                return result("not_applicable", "object_category_not_applicable")
        if rule.source.kind == "synthetic" and not context.allow_synthetic:
            return result("unavailable_reference", "synthetic_rule_disabled")
        check = rule.check
        if isinstance(check, UniqueKeyCheck):
            if not check.grain_confirmed:
                return result("unavailable_reference", "key_grain_not_confirmed")
            key = self._key(record, check)
            if key is None or counts is None:
                return result("insufficient_data", "key_or_comparison_population_unavailable")
            if counts[key] > 1:
                return result(
                    "violation",
                    "duplicate_confirmed_composite_key",
                    "confirmed_contradiction",
                    "inconsistency",
                )
            return result("passed", "key_unique_in_supplied_population")
        missing = [n for n in rule.dependencies if n not in fields or fields[n].state == "missing"]
        if isinstance(check, RequiredCheck):
            if missing:
                return result(
                    "violation",
                    "required_for_declared_category",
                    "confirmed_contradiction",
                    "missing",
                )
        if missing:
            return result("insufficient_data", "missing_dependencies")
        if any(fields[n].state == "not_applicable" for n in rule.dependencies):
            return result("not_applicable", "dependency_not_applicable")
        if any(fields[n].state == "parse_error" for n in rule.dependencies):
            if isinstance(check, FieldContractCheck):
                return result("violation", "source_parse_error", "format_error", "parse_error")
            return result("insufficient_data", "dependency_parse_error")
        if any(fields[n].state == "unknown" for n in rule.dependencies):
            return result("insufficient_data", "dependency_semantics_unknown")
        if isinstance(check, RequiredCheck):
            return result("passed", "required_value_observed")
        if isinstance(check, CodeCheck):
            book = self.codebooks.get(check.codebook_id)
            if book is None or (book.source.kind == "synthetic" and not context.allow_synthetic):
                return result("unavailable_reference", "codebook_unavailable")
            if check.codebook_version is None or book.version != check.codebook_version:
                return result("unavailable_reference", "codebook_version_unconfirmed")
            value = fields[check.field].normalized_value
            if type(value) is not str:
                return result("violation", "code_type_mismatch", "format_error", "parse_error")
            if value not in book.codes:
                heuristic = book.source.kind == "heuristic"
                evidence = (
                    Evidence(
                        code="codebook_source",
                        reference_id=f"{book.id}@{book.version}",
                        message=f"{book.source.reference_id}; {book.source.locator}",
                    ),
                )
                return result(
                    "violation",
                    "unknown_dictionary_code",
                    "statistical_suspicion" if heuristic else "confirmed_contradiction",
                    "anomaly" if heuristic else "reference_mismatch",
                    evidence,
                )
            return result("passed", "code_present_in_versioned_dictionary")
        if isinstance(check, FieldContractCheck):
            value = fields[check.field]
            expected = {"text": (str,), "integer": (int,), "number": (int, float)}[check.value_type]
            if type(value.normalized_value) not in expected:
                return result("violation", "value_type_mismatch", "format_error", "parse_error")
            if check.value_type != "text" and (value.unit is None or check.unit is None):
                return result("insufficient_data", "numeric_unit_unconfirmed")
            if value.unit != check.unit:
                return result("violation", "declared_unit_mismatch", "format_error", "parse_error")
            return result("passed", "type_and_declared_unit_match")
        if isinstance(check, RoundingCheck):
            value = fields[check.field]
            if value.unit != check.unit or type(value.normalized_value) not in (int, float):
                return result("insufficient_data", "rounding_units_or_type_incompatible")
            with localcontext() as decimal_context:
                decimal_context.prec = 2048
                aligned = Decimal(str(value.normalized_value)) % Decimal(check.step) == 0
            if not aligned:
                return result(
                    "violation",
                    "declared_rounding_increment_mismatch",
                    "format_error",
                    "parse_error",
                )
            return result("passed", "declared_rounding_increment_match")
        if isinstance(check, StockBalanceCheck):
            if (
                not check.relationship_confirmed
                or len(set(check.quantity_levels)) != 1
                or check.quantity_levels[0] != context.record_level
                or len(set(check.stock_kinds)) != 1
                or check.tolerance_kind == "unavailable"
            ):
                return result(
                    "unavailable_reference", "stock_relationship_or_tolerance_unconfirmed"
                )
            values = [fields[n] for n in (check.per_ha, check.area, check.total)]
            if [v.unit for v in values] != ["m3/ha", "ha", "m3"]:
                return result("insufficient_data", "stock_units_incompatible")
            if any(type(v.normalized_value) not in (int, float) for v in values):
                return result("insufficient_data", "stock_types_incompatible")
            with localcontext() as decimal_context:
                decimal_context.prec = 2048
                a, b, c = [Decimal(str(v.normalized_value)) for v in values]
                if check.tolerance_kind == "rounding":
                    da, db, dc = [Decimal(q) / 2 for q in check.rounding_steps]
                    products = [(a + x * da) * (b + y * db) for x in (-1, 1) for y in (-1, 1)]
                    compatible = max(products) >= c - dc and min(products) <= c + dc
                else:
                    compatible = abs(a * b - c) <= Decimal(check.absolute_tolerance)
            if compatible:
                return result("passed", "stock_values_within_declared_tolerance")
            heuristic = check.tolerance_kind == "heuristic"
            return result(
                "violation",
                "stock_balance_outside_tolerance",
                "statistical_suspicion" if heuristic else "confirmed_contradiction",
                "anomaly" if heuristic else "inconsistency",
            )
        raise TypeError("unsupported_check")
