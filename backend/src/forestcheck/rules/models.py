from decimal import Decimal
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, model_validator

from forestcheck.domain.contracts import Contract, Finding, Text, ValueObject

DecimalText = Annotated[str, Field(strict=True, pattern=r"^\d+(?:\.\d+)?$", max_length=64)]


class RuleSource(ValueObject):
    reference_id: Text
    locator: Text
    kind: Literal[
        "documented", "observed_header", "explicit_configuration", "heuristic", "synthetic"
    ]


class FieldContractCheck(ValueObject):
    kind: Literal["field_contract"]
    field: Text
    value_type: Literal["text", "number", "integer"]
    unit: Text | None


class RequiredCheck(ValueObject):
    kind: Literal["required"]
    field: Text


class CodeCheck(ValueObject):
    kind: Literal["code"]
    field: Text
    codebook_id: Text
    codebook_version: Text | None = None


class UniqueKeyCheck(ValueObject):
    kind: Literal["unique_key"]
    fields: Annotated[tuple[Text, ...], Field(min_length=1)]
    grain_confirmed: StrictBool


class RoundingCheck(ValueObject):
    kind: Literal["rounding"]
    field: Text
    unit: Text
    step: DecimalText

    @model_validator(mode="after")
    def positive_step(self) -> Self:
        if Decimal(self.step) <= 0:
            raise ValueError("positive_step_required")
        return self


class StockBalanceCheck(ValueObject):
    kind: Literal["stock_balance"]
    per_ha: Text
    area: Text
    total: Text
    quantity_levels: tuple[Text, Text, Text]
    stock_kinds: tuple[Text, Text]
    relationship_confirmed: StrictBool
    tolerance_kind: Literal["rounding", "explicit", "heuristic", "unavailable"]
    rounding_steps: tuple[DecimalText, DecimalText, DecimalText] | None = None
    absolute_tolerance: DecimalText | None = None

    @model_validator(mode="after")
    def complete_tolerance(self) -> Self:
        if len({self.per_ha, self.area, self.total}) != 3:
            raise ValueError("distinct_quantities_required")
        if self.tolerance_kind == "rounding":
            if self.rounding_steps is None or self.absolute_tolerance is not None:
                raise ValueError("rounding_steps_required")
            if any(Decimal(v) <= 0 for v in self.rounding_steps):
                raise ValueError("positive_rounding_steps_required")
        elif self.tolerance_kind in ("explicit", "heuristic"):
            if self.absolute_tolerance is None or self.rounding_steps is not None:
                raise ValueError("absolute_tolerance_required")
        elif self.rounding_steps is not None or self.absolute_tolerance is not None:
            raise ValueError("unavailable_tolerance_has_no_value")
        return self


Check = Annotated[
    FieldContractCheck
    | RequiredCheck
    | CodeCheck
    | UniqueKeyCheck
    | RoundingCheck
    | StockBalanceCheck,
    Field(discriminator="kind"),
]


class RuleDefinition(Contract):
    id: Text
    version: Text
    record_levels: Annotated[tuple[Text, ...], Field(min_length=1)]
    categories: tuple[Text, ...] = ()
    dependencies: Annotated[tuple[Text, ...], Field(min_length=1)]
    source: RuleSource
    severity: Literal["info", "warning", "error", "critical"]
    priority: Annotated[int, Field(strict=True, ge=0, le=3)]
    explanation: Text
    check: Check

    @model_validator(mode="after")
    def explicit_dependencies(self) -> Self:
        check = self.check
        required = (
            check.fields
            if isinstance(check, UniqueKeyCheck)
            else (check.per_ha, check.area, check.total)
            if isinstance(check, StockBalanceCheck)
            else (check.field,)
        )
        if len(set(self.dependencies)) != len(self.dependencies) or set(required) != set(
            self.dependencies
        ):
            raise ValueError("dependencies_must_match_check")
        if isinstance(check, UniqueKeyCheck) and len(set(check.fields)) != len(check.fields):
            raise ValueError("duplicate_key_component")
        if isinstance(check, RequiredCheck) and not self.categories:
            raise ValueError("required_field_needs_object_categories")
        return self


class Codebook(Contract):
    id: Text
    version: Text
    source: RuleSource
    codes: Annotated[tuple[Text, ...], Field(min_length=1)]


class EvaluationContext(ValueObject):
    analysis_run_id: UUID
    record_level: Text
    category: Text | None = None
    allow_synthetic: StrictBool = False


class RuleOutcome(Contract):
    record_id: UUID
    rule_id: Text
    rule_version: Text
    status: Literal[
        "passed", "violation", "not_applicable", "insufficient_data", "unavailable_reference"
    ]
    reason: Text
    signal_class: (
        Literal["format_error", "confirmed_contradiction", "statistical_suspicion"] | None
    ) = None
    finding: Finding | None = None

    @model_validator(mode="after")
    def findings_only_for_violations(self) -> Self:
        if (self.status == "violation") != (
            self.finding is not None and self.signal_class is not None
        ):
            raise ValueError("violation_requires_finding_and_class")
        if self.status != "violation" and (
            self.finding is not None or self.signal_class is not None
        ):
            raise ValueError("abstention_or_pass_cannot_have_finding")
        return self
