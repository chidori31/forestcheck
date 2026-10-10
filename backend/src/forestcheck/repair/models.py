"""Versioned, immutable recommendation sidecars; no source mutation or expert label."""

from decimal import Decimal
from typing import Annotated, Literal, Self
from uuid import UUID

from forestcheck.domain.contracts import (
    Contract,
    Digest,
    FieldValue,
    PositiveInt,
    Suggestion,
    Text,
    ValueObject,
)
from forestcheck.rules.models import DecimalText, RuleOutcome, RuleSource
from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

Number = StrictFloat | StrictInt
Method = Literal["arithmetic", "growth", "conditional_model", "baseline", "comparable"]


class DatasetBinding(Contract):
    dataset_id: UUID
    dataset_revision: PositiveInt
    source_sha256: Digest
    mapping_sha256: Digest
    snapshot_sha256: Digest


class PrecisionPolicy(ValueObject):
    field_name: Text
    unit: Text
    step: DecimalText
    source: RuleSource

    @model_validator(mode="after")
    def valid_step(self) -> Self:
        if Decimal(self.step) <= 0:
            raise ValueError("positive_precision_step_required")
        return self


class SourceSupport(ValueObject):
    source_id: Text
    version: Text
    method: Method
    status: Literal["available", "unavailable_reference", "insufficient_data", "not_applicable"]
    basis: Text
    dependency_roots: tuple[Text, ...]
    input_fields: tuple[Text, ...]
    limitations: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def explicit_lineage(self) -> Self:
        if self.status == "available" and not self.dependency_roots:
            raise ValueError("available_source_requires_dependency_root")
        if len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError("duplicate_input_field")
        if len(set(self.dependency_roots)) != len(self.dependency_roots):
            raise ValueError("duplicate_dependency_root")
        return self


class NumericRange(ValueObject):
    lower: Number
    upper: Number

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("unordered_range")
        return self


class Candidate(ValueObject):
    support: SourceSupport
    value: Number | None = None
    lower: Number | None = None
    upper: Number | None = None
    unit: Text
    uncertainty_kind: Literal[
        "uncalibrated_estimate", "calibrated_interval", "deterministic_relation"
    ] = "uncalibrated_estimate"
    uncertainty_basis: Text

    @model_validator(mode="after")
    def usable_value(self) -> Self:
        if self.support.status != "available":
            raise ValueError("candidate_requires_available_support")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("both_interval_bounds_required")
        if self.value is None and self.lower is None:
            raise ValueError("candidate_requires_point_or_interval")
        if self.lower is not None:
            if self.lower > self.upper:
                raise ValueError("unordered_candidate_interval")
            if self.value is not None and not self.lower <= self.value <= self.upper:
                raise ValueError("point_outside_candidate_interval")
        return self


class Uncertainty(ValueObject):
    category: Literal["uncalibrated_estimate", "deterministic_relation", "unavailable"]
    basis: Text


class Explanation(ValueObject):
    observed: Text
    trigger: tuple[Text, ...]
    comparisons: tuple[Text, ...]
    applicability: tuple[Text, ...]
    considered_range: Text
    missing: tuple[Text, ...]
    specialist_actions: tuple[Text, ...]


class UnitHypothesis(ValueObject):
    source_id: Text
    observed_unit: Text | None
    candidate_unit: Text
    status: Literal["unconfirmed"] = "unconfirmed"
    next_action: Text = "Сверить единицу источника; преобразование масштаба не выполнялось."


class SuggestionV2(Suggestion):
    """Explicit v2 extension. Legacy v1 parsing fails rather than dropping evidence."""

    schema_version: Literal["2.0"]
    before: FieldValue | None
    binding: DatasetBinding
    method: Method | None
    evidence: tuple[SourceSupport, ...]
    candidate_range: NumericRange | None = None
    alternatives: tuple[Candidate, ...] = ()
    unit_hypotheses: tuple[UnitHypothesis, ...] = ()
    status: Literal["proposed", "refused", "manual_review", "no_change"]
    refusal_reason: Text | None = None
    limitations: tuple[Text, ...]
    uncertainty: Uncertainty
    support_families: tuple[tuple[Text, ...], ...]
    explanation: Explanation

    @model_validator(mode="after")
    def consistent_proposal(self) -> Self:
        if self.before is not None and self.before.name != self.field_name:
            raise ValueError("before_field_mismatch")
        if self.status in ("refused", "manual_review") and self.proposed_value is not None:
            raise ValueError("refusal_cannot_choose_numeric_value")
        if self.status == "proposed" and type(self.proposed_value) not in (int, float):
            raise ValueError("numeric_proposal_required")
        if self.status == "no_change" and (
            self.before is None
            or self.before.state != "observed"
            or type(self.before.normalized_value) not in (int, float)
            or type(self.proposed_value) not in (int, float)
            or self.proposed_value != self.before.normalized_value
        ):
            raise ValueError("no_change_requires_exact_observed_numeric_value")
        if self.status in ("refused", "manual_review") and self.refusal_reason is None:
            raise ValueError("refusal_reason_required")
        if self.status in ("proposed", "no_change"):
            if (
                self.before is None
                or self.unit != self.before.unit
                or self.refusal_reason is not None
            ):
                raise ValueError("selected_value_requires_matching_before_unit")
            if not any(e.method == self.method and e.status == "available" for e in self.evidence):
                raise ValueError("selected_method_requires_evidence")
            if (
                self.candidate_range is not None
                and not self.candidate_range.lower
                <= self.proposed_value
                <= self.candidate_range.upper
            ):
                raise ValueError("selected_value_outside_range")
        return self


class AggregateRelation(Contract):
    """Caller-confirmed sum of like quantities, never inferred from shared keys."""

    id: Text
    version: Text
    component_record_ids: Annotated[tuple[UUID, ...], Field(min_length=1)]
    component_field: Text
    total_record_id: UUID
    total_field: Text
    unit: Text
    quantity_kind: Text
    component_level: Text
    total_level: Text
    relationship_confirmed: StrictBool
    absolute_tolerance: DecimalText
    source: RuleSource

    @model_validator(mode="after")
    def distinct_members(self) -> Self:
        if len(set(self.component_record_ids)) != len(self.component_record_ids):
            raise ValueError("aggregate_duplicate_component")
        if self.total_record_id in self.component_record_ids:
            raise ValueError("aggregate_cannot_include_own_total")
        return self


class AggregateOutcome(ValueObject):
    relation_id: Text
    relation_version: Text
    status: Literal["passed", "violation", "insufficient_data", "unavailable_reference"]
    reason: Text


class CandidateValidation(Contract):
    before: tuple[RuleOutcome, ...] = ()
    after: tuple[RuleOutcome, ...] = ()
    aggregate_before: tuple[AggregateOutcome, ...] = ()
    aggregate_after: tuple[AggregateOutcome, ...] = ()
    new_violations: tuple[Text, ...] = ()
    degraded_checks: tuple[Text, ...] = ()
    unresolved_violations: tuple[Text, ...] = ()
    limitations: tuple[Text, ...] = ()


class PatchOperation(ValueObject):
    field_name: Text
    before: FieldValue
    after: FieldValue

    @model_validator(mode="after")
    def preserve_original(self) -> Self:
        if self.before.name != self.field_name or self.after.name != self.field_name:
            raise ValueError("patch_field_mismatch")
        if (
            self.before.original_value != self.after.original_value
            or type(self.before.original_value) is not type(self.after.original_value)
            or self.before.source != self.after.source
            or self.before.unit != self.after.unit
        ):
            raise ValueError("patch_must_preserve_raw_source_and_units")
        if self.after.state != "observed" or type(self.after.normalized_value) not in (int, float):
            raise ValueError("patch_requires_observed_numeric_after")
        if self.before == self.after:
            raise ValueError("patch_must_propose_change")
        return self


class ProposedPatch(Contract):
    id: UUID
    status: Literal["proposed_only"] = "proposed_only"
    suggestion_id: UUID
    binding: DatasetBinding
    record_id: UUID
    record_revision: PositiveInt
    record_sha256: Digest
    operation: PatchOperation
    provenance: Annotated[tuple[SourceSupport, ...], Field(min_length=1)]
    validation_sha256: Digest
    integrity_sha256: Digest

    @model_validator(mode="after")
    def integrity(self) -> Self:
        # Integrity is accidental-tamper detection, not authentication of a writable registry.
        from forestcheck.repair.engine import canonical_digest

        if (
            canonical_digest(self.model_dump(mode="json", exclude={"integrity_sha256"}))
            != self.integrity_sha256
        ):
            raise ValueError("patch_integrity_mismatch")
        return self


class RecommendationResult(Contract):
    suggestion: SuggestionV2
    patch: ProposedPatch | None
    validation: CandidateValidation

    @model_validator(mode="after")
    def bound_patch(self) -> Self:
        if (self.suggestion.status == "proposed") != (self.patch is not None):
            raise ValueError("only_proposed_suggestion_has_patch")
        if self.patch is not None:
            from forestcheck.repair.engine import canonical_digest

            if self.patch.validation_sha256 != canonical_digest(
                self.validation.model_dump(mode="json")
            ):
                raise ValueError("patch_validation_mismatch")
            if (
                self.patch.suggestion_id != self.suggestion.id
                or self.patch.record_id != self.suggestion.record_id
                or self.patch.record_revision != self.suggestion.record_revision
                or self.patch.binding != self.suggestion.binding
                or self.patch.operation.before != self.suggestion.before
                or self.patch.operation.after.normalized_value != self.suggestion.proposed_value
                or self.patch.provenance != self.suggestion.evidence
            ):
                raise ValueError("suggestion_patch_mismatch")
        return self


def parse_suggestion(payload: dict) -> Suggestion | SuggestionV2:
    """Dispatch declared versions only; no upgrade or evidence-dropping conversion."""
    version = payload.get("schema_version")
    if version == "1.0":
        return Suggestion.model_validate(payload)
    if version == "2.0":
        return SuggestionV2.model_validate(payload)
    raise ValueError("unsupported_suggestion_version")
