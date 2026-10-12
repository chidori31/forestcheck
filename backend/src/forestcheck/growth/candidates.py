"""Immutable bibliographic candidates, deliberately separate from admitted tables.

Statuses record supplied review evidence; they do not verify the truth of that
evidence or authorize numerical use. There is no conversion to GrowthTable,
lookup, interpolation, scaling, file loading or automatic registry admission.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from forestcheck.domain.contracts import Contract, Digest, ValueObject
from forestcheck.growth.registry import AgeRange, Number, ReferenceText, SourceReference


class CandidateIndicator(ValueObject):
    name: ReferenceText
    meaning: ReferenceText | None = None
    unit: ReferenceText | None = None


class GrowthReferenceCandidate(Contract):
    candidate_id: ReferenceText
    candidate_version: ReferenceText
    source: SourceReference
    public_url: Annotated[str, Field(strict=True, pattern=r"^https?://[^\s]+$")]
    source_access: Literal["tables_inspected", "unavailable_text"]
    inspected_locations: tuple[ReferenceText, ...] = ()
    species: tuple[ReferenceText, ...] = ()
    regions: tuple[ReferenceText, ...] = ()
    origin: ReferenceText | None = None
    stand_form: Literal["normal", "modal", "other", "unknown"] = "unknown"
    purity: Literal["pure", "mixed", "unknown"] = "unknown"
    fullness_value: Number | None = None
    fullness_definition: ReferenceText | None = None
    bonitet: tuple[ReferenceText, ...] = ()
    conditions: ReferenceText | None = None
    age_range: AgeRange | None = None
    indicators: tuple[CandidateIndicator, ...] = ()
    unresolved: tuple[ReferenceText, ...] = ()
    status: Literal["found", "applicability_checked", "expert_confirmed"] = "found"
    checked_scope_sha256: Digest | None = None
    applicability_evidence: SourceReference | None = None
    expert_confirmation: SourceReference | None = None

    @model_validator(mode="after")
    def evidence_matches_claims(self) -> Self:
        inspected = self.source_access == "tables_inspected"
        if inspected != bool(self.inspected_locations):
            raise ValueError("table_inspection_requires_exact_locations")
        if self.fullness_value is not None and self.fullness_definition is None:
            raise ValueError("fullness_requires_stated_definition")
        checked = self.status != "found"
        if checked and (
            not inspected or self.checked_scope_sha256 is None
            or self.applicability_evidence is None or self.unresolved
        ):
            raise ValueError("checked_status_requires_resolved_scope_and_evidence")
        if not checked and (
            self.checked_scope_sha256 is not None or self.applicability_evidence is not None
        ):
            raise ValueError("applicability_evidence_requires_checked_status")
        if (self.status == "expert_confirmed") != (self.expert_confirmation is not None):
            raise ValueError("expert_status_requires_explicit_confirmation")
        return self


class GrowthCandidateRegistry(Contract):
    """A versioned candidate catalogue; never an operational reference registry."""

    candidates: tuple[GrowthReferenceCandidate, ...] = ()

    @model_validator(mode="after")
    def unique_versions(self) -> Self:
        identities = [(item.candidate_id, item.candidate_version) for item in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate_candidate_version")
        return self
