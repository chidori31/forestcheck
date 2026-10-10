"""Blank manual-review forms with a separate, auditable sampling sidecar."""

from hashlib import sha256
from typing import Annotated, Literal, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_validator

from forestcheck.domain.contracts import Contract, Digest, PositiveInt, Text
from forestcheck.evaluation.split import SplitManifest


class AnnotationForm(Contract):
    id: UUID
    record_id: UUID
    record_revision: PositiveInt = 1
    field_name: Text | None = None
    suggestion_id: UUID | None = None
    error_assessment: Literal["confirmed_error", "no_error", "uncertain"] | None = None
    correction_assessment: Literal["correct", "incorrect", "uncertain", "not_applicable"] | None = (
        None
    )
    suggestion_decision: Literal["accepted", "rejected", "deferred", "not_considered"] | None = None
    reviewer_id: Text | None = None
    note: Text | None = None

    @model_validator(mode="after")
    def independent_decisions(self) -> Self:
        assessed_fix = self.correction_assessment not in {None, "not_applicable"}
        decided_fix = self.suggestion_decision not in {None, "not_considered"}
        if (assessed_fix or decided_fix) and self.suggestion_id is None:
            raise ValueError("suggestion_reference_required")
        if (
            any((self.error_assessment, self.correction_assessment, self.suggestion_decision))
            and not self.reviewer_id
        ):
            raise ValueError("human_reviewer_required")
        if self.suggestion_decision == "accepted" and (
            self.error_assessment != "confirmed_error" or self.correction_assessment != "correct"
        ):
            raise ValueError("accepted_requires_confirmed_error_and_correct_fix")
        return self


class SamplingEntry(Contract):
    record_id: UUID
    group_id: Digest
    stratum: Literal["flagged", "control"]
    population_size: Annotated[int, Field(strict=True, ge=1)]
    sample_size: Annotated[int, Field(strict=True, ge=1)]
    inclusion_probability: Annotated[float, Field(strict=True, gt=0, le=1)]


class AnnotationQueue(Contract):
    seed: Text
    split_digest: Digest
    forms: tuple[AnnotationForm, ...]
    sampling: tuple[SamplingEntry, ...]
    expert_status: Literal["not_assigned"] = "not_assigned"


def _order(ids: set[UUID], *, seed: str, stratum: str) -> list[UUID]:
    return sorted(ids, key=lambda item: sha256(f"{seed}:{stratum}:{item}".encode()).digest())


def build_annotation_queue(
    candidates: tuple[UUID, ...],
    *,
    flagged: tuple[UUID, ...],
    manifest: SplitManifest,
    seed: str,
    control_count: int,
    max_flagged: int,
) -> AnnotationQueue:
    """Select flagged cases and random unflagged controls, without answering forms.

    Sampling strata stay out of the form to support blind review. Controls are
    unflagged observations, never assumed correct. No holdout queue at stage 5.
    """
    if not seed or type(control_count) is not int or type(max_flagged) is not int:
        raise ValueError("invalid_sampling_parameters")
    if control_count < 1 or max_flagged < 1:
        raise ValueError("positive_control_and_flagged_budgets_required")
    pool, flags = set(candidates), set(flagged)
    if len(pool) != len(candidates) or len(flags) != len(flagged) or not flags <= pool:
        raise ValueError("invalid_candidate_ids")
    assignments = {
        item: manifest.assert_allowed(
            item, roles={"train", "tune", "calibration", "development_cv"}
        )
        for item in pool
    }
    controls = pool - flags
    if len(controls) < control_count:
        raise ValueError("requested_controls_required")
    split_digest = manifest.digest
    forms, sampling = [], []
    for stratum, population, budget in (
        ("flagged", flags, max_flagged),
        ("control", controls, control_count),
    ):
        selected = _order(population, seed=seed, stratum=stratum)[:budget]
        for record_id in selected:
            forms.append(
                AnnotationForm(
                    schema_version="1.0",
                    id=uuid5(
                        NAMESPACE_URL, f"forestcheck:annotation:{split_digest}:{seed}:{record_id}"
                    ),
                    record_id=record_id,
                )
            )
            sampling.append(
                SamplingEntry(
                    schema_version="1.0",
                    record_id=record_id,
                    group_id=assignments[record_id].group_id,
                    stratum=stratum,
                    population_size=len(population),
                    sample_size=len(selected),
                    inclusion_probability=len(selected) / len(population),
                )
            )
    # Hide sampling order as well as its stratum metadata from the review forms.
    forms.sort(key=lambda item: sha256(f"{seed}:blind:{item.id}".encode()).digest())
    return AnnotationQueue(
        schema_version="1.0",
        seed=seed,
        split_digest=split_digest,
        forms=tuple(forms),
        sampling=tuple(sampling),
    )
