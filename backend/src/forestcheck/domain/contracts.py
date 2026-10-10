"""Immutable v1.0 snapshots. Deliberately reject unknown schemas and fields."""
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

Text = Annotated[str, Field(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]
Scalar = StrictStr | StrictBool | StrictInt | StrictFloat | None


class ValueObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Contract(ValueObject):
    schema_version: Literal["1.0"]


class Entity(Contract):
    id: UUID
    revision: PositiveInt = 1


class SourceCoordinates(ValueObject):
    """Opaque artifact reference, never an absolute filesystem path."""

    artifact_id: UUID
    table: Text
    row: PositiveInt
    column: Text


class BusinessKey(ValueObject):
    """Unconfirmed business dimensions; not a row identifier or uniqueness claim."""

    territory: Text | None = None
    compartment: Text | None = None
    stand: Text
    layer: Text | None = None
    element: Text | None = None


class FieldValue(Contract):
    name: Text
    original_value: Scalar
    normalized_value: Scalar
    unit: Text | None
    state: Literal["observed", "missing", "not_applicable", "parse_error", "unknown"]
    source: SourceCoordinates

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        if (self.state == "observed") != (self.normalized_value is not None):
            raise ValueError("only_observed_has_normalized_value")
        return self


class Dataset(Entity):
    name: Text
    created_at: AwareDatetime
    analytical_format: Literal["parquet"] = "parquet"
    artifact_id: UUID | None = None


class Record(Entity):
    dataset_id: UUID
    dataset_revision: PositiveInt = 1
    business_key: BusinessKey | None = None
    fields: tuple[FieldValue, ...]

    @model_validator(mode="after")
    def unique_fields(self) -> Self:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("duplicate_field_name")
        return self


class EvaluationLabel(Contract):
    """Evaluation-only sidecar; never embedded into Record fields."""

    record_id: UUID
    record_revision: PositiveInt = 1
    field_name: Text
    is_error: StrictBool | None
    expected_value: Scalar = None
    split: Literal["train", "validation", "test"]


class Evidence(ValueObject):
    code: Text
    message: Text
    source: SourceCoordinates | None = None
    reference_id: Text | None = None


class AnomalyScore(ValueObject):
    """A raw score on a named scale, NOT a probability or calibrated confidence."""

    value: StrictFloat | StrictInt
    scale: Text
    higher_is_worse: StrictBool


class Finding(Entity):
    record_id: UUID
    record_revision: PositiveInt = 1
    analysis_run_id: UUID
    field_name: Text | None = None
    signal_kind: Literal["missing", "parse_error", "inconsistency", "anomaly", "reference_mismatch"]
    severity: Literal["info", "warning", "error", "critical"]
    priority: Annotated[int, Field(strict=True, ge=0, le=3)]
    evidence: Annotated[tuple[Evidence, ...], Field(min_length=1)]
    review_status: Literal["unreviewed", "in_review", "reviewed"]
    anomaly_score: AnomalyScore | None = None


class Suggestion(Entity):
    finding_id: UUID
    record_id: UUID
    record_revision: PositiveInt = 1
    field_name: Text
    proposed_value: Scalar
    unit: Text | None
    rationale: Text
    model_version_id: UUID | None = None


class Review(Entity):
    finding_id: UUID
    finding_revision: PositiveInt = 1
    reviewer_id: Text
    reviewed_at: AwareDatetime
    error_assessment: Literal["confirmed_error", "no_error", "uncertain"]
    suggestion_id: UUID | None = None
    suggestion_decision: Literal["not_considered", "accepted", "rejected", "deferred"]
    note: Text | None = None

    @model_validator(mode="after")
    def independent_decisions(self) -> Self:
        if self.suggestion_decision != "not_considered" and self.suggestion_id is None:
            raise ValueError("suggestion_reference_required")
        if self.suggestion_decision == "accepted" and self.error_assessment != "confirmed_error":
            raise ValueError("accepted_correction_requires_confirmed_error")
        return self


class AnalysisRun(Entity):
    dataset_id: UUID
    dataset_revision: PositiveInt = 1
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    config_digest: Digest
    model_version_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def valid_lifecycle(self) -> Self:
        if self.status in ("running", "completed") and self.started_at is None:
            raise ValueError("started_at_required")
        if self.status in ("completed", "failed", "cancelled") and self.finished_at is None:
            raise ValueError("finished_at_required")
        if self.status == "pending" and (self.started_at or self.finished_at):
            raise ValueError("pending_run_has_no_execution_times")
        if self.status == "running" and self.finished_at:
            raise ValueError("running_run_not_finished")
        if self.started_at and self.started_at < self.created_at:
            raise ValueError("invalid_start_time")
        if self.finished_at and self.finished_at < (self.started_at or self.created_at):
            raise ValueError("invalid_finish_time")
        return self


class ModelVersion(Entity):
    """Descriptor only. Does not load, train or execute a model."""

    name: Text
    model_version: Text
    created_at: AwareDatetime
    artifact_id: UUID
    artifact_sha256: Digest
    feature_schema_version: Literal["1.0"]
