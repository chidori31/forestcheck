"""Workflow sidecars v1; existing domain/import/rule/model contracts remain intact."""

from typing import Annotated, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import Field, StrictBool, model_validator

from forestcheck.domain.contracts import (
    AnalysisRun,
    Contract,
    Digest,
    FieldValue,
    Finding,
    Review,
    Text,
)
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.ml.data import MLPlan
from forestcheck.repair.models import AggregateRelation, PrecisionPolicy, RecommendationResult
from forestcheck.rules.models import Codebook, RuleDefinition

Version = Annotated[int, Field(strict=True, ge=0)]


class TrustedModel(Contract):
    model_id: UUID
    target: Text
    registry: Text  # Local administrator only; never an HTTP input or response.
    run_id: UUID
    train_sha256: Digest
    metadata_sha256: Digest


class AnalysisProfile(Contract):
    id: UUID
    name: Text
    version: Text
    mapping_sha256: Digest
    record_level: Text
    category: Text | None = None
    source_kind: Literal["real_observation", "synthetic"]
    rules: Annotated[tuple[RuleDefinition, ...], Field(min_length=1)]
    codebooks: tuple[Codebook, ...] = ()
    numeric_fields: dict[str, str] = {}
    precision: tuple[PrecisionPolicy, ...] = ()
    aggregates: tuple[AggregateRelation, ...] = ()
    models: tuple[TrustedModel, ...] = ()
    ml_plan: MLPlan | None = None
    split_manifest_path: Text | None = None
    reference_parquet_path: Text | None = None
    allowed_record_ids: tuple[UUID, ...] | None = None
    limitations: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def bound_models(self):
        if len({r.id for r in self.rules}) != len(self.rules):
            raise ValueError("duplicate_rules")
        if len({p.field_name for p in self.precision}) != len(self.precision):
            raise ValueError("duplicate_precision")
        if self.models:
            if not self.ml_plan or not self.allowed_record_ids or not self.split_manifest_path:
                raise ValueError("models_require_bound_plan_and_cohort")
            if self.ml_plan.mapping_sha256 != self.mapping_sha256:
                raise ValueError("model_mapping_mismatch")
            if self.ml_plan.source_kind != self.source_kind:
                raise ValueError("model_source_kind_mismatch")
            if len({m.model_id for m in self.models}) != len(self.models):
                raise ValueError("duplicate_models")
        return self


class MappingRequest(Contract):
    expected_version: Version
    mapping: ImportMapping


class AnalysisRequest(Contract):
    version_id: UUID
    profile_id: UUID
    use_models: StrictBool = True


class ReviewRequest(Contract):
    finding_id: UUID
    expected_version: Version
    action: Literal["decide", "revoke"] = "decide"
    error_assessment: Literal["confirmed_error", "no_error", "uncertain"]
    suggestion_id: UUID | None = None
    suggestion_decision: Literal["not_considered", "accepted", "rejected", "deferred"]
    comment: Annotated[str, Field(max_length=2000)] | None = None
    revoke_event_id: UUID | None = None

    @model_validator(mode="after")
    def independent_decisions(self):
        if self.suggestion_decision != "not_considered" and self.suggestion_id is None:
            raise ValueError("suggestion_required")
        if self.suggestion_decision == "accepted" and self.error_assessment != "confirmed_error":
            raise ValueError("confirmed_error_required")
        if (self.action == "revoke") != (self.revoke_event_id is not None):
            raise ValueError("revocation_reference_required")
        if self.action == "revoke" and self.suggestion_decision == "accepted":
            raise ValueError("revocation_cannot_accept")
        return self


class VersionRequest(Contract):
    base_version_id: UUID
    analysis_id: UUID
    expected_version: Version
    review_event_ids: Annotated[tuple[UUID, ...], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def unique_events(self):
        if len(set(self.review_event_ids)) != len(self.review_event_ids):
            raise ValueError("duplicate_review_event")
        return self


class ExportRequest(Contract):
    version_id: UUID
    kind: Literal["original", "report", "corrected"]
    format: Literal["json", "csv", "xlsx"]
    analysis_id: UUID | None = None


class SchemaRequest(Contract):
    encoding: Literal["utf-8", "utf-8-sig", "cp1251"]
    delimiter: Literal[";", ",", "\t", "|"]


T = TypeVar("T")


class Page(Contract, Generic[T]):
    items: list[T]
    total: int
    offset: int
    limit: int
    next_offset: int | None


class DatasetView(Contract):
    id: UUID
    name: str
    state: str
    etag: int
    current_version_id: UUID | None
    created_at: float


class FindingItem(Contract):
    schema_version: Literal["1.0"] = "1.0"
    finding: Finding
    review_version: int
    latest_review: dict | None = None


class RecordQuery(Contract):
    record_ids: Annotated[tuple[UUID, ...], Field(min_length=1, max_length=200)]


class FindingDetail(FindingItem):
    suggestions: list[RecommendationResult]


class SchemaView(Contract):
    dataset_id: UUID
    headers: list[str]
    encoding: str
    delimiter: str
    mapping: ImportMapping | None
    import_report: dict | None
    units_inferred: Literal[False]
    mapping_sha256: str | None = None


JobKind = Literal["import", "analysis", "version", "export"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class QueuedJob(Contract):
    id: UUID
    kind: JobKind
    status: Literal["queued"]


class JobView(Contract):
    id: UUID
    kind: JobKind
    status: JobStatus
    operator: str
    request: dict
    request_sha256: Digest
    result: dict | None
    attempts: int
    max_attempts: int
    cancel_requested: Literal[0, 1]
    error_code: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    progress_percent: None
    analysis_run: AnalysisRun | None = None


class ReviewEvent(Contract):
    id: UUID
    finding_id: UUID
    sequence: int
    operator: str
    action: Literal["decide", "revoke"]
    revoke_event_id: UUID | None
    review: Review
    before: FieldValue | None
    after: FieldValue | None
    analysis_id: UUID
    version_id: UUID
