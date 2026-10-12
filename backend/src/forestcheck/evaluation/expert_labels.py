"""Validate future expert answers as sidecars; no training, scoring or record edits.

The caller supplies the trusted snapshot and a declared history of exposure.
Identifiers and hashes verify those bindings, not the honesty of expert answers
or the completeness of as-yet unknown spatial/temporal relationships.
"""

from collections import defaultdict
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, model_validator

from forestcheck.domain.contracts import (
    Contract,
    Digest,
    PositiveInt,
    Record,
    Scalar,
    SourceCoordinates,
    Suggestion,
    Text,
)
from forestcheck.evaluation.annotation import AnnotationForm
from forestcheck.evaluation.split import Role, SplitIdentity, SplitManifest, hash_value
from forestcheck.repair.models import DatasetBinding, SuggestionV2

Count = Annotated[int, Field(strict=True, ge=0)]
Purpose = Literal["dev", "review_only", "independent_evaluation"]
SemanticChange = Literal["height_semantics", "bundle_deduplication", "features"]
BoundSuggestion = Annotated[Suggestion | SuggestionV2, Field(discriminator="schema_version")]


class CorrectValue(Contract):
    """None as value is an explicitly supplied null, not an absent answer."""

    value: Scalar
    unit: Text | None


class ExpertLabel(AnnotationForm):
    """Existing independent decisions plus an exact source/before-value binding."""

    field_name: Text
    object_id: Digest
    source: SourceCoordinates
    original_value: Scalar
    error_kind: Text | None = None
    reviewer_id: Text
    note: Text
    correct_value: CorrectValue | None = None

    @model_validator(mode="after")
    def error_has_kind(self) -> Self:
        if self.error_assessment == "confirmed_error" and self.error_kind is None:
            raise ValueError("confirmed_error_kind_required")
        return self


class ExpertLabelBatch(Contract):
    import_version: Literal["expert-labels-1.0.0"]
    dataset_id: UUID
    dataset_revision: PositiveInt
    snapshot_sha256: Digest
    preprocessing_version: Text
    split_digest: Digest
    purpose: Purpose
    labels: Annotated[tuple[ExpertLabel, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def distinct_annotations(self) -> Self:
        ids = [label.id for label in self.labels]
        cells = [(label.record_id, label.field_name) for label in self.labels]
        if len(set(ids)) != len(ids) or len(set(cells)) != len(cells):
            raise ValueError("duplicate_expert_annotation")
        return self


class ExpertLabelSnapshot(Contract):
    """Trusted, explicitly selected scope. Its digest binds records and mappings.

    This is a sidecar adapter around Record and SplitIdentity, not a new import
    format. File content hashes come from the verified importer receipt; this
    module performs no filesystem access and cannot recompute source files.
    """

    dataset_id: UUID
    dataset_revision: PositiveInt
    preprocessing_version: Text
    source_sha256: dict[UUID, Digest]
    records: Annotated[tuple[Record, ...], Field(min_length=1)]
    identities: Annotated[tuple[SplitIdentity, ...], Field(min_length=1)]
    dataset_bindings: tuple[DatasetBinding, ...] = ()
    suggestions: tuple[BoundSuggestion, ...] = ()

    @model_validator(mode="after")
    def complete_binding(self) -> Self:
        records = {row.id: row for row in self.records}
        identities = {item.record_id: item for item in self.identities}
        if (len(records) != len(self.records) or len(identities) != len(self.identities)
                or records.keys() != identities.keys()):
            raise ValueError("snapshot_record_identity_binding")
        for row in self.records:
            if (row.dataset_id != self.dataset_id or row.dataset_revision != self.dataset_revision
                    or identities[row.id].dataset_id != self.dataset_id):
                raise ValueError("snapshot_dataset_binding")
            if any(value.source.artifact_id not in self.source_sha256 for value in row.fields):
                raise ValueError("snapshot_source_hash_required")
        seen = set()
        for suggestion in self.suggestions:
            row = records.get(suggestion.record_id)
            if (suggestion.id in seen or row is None or suggestion.record_revision != row.revision
                    or suggestion.field_name not in {value.name for value in row.fields}):
                raise ValueError("snapshot_suggestion_binding")
            if isinstance(suggestion, SuggestionV2):
                value = next(value for value in row.fields if value.name == suggestion.field_name)
                if (suggestion.binding not in self.dataset_bindings
                        or suggestion.binding.dataset_id != self.dataset_id
                        or suggestion.binding.dataset_revision != self.dataset_revision
                        or suggestion.binding.source_sha256 != self.source_sha256[value.source.artifact_id]
                        or suggestion.before is None
                        or hash_value(suggestion.before.model_dump(mode="json"))
                        != hash_value(value.model_dump(mode="json"))):
                    raise ValueError("snapshot_v2_suggestion_binding")
            seen.add(suggestion.id)
        return self

    @property
    def digest(self) -> str:
        return hash_value(self.model_dump(mode="json"))


class ObjectLink(Contract):
    """A separately established relation, e.g. a confirmed complete-bundle copy."""

    object_id: Digest
    related_record_id: UUID
    basis: Text


class ExpertLabelExposure(Contract):
    split_digest: Digest
    opened_holdout_groups: tuple[Digest, ...]
    tuning_identities: tuple[SplitIdentity, ...] = ()
    object_links: tuple[ObjectLink, ...] = ()
    unresolved_linkage: StrictBool = True


class LabelOverlap(Contract):
    record_id: UUID
    object_id: Digest
    group_id: Digest
    roles: tuple[Role, ...]
    related_record_ids: tuple[UUID, ...]
    tuning_overlap: StrictBool
    opened_holdout: StrictBool


class ExpertLabelImportReport(Contract):
    import_version: Literal["expert-labels-1.0.0"] = "expert-labels-1.0.0"
    dataset_id: UUID
    dataset_revision: PositiveInt
    preprocessing_version: Text
    snapshot_sha256: Digest
    split_digest: Digest
    batch_sha256: Digest
    purpose: Purpose
    overlaps: tuple[LabelOverlap, ...]
    independent_evaluation_eligible: StrictBool
    labelled_fields: Count
    confirmed_errors: Count
    confirmed_no_errors: Count
    uncertain_or_unassessed: Count
    unlabelled_fields: Count
    metric_status: Literal["NOT_RUN"] = "NOT_RUN"
    limitations: tuple[Text, ...]


class ExpertLabelImport(Contract):
    labels: tuple[ExpertLabel, ...]
    report: ExpertLabelImportReport


def _exposure_components(snapshot, manifest, exposure):
    """Undirected closure of every declared dependency, including historic aliases.

    Indexed union/find avoids comparing every frozen record with every label.
    A shared record, parent, group or copy identifies a dependent component;
    explicit object links join components in either orientation and any order.
    """
    population = (*snapshot.identities, *manifest.assignments, *exposure.tuning_identities)
    parents = {item.record_id: item.record_id for item in population}
    sizes = dict.fromkeys(parents, 1)
    owners = {}

    def find(record_id):
        while parents[record_id] != record_id:
            parents[record_id] = parents[parents[record_id]]
            record_id = parents[record_id]
        return record_id

    def join(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parents[right] = left
        sizes[left] += sizes[right]

    for item in population:
        for field in ("parent_id", "group_id", "duplicate_id"):
            value = getattr(item, field)
            if value is not None:
                owner = owners.setdefault((field, value), item.record_id)
                join(item.record_id, owner)
    for link in exposure.object_links:
        owner = owners.setdefault(("parent_id", link.object_id), link.related_record_id)
        join(owner, link.related_record_id)
    return {record_id: find(record_id) for record_id in parents}


def import_expert_labels(
    batch: ExpertLabelBatch,
    *,
    snapshot: ExpertLabelSnapshot,
    manifest: SplitManifest,
    exposure: ExpertLabelExposure,
) -> ExpertLabelImport:
    """Validate the complete batch before returning any answers; never fill gaps.

    `dev` cannot consume held-out objects. Independent evaluation rejects known
    training, development, calibration and opened-holdout overlaps. A separate
    review-only import can document these answers without changing any split.
    """
    # Revalidate even model_copy/model_construct input; nested Contract objects
    # are immutable by normal API but their dict members need not be.
    batch = ExpertLabelBatch.model_validate(batch.model_dump(mode="json"))
    snapshot = ExpertLabelSnapshot.model_validate(snapshot.model_dump(mode="json"))
    manifest = SplitManifest.model_validate(manifest.model_dump(mode="json"))
    exposure = ExpertLabelExposure.model_validate(exposure.model_dump(mode="json"))
    if (batch.dataset_id != snapshot.dataset_id
            or batch.dataset_revision != snapshot.dataset_revision
            or batch.snapshot_sha256 != snapshot.digest
            or batch.preprocessing_version != snapshot.preprocessing_version
            or batch.split_digest != manifest.digest or exposure.split_digest != manifest.digest):
        raise ValueError("expert_label_batch_binding_mismatch")
    assignments = {item.record_id: item for item in manifest.assignments}
    holdout_groups = {item.group_id for item in manifest.assignments if item.role == "holdout"}
    opened_holdout_groups = set(exposure.opened_holdout_groups)
    prior_exposed_groups = set(manifest.prior_exposed_groups)
    if not opened_holdout_groups <= holdout_groups:
        raise ValueError("unknown_opened_holdout_group")
    if any(link.related_record_id not in assignments for link in exposure.object_links):
        raise ValueError("unknown_related_split_record")
    components = _exposure_components(snapshot, manifest, exposure)
    assignments_by_component = defaultdict(list)
    for item in manifest.assignments:
        assignments_by_component[components[item.record_id]].append(item)
    tuning_components = {components[item.record_id] for item in exposure.tuning_identities}
    tuning_components.update(
        components[item.record_id] for item in manifest.assignments
        if item.group_id in prior_exposed_groups
    )
    records = {row.id: row for row in snapshot.records}
    identities = {item.record_id: item for item in snapshot.identities}
    suggestions = {item.id: item for item in snapshot.suggestions}
    overlaps = []
    for label in batch.labels:
        row = records.get(label.record_id)
        if row is None or row.revision != label.record_revision:
            raise ValueError("expert_label_record_binding_mismatch")
        identity = identities[row.id]
        existing = assignments.get(row.id)
        if existing is not None and any(
            getattr(existing, key) != getattr(identity, key)
            for key in ("dataset_id", "group_id", "parent_id", "duplicate_id")
        ):
            raise ValueError("expert_label_split_identity_binding_mismatch")
        if identity.parent_id != label.object_id:
            raise ValueError("expert_label_object_binding_mismatch")
        value = next((value for value in row.fields if value.name == label.field_name), None)
        if value is None:
            raise ValueError("expert_label_field_binding_mismatch")
        if label.source != value.source:
            raise ValueError("expert_label_source_binding_mismatch")
        if (type(label.original_value) is not type(value.original_value)
                or label.original_value != value.original_value):
            raise ValueError("expert_label_before_value_mismatch")
        if label.correct_value is not None and label.correct_value.unit != value.unit:
            raise ValueError("expert_label_correct_value_unit_mismatch")
        if label.suggestion_id is not None:
            suggestion = suggestions.get(label.suggestion_id)
            if (suggestion is None or suggestion.record_id != row.id
                    or suggestion.record_revision != row.revision
                    or suggestion.field_name != value.name or suggestion.unit != value.unit):
                raise ValueError("expert_label_suggestion_binding_mismatch")
        component = components[row.id]
        related = assignments_by_component[component]
        roles = tuple(sorted({item.role for item in related}))
        tuning = (any(role in {"train", "tune", "development_cv"} for role in roles)
                  or component in tuning_components)
        opened = any(item.group_id in opened_holdout_groups for item in related)
        if batch.purpose == "dev" and "holdout" in roles:
            raise ValueError("holdout_cannot_be_reassigned_to_development")
        if batch.purpose == "dev" and "calibration" in roles:
            raise ValueError("calibration_cannot_be_reassigned_to_development")
        if batch.purpose == "independent_evaluation" and (
            tuning or opened or "calibration" in roles or exposure.unresolved_linkage
        ):
            raise ValueError("independent_evaluation_conflicts_with_exposure_or_links")
        overlaps.append(LabelOverlap(
            schema_version="1.0", record_id=row.id, object_id=identity.parent_id,
            group_id=identity.group_id, roles=roles,
            related_record_ids=tuple(sorted((item.record_id for item in related), key=str)),
            tuning_overlap=tuning, opened_holdout=opened,
        ))
    positive = sum(label.error_assessment == "confirmed_error" for label in batch.labels)
    negative = sum(label.error_assessment == "no_error" for label in batch.labels)
    report = ExpertLabelImportReport(
        schema_version="1.0", snapshot_sha256=snapshot.digest, split_digest=manifest.digest,
        dataset_id=snapshot.dataset_id, dataset_revision=snapshot.dataset_revision,
        preprocessing_version=snapshot.preprocessing_version,
        batch_sha256=hash_value(batch.model_dump(mode="json")), purpose=batch.purpose,
        overlaps=tuple(overlaps),
        independent_evaluation_eligible=batch.purpose == "independent_evaluation",
        labelled_fields=len(batch.labels), confirmed_errors=positive, confirmed_no_errors=negative,
        uncertain_or_unassessed=len(batch.labels) - positive - negative,
        unlabelled_fields=sum(len(row.fields) for row in snapshot.records) - len(batch.labels),
        limitations=(
            "Import checks declared identity and exposure; it does not verify expert truth.",
            "Unlabelled fields are not negative answers; metrics and training are NOT_RUN.",
            "Recall on labelled errors is not whole-dataset recall; partial labels do not establish precision.",
            "Eligibility applies only to declared links and exposure, not proven spatial/temporal independence.",
            "Correct values and accepted suggestions are not applied and do not become training labels.",
        ),
    )
    return ExpertLabelImport(schema_version="1.0", labels=batch.labels, report=report)


class PreprocessingArtifactBinding(Contract):
    artifact_id: Text
    artifact_kind: Literal["model", "result"]
    preprocessing_version: Text
    snapshot_sha256: Digest
    mapping_sha256: Digest


class ArtifactCompatibility(Contract):
    artifact_id: Text
    artifact_kind: Literal["model", "result"]
    status: Literal["compatible", "incompatible"]
    reasons: tuple[Text, ...]


class PreprocessingCompatibilityReport(Contract):
    preprocessing_version: Text
    snapshot_sha256: Digest
    mapping_sha256: Digest
    changes: tuple[SemanticChange, ...]
    decisions: tuple[ArtifactCompatibility, ...]
    action: Literal["bindings_unchanged", "preserve_old_artifacts_new_experiment_required"]
    limitations: tuple[Text, ...]


def report_preprocessing_compatibility(
    artifacts,
    *,
    preprocessing_version: str,
    snapshot_sha256: str,
    mapping_sha256: str,
    changes: tuple[SemanticChange, ...] = (),
) -> PreprocessingCompatibilityReport:
    """Report compatibility of declared bindings without opening or changing artifacts.

    Semantic changes require a new version. Old models/results, train/test rows,
    and split files are preserved; this report never authorizes loading a model.
    """
    bindings = tuple(PreprocessingArtifactBinding.model_validate(
        item.model_dump(mode="json") if isinstance(item, PreprocessingArtifactBinding) else item
    ) for item in artifacts)
    target = PreprocessingArtifactBinding(
        schema_version="1.0", artifact_id="requested-binding", artifact_kind="model",
        preprocessing_version=preprocessing_version, snapshot_sha256=snapshot_sha256,
        mapping_sha256=mapping_sha256,
    )
    if len({item.artifact_id for item in bindings}) != len(bindings):
        raise ValueError("duplicate_artifact_binding")
    if changes and any(item.preprocessing_version == target.preprocessing_version for item in bindings):
        raise ValueError("semantic_change_requires_new_preprocessing_version")
    decisions = []
    for item in bindings:
        reasons = tuple(f"{key}_changed" for key in (
            "preprocessing_version", "snapshot_sha256", "mapping_sha256"
        ) if getattr(item, key) != getattr(target, key))
        decisions.append(ArtifactCompatibility(
            schema_version="1.0", artifact_id=item.artifact_id, artifact_kind=item.artifact_kind,
            status="incompatible" if reasons else "compatible", reasons=reasons,
        ))
    changed = bool(changes) or any(item.status == "incompatible" for item in decisions)
    return PreprocessingCompatibilityReport(
        schema_version="1.0", preprocessing_version=target.preprocessing_version,
        snapshot_sha256=target.snapshot_sha256, mapping_sha256=target.mapping_sha256,
        changes=changes, decisions=tuple(decisions),
        action="preserve_old_artifacts_new_experiment_required" if changed else "bindings_unchanged",
        limitations=(
            "Declared metadata comparison only; trusted registry loading still verifies native artifacts.",
            "No split, model, result or source is modified; training and new evaluation are NOT_RUN.",
            "Confirmed copies across prior partitions limit old results; never remove convenient test rows.",
        ),
    )
