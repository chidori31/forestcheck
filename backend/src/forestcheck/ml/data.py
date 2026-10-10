"""Verified local mappings and frozen cohorts. Targets are never feature columns."""

import json
import math
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from forestcheck.domain.contracts import Contract, Digest, Record, Text
from forestcheck.evaluation.split import SplitManifest, hash_value
from forestcheck.ingest.csv_import import checked_private, checked_source, regular_output, sha256
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.rules.local import read_private_json
from pydantic import Field, model_validator

Alias = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class FeatureBinding(Contract):
    name: Alias
    source_field: Text
    kind: Literal["numeric", "categorical", "context"]
    unit: Text | None
    provenance: Text
    dependency_status: Literal[
        "independent_measurement_confirmed", "unverified_measurement_procedure"
    ]


class TargetBinding(Contract):
    name: Alias
    source_field: Text
    unit: Text
    provenance: Text


class MLPlan(Contract):
    plan_version: Literal["ml-plan-1.0.0"]
    seed: Annotated[int, Field(strict=True, ge=0)]
    table: Alias
    dataset_id: UUID
    mapping_version: Text
    mapping_sha256: Digest
    split_sha256: Digest
    snapshot_sha256: Digest
    source_sha256: Digest
    source_kind: Literal["real_observation", "synthetic"]
    features: tuple[FeatureBinding, ...]
    targets: Annotated[tuple[TargetBinding, ...], Field(min_length=1, max_length=3)]
    excluded_fields: tuple[Text, ...]
    exclusion_rationale: Text
    train_folds: dict[Digest, Annotated[int, Field(strict=True, ge=0, le=2)]]
    calibration_fit_groups: tuple[Digest, ...]
    calibration_check_groups: tuple[Digest, ...]
    iterations: Annotated[int, Field(strict=True, ge=1, le=120)] = 120
    cpu_threads: Literal[2] = 2
    wall_seconds: Literal[900] = 900
    memory_mib: Literal[4096] = 4096
    queue_depth: Annotated[int, Field(strict=True, ge=1)] = 40
    limitations: tuple[Text, ...]

    @model_validator(mode="after")
    def no_leakage(self):
        names = [f.name for f in self.features]
        fields = [f.source_field for f in self.features]
        targets = {t.source_field for t in self.targets}
        if len(names) != len(set(names)) or len(fields) != len(set(fields)):
            raise ValueError("duplicate_predictor")
        if len(targets) != len(self.targets) or len({t.name for t in self.targets}) != len(
            self.targets
        ):
            raise ValueError("duplicate_target")
        if set(fields) & (targets | set(self.excluded_fields)) or set(names) & {
            t.name for t in self.targets
        }:
            raise ValueError("target_or_excluded_source")
        if not any(f.kind == "numeric" for f in self.features):
            raise ValueError("independent_numeric_input_required")
        if any(f.kind == "numeric" and f.unit is None for f in self.features):
            raise ValueError("numeric_unit_required")
        fit, check = set(self.calibration_fit_groups), set(self.calibration_check_groups)
        if not fit or not check or fit & check or (fit | check) & self.train_folds.keys():
            raise ValueError("calibration_group_overlap_or_empty")
        if set(self.train_folds.values()) != {0, 1, 2}:
            raise ValueError("three_nonempty_train_folds_required")
        return self

    @property
    def digest(self):
        return hash_value(self.model_dump(mode="json"))


class MLSample(Contract):
    record_id: UUID
    group_id: Digest
    role: Literal["train", "tune", "calibration"]
    features: dict
    targets: dict
    target_states: dict


def freeze_subsets(manifest: SplitManifest, *, seed: int):
    if manifest.strategy == "group_cv":
        raise ValueError("nested_cv_training_not_implemented")
    train = {a.group_id for a in manifest.assignments if a.role == "train"}
    cal = {a.group_id for a in manifest.assignments if a.role == "calibration"}
    train_order = sorted(train, key=lambda g: hash_value([seed, "oof", g]))
    cal_order = sorted(cal, key=lambda g: hash_value([seed, "calibration-check", g]))
    if len(train_order) < 3 or len(cal_order) < 4:
        raise ValueError("insufficient_groups_for_ml_protocol")
    return (
        {g: i % 3 for i, g in enumerate(train_order)},
        tuple(cal_order[::2]),
        tuple(cal_order[1::2]),
    )


def extract_sample(record: Record, plan: MLPlan, *, group_id: str, role: str) -> MLSample:
    if role not in {"train", "tune", "calibration"}:
        raise ValueError("holdout_forbidden")
    if record.dataset_id != plan.dataset_id:
        raise ValueError("dataset_mismatch")
    fields = {f.name: f for f in record.fields}

    def value(binding, numeric):
        field = fields.get(binding.source_field)
        if field is None:
            # Entirely absent target is a missing target, not known truth.
            if isinstance(binding, TargetBinding):
                return None, "missing"
            raise ValueError("required_predictor_column_missing")
        if field.unit != binding.unit:
            raise ValueError("unit_mismatch")
        if field.state != "observed":
            return None, field.state
        v = field.normalized_value
        if numeric and (type(v) not in (int, float) or not math.isfinite(v)):
            return None, "parse_error"
        if not numeric and not isinstance(v, str):
            raise ValueError("category_type_mismatch")
        return v, field.state

    features = {f.name: value(f, f.kind == "numeric")[0] for f in plan.features}
    targets = {t.name: value(t, True) for t in plan.targets}
    return MLSample(
        schema_version="1.0",
        record_id=record.id,
        group_id=group_id,
        role=role,
        features=features,
        targets={k: v[0] for k, v in targets.items()},
        target_states={k: v[1] for k, v in targets.items()},
    )


class LocalData:
    def __init__(self, config_path: Path, *, workspace_root: Path):
        path = Path(config_path)
        if not regular_output(path) or path.stat().st_size > 32 * 1024**2:
            raise ValueError("config_type_or_size")
        config = json.loads(path.read_text("utf-8-sig"))
        self.source_root = Path(config["source_dir"])
        if not self.source_root.is_absolute():
            raise ValueError("absolute_source_required")
        self.private = checked_private(
            Path(config["private_dir"]), self.source_root.resolve() / "boundary.csv", workspace_root
        )
        from forestcheck.evaluation.local import safe_directory

        safe_directory(self.private / "stage5")
        safe_directory(self.private / "stage6")
        receipt = read_private_json(self.private / "stage5/receipt.json")
        splitpath = self.private / "stage5/split-manifest.json"
        if not regular_output(splitpath) or splitpath.stat().st_size > 256 * 1024**2:
            raise ValueError("split_type_or_size")
        if sha256(splitpath) != receipt["split_file_sha256"]:
            raise ValueError("split_integrity")
        self.manifest = SplitManifest.model_validate_json(splitpath.read_bytes())
        if self.manifest.digest != receipt["manifest_digest"]:
            raise ValueError("split_digest")
        self.plan = MLPlan.model_validate(read_private_json(self.private / "stage6/plan.json"))
        plan = self.plan
        if plan.split_sha256 != self.manifest.digest:
            raise ValueError("plan_split_mismatch")
        expected_folds, fit, check = freeze_subsets(self.manifest, seed=plan.seed)
        if (
            plan.train_folds != expected_folds
            or plan.calibration_fit_groups != fit
            or plan.calibration_check_groups != check
        ):
            raise ValueError("subsplit_changed")
        self.folder = self.private / "imports" / str(plan.dataset_id)
        if self.folder.resolve() != self.folder or not self.folder.is_dir():
            raise ValueError("snapshot_directory")
        meta = read_private_json(self.folder / "manifest.json")
        name = meta.get("source_name", "")
        if not name or any(c in name for c in "/\\:") or name in (".", ".."):
            raise ValueError("source_basename")
        self.source = checked_source(self.source_root / name)
        self.mapping = ImportMapping.model_validate(read_private_json(self.folder / "mapping.json"))
        if self.mapping.mapping_version != plan.mapping_version or self.mapping.table != plan.table:
            raise ValueError("mapping_version_or_table")
        columns = {c.field: c for c in self.mapping.columns}
        for b in (*plan.features, *plan.targets):
            if b.source_field not in columns or columns[b.source_field].unit != b.unit:
                raise ValueError("mapping_binding_mismatch")
        self.verify_inputs()

    def verify_inputs(self):
        p = self.plan
        for file, digest, key in (
            (self.source, p.source_sha256, "source"),
            (self.folder / "normalized.parquet", p.snapshot_sha256, "parquet"),
            (self.folder / "mapping.json", p.mapping_sha256, "mapping"),
        ):
            if not regular_output(file) or sha256(file) != digest:
                raise ValueError("input_integrity")
            if self.manifest.inputs[f"{p.table}:{key}"] != digest:
                raise ValueError("split_input_mismatch")

    def load(self, roles: set[str]) -> list[MLSample]:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        if not roles or not roles <= {"train", "tune", "calibration"}:
            raise ValueError("holdout_forbidden")
        assignments = {
            str(a.record_id): a
            for a in self.manifest.assignments
            if a.dataset_id == self.plan.dataset_id and a.role in roles
        }
        samples = []
        seen = set()
        allowed = pa.array(list(assignments), type=pa.string())
        # Only selected UUIDs reach JSON decoding/Record/feature extraction.
        for batch in pq.ParquetFile(self.folder / "normalized.parquet").iter_batches(
            columns=["record_id", "record_json"], batch_size=2048
        ):
            for row in batch.filter(pc.is_in(batch.column(0), value_set=allowed)).to_pylist():
                assignment = assignments.get(row["record_id"])
                if assignment is None:
                    continue
                record = Record.model_validate_json(row["record_json"])
                if record.id != assignment.record_id or record.id in seen:
                    raise ValueError("record_identity_mismatch_or_duplicate")
                samples.append(
                    extract_sample(
                        record, self.plan, group_id=assignment.group_id, role=assignment.role
                    )
                )
                seen.add(record.id)
        if seen != {a.record_id for a in assignments.values()}:
            raise ValueError("selected_records_missing")
        self.verify_inputs()
        return sorted(samples, key=lambda s: str(s.record_id))
