"""Private development-only runner for existing immutable imports."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import AwareDatetime, Field, model_validator

from forestcheck.domain.contracts import Contract, Digest, Record, Text, ValueObject
from forestcheck.growth import GrowthQuery, GrowthTableRegistry
from forestcheck.ingest.csv_import import (
    canonical,
    checked_private,
    checked_source,
    regular_output,
    sha256,
)
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.rules.composition import parse_composition
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.models import EvaluationContext, RuleDefinition

RUNNER_VERSION = "rules-local-1.0.0"

Count = Annotated[int, Field(strict=True, ge=0)]


class SplitInput(ValueObject):
    dataset_id: UUID
    source_sha256: Digest
    mapping_sha256: Digest
    parquet_sha256: Digest


class DevelopmentAssignment(ValueObject):
    record_id: UUID
    group_digest: Digest


class SplitCounts(ValueObject):
    all_records: Count
    development_records: Count
    all_groups: Count
    unassigned_missing_key: Count


class DevelopmentSplit(Contract):
    split_id: Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
    inputs: dict[Text, SplitInput]
    assignments: dict[Text, tuple[DevelopmentAssignment, ...]]
    purpose: Literal["development_only_no_final_test_claim"] = (
        "development_only_no_final_test_claim"
    )
    created_at: AwareDatetime | None = None
    selection: Text | None = None
    grouping_basis: Text | None = None
    seed: Text | None = None
    development_group_digests: tuple[Digest, ...] | None = None
    all_group_count: Count | None = None
    counts: dict[Text, SplitCounts] | None = None
    remaining_policy: Text | None = None
    stage5_handoff: Text | None = None

    @model_validator(mode="after")
    def consistent_assignments(self):
        if not self.inputs or set(self.inputs) != set(self.assignments):
            raise ValueError("split_table_mismatch")
        if self.counts is not None and set(self.counts) != set(self.inputs):
            raise ValueError("split_count_tables_mismatch")
        for alias, assignments in self.assignments.items():
            if len({a.record_id for a in assignments}) != len(assignments):
                raise ValueError("duplicate_split_assignment")
            if self.counts is not None and self.counts[alias].development_records != len(
                assignments
            ):
                raise ValueError("split_assignment_count_mismatch")
            if self.development_group_digests is not None:
                groups = set(self.development_group_digests)
                if any(a.group_digest not in groups for a in assignments):
                    raise ValueError("assignment_outside_development_groups")
        return self


class DevelopmentPlan(Contract):
    version: Text
    split_sha256: Digest
    rules_by_table: dict[Text, tuple[RuleDefinition, ...]]
    composition_fields: dict[Text, Text]
    blocked_requirements: tuple[Text, ...]

    @model_validator(mode="after")
    def declared_tables(self):
        if not self.rules_by_table or not set(self.composition_fields) <= set(self.rules_by_table):
            raise ValueError("invalid_plan_tables")
        return self


def read_private_json(path):
    if not regular_output(path) or path.stat().st_size > 32 * 1024**2:
        raise ValueError("private_file_type_or_size")
    return json.loads(path.read_text("utf-8"))


def run_development(config_path: Path, *, workspace_root: Path) -> Path:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    config = json.loads(Path(config_path).read_text("utf-8-sig"))
    source_root = Path(config["source_dir"])
    if not source_root.is_absolute():
        raise ValueError("absolute_source_required")
    private = checked_private(
        Path(config["private_dir"]), source_root.resolve() / "boundary.csv", workspace_root
    )
    stage = private / "stage4"
    if stage.resolve() != stage or stage.is_symlink():
        raise ValueError("private_stage_link")
    plan_path, split_path = stage / "rule-plan.json", stage / "development-split.json"
    plan = DevelopmentPlan.model_validate(read_private_json(plan_path))
    split_hash = sha256(split_path)
    if split_hash != plan.split_sha256:
        raise ValueError("split_integrity")
    split = DevelopmentSplit.model_validate(read_private_json(split_path)).model_dump(mode="json")
    if (
        split.get("schema_version") != "1.0"
        or not split.get("split_id")
        or set(split["inputs"]) != set(plan.rules_by_table)
        or set(split["assignments"]) != set(plan.rules_by_table)
    ):
        raise ValueError("split_contract")
    sources, snapshots, input_hashes = {}, {}, {}
    for alias, descriptor in split["inputs"].items():
        folder = private / "imports" / str(UUID(descriptor["dataset_id"]))
        if folder.resolve() != folder or folder.is_symlink():
            raise ValueError("snapshot_link")
        manifest = read_private_json(folder / "manifest.json")
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("dataset_id") != descriptor["dataset_id"]
        ):
            raise ValueError("import_manifest_version_or_identity")
        name = manifest["source_name"]
        if not name or any(c in name for c in "/\\:") or name in (".", ".."):
            raise ValueError("source_basename_required")
        source = checked_source(source_root / name)
        if (
            sha256(source) != descriptor["source_sha256"]
            or manifest["source_sha256"] != descriptor["source_sha256"]
        ):
            raise ValueError("source_integrity")
        parquet = folder / "normalized.parquet"
        if not regular_output(parquet) or sha256(parquet) != descriptor["parquet_sha256"]:
            raise ValueError("import_integrity")
        mapping_path = folder / "mapping.json"
        mapping = ImportMapping.model_validate(read_private_json(mapping_path))
        if sha256(mapping_path) != descriptor["mapping_sha256"]:
            raise ValueError("mapping_integrity")
        known = {c.field for c in mapping.columns}
        if any(not set(r.dependencies) <= known for r in plan.rules_by_table[alias]):
            raise ValueError("rule_dependencies_not_in_import_contract")
        if alias in plan.composition_fields and plan.composition_fields[alias] not in known:
            raise ValueError("composition_field_not_in_import_contract")
        snapshots[alias] = (parquet, mapping)
        sources[alias] = source
        input_hashes[alias] = descriptor
    plan_json = canonical(plan.model_dump(mode="json"))
    plan_hash = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()
    run_id = uuid5(NAMESPACE_URL, f"{RUNNER_VERSION}:{plan_hash}:{split_hash}")
    parent = stage / "runs"
    parent.mkdir(exist_ok=True)
    if parent.resolve() != parent or parent.is_symlink():
        raise ValueError("private_runs_link")
    destination = parent / str(run_id)
    files = {
        "outcomes.jsonl",
        "findings.jsonl",
        "composition.jsonl",
        "growth.jsonl",
        "report.json",
        "plan.json",
    }
    if destination.exists():
        if destination.resolve() != destination or destination.is_symlink():
            raise ValueError("private_run_link")
        manifest = read_private_json(destination / "manifest.json")
        expected_metadata = {
            "schema_version": "1.0",
            "runner_version": RUNNER_VERSION,
            "run_id": str(run_id),
            "plan_sha256": plan_hash,
            "split_sha256": split_hash,
            "inputs": input_hashes,
        }
        if set(manifest) != set(expected_metadata) | {"output_hashes"} or any(
            manifest[k] != v for k, v in expected_metadata.items()
        ):
            raise ValueError("run_manifest_integrity")
        if set(manifest["output_hashes"]) != files:
            raise ValueError("run_manifest_integrity")
        for name in files:
            if (
                not regular_output(destination / name)
                or sha256(destination / name) != manifest["output_hashes"][name]
            ):
                raise ValueError("run_output_integrity")
        return destination
    staging = parent / (".incomplete-" + uuid4().hex)
    staging.mkdir(mode=0o700)
    (staging / "plan.json").write_text(plan_json, encoding="utf-8")
    statuses, growth_statuses, composition_statuses, signals = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    tables, total = {}, 0
    with (
        (staging / "outcomes.jsonl").open("w", encoding="utf-8") as outcomes,
        (staging / "findings.jsonl").open("w", encoding="utf-8") as findings,
        (staging / "composition.jsonl").open("w", encoding="utf-8") as compositions,
        (staging / "growth.jsonl").open("w", encoding="utf-8") as growth,
    ):
        registry = GrowthTableRegistry()
        for alias, (parquet, mapping) in snapshots.items():
            selected = [str(UUID(a["record_id"])) for a in split["assignments"][alias]]
            if len(selected) != len(set(selected)):
                raise ValueError("duplicate_split_assignment")
            selected_set, seen = set(selected), set()
            batch_values = pa.array(selected, type=pa.string())
            engine = RuleEngine(list(plan.rules_by_table[alias]))
            context = EvaluationContext(analysis_run_id=run_id, record_level=mapping.record_level)
            table_counts = Counter()
            for batch in pq.ParquetFile(parquet).iter_batches(
                batch_size=512, columns=["record_id", "record_json"]
            ):
                chosen = batch.filter(pc.is_in(batch.column(0), value_set=batch_values))
                for payload in chosen.to_pylist():
                    row = Record.model_validate_json(payload["record_json"])
                    if (
                        str(row.id) != payload["record_id"]
                        or str(row.dataset_id) != split["inputs"][alias]["dataset_id"]
                    ):
                        raise ValueError("record_identity_mismatch")
                    if str(row.id) in seen:
                        raise ValueError("duplicate_technical_record")
                    seen.add(str(row.id))
                    total += 1
                    for outcome in engine.evaluate(row, context):
                        outcomes.write(outcome.model_dump_json() + "\n")
                        statuses[outcome.status] += 1
                        table_counts[outcome.status] += 1
                        if outcome.finding:
                            findings.write(outcome.finding.model_dump_json() + "\n")
                            signals[outcome.signal_class] += 1
                    if alias in plan.composition_fields:
                        f = next(
                            (f for f in row.fields if f.name == plan.composition_fields[alias]),
                            None,
                        )
                        if f is None or f.state in ("missing", "unknown", "parse_error"):
                            result = {
                                "status": "insufficient_data",
                                "reason": "composition_input_unavailable",
                            }
                        elif f.state == "not_applicable":
                            result = {
                                "status": "not_applicable",
                                "reason": "composition_not_applicable",
                            }
                        else:
                            result = parse_composition(str(f.original_value)).model_dump(
                                mode="json"
                            )
                        composition_statuses[result["status"]] += 1
                        compositions.write(
                            canonical(dict(record_id=str(row.id), result=result)) + "\n"
                        )
                    # No approved metadata/reference is invented to obtain a nominal comparison.
                    lookup = registry.lookup(GrowthQuery(schema_version="1.0"))
                    growth_statuses[lookup.status] += 1
                    growth.write(
                        canonical(
                            dict(record_id=str(row.id), result=lookup.model_dump(mode="json"))
                        )
                        + "\n"
                    )
            if seen != selected_set:
                raise ValueError("selected_records_missing")
            tables[alias] = {"records_evaluated": len(seen), "rule_statuses": dict(table_counts)}
    for alias, source in sources.items():
        if sha256(checked_source(source)) != input_hashes[alias]["source_sha256"]:
            raise ValueError("source_changed")
        if sha256(snapshots[alias][0]) != input_hashes[alias]["parquet_sha256"]:
            raise ValueError("import_changed")
    if (
        sha256(split_path) != split_hash
        or canonical(
            DevelopmentPlan.model_validate(read_private_json(plan_path)).model_dump(mode="json")
        )
        != plan_json
    ):
        raise ValueError("run_configuration_changed")
    report = {
        "schema_version": "1.0",
        "runner_version": RUNNER_VERSION,
        "records_evaluated": total,
        "rule_statuses": dict(statuses),
        "signal_classes": dict(signals),
        "tables": tables,
        "growth_statuses": dict(growth_statuses),
        "composition_statuses": dict(composition_statuses),
        "coverage_blockers": list(plan.blocked_requirements),
        "accuracy_evaluated": False,
        "development_only": True,
        "source_unchanged": True,
        "import_unchanged": True,
    }
    (staging / "report.json").write_text(canonical(report), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "runner_version": RUNNER_VERSION,
        "run_id": str(run_id),
        "plan_sha256": plan_hash,
        "split_sha256": split_hash,
        "inputs": input_hashes,
        "output_hashes": {name: sha256(staging / name) for name in files},
    }
    (staging / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
    staging.rename(destination)
    return destination
