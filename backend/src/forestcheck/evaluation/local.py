"""Offline split freeze from verified stage-3 identities and stage-4 exposure.

Only record identities are used before freeze. After freeze, bounded train/tune
records are validated structurally, without running rules or computing metrics.
"""

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from forestcheck.domain.contracts import Record
from forestcheck.evaluation.split import SplitIdentity, canonical, freeze_split, hash_value
from forestcheck.ingest.csv_import import checked_private, checked_source, regular_output, sha256
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.rules.local import DevelopmentPlan, DevelopmentSplit, read_private_json

SEED = "forestcheck-stage5-quarter-v1"


def safe_directory(path: Path):
    if path.resolve() != path or path.is_symlink() or not path.is_dir():
        raise ValueError("private_directory_link_or_type")


def prepare_evaluation(config_path: Path, *, workspace_root: Path) -> Path:
    import pyarrow.parquet as pq

    started = time.perf_counter()
    config = json.loads(Path(config_path).read_text("utf-8-sig"))
    source_root = Path(config["source_dir"])
    if not source_root.is_absolute():
        raise ValueError("absolute_source_required")
    private = checked_private(Path(config["private_dir"]), source_root.resolve() / "boundary.csv", workspace_root)
    prior_dir = private / "stage4"
    safe_directory(prior_dir)
    prior_path = prior_dir / "development-split.json"
    prior = DevelopmentSplit.model_validate(read_private_json(prior_path))
    prior_hash = sha256(prior_path)
    plan = DevelopmentPlan.model_validate(read_private_json(prior_dir / "rule-plan.json"))
    if plan.split_sha256 != prior_hash:
        raise ValueError("prior_split_integrity")
    prior_ids = {a.record_id for assignments in prior.assignments.values() for a in assignments}
    inputs, snapshots, sources, mappings = {"stage4_split": prior_hash}, {}, {}, {}
    # Check every source/snapshot hash BEFORE trusting object identities.
    for alias, descriptor in prior.inputs.items():
        folder = private / "imports" / str(descriptor.dataset_id)
        safe_directory(folder)
        meta = read_private_json(folder / "manifest.json")
        name = meta.get("source_name", "")
        if not name or any(c in name for c in "/\\:") or name in (".", ".."):
            raise ValueError("source_basename_required")
        source = checked_source(source_root / name)
        if sha256(source) != descriptor.source_sha256 or meta.get("source_sha256") != descriptor.source_sha256:
            raise ValueError("source_integrity")
        if meta.get("schema_version") != "1.0" or meta.get("dataset_id") != str(descriptor.dataset_id):
            raise ValueError("import_manifest_version_or_identity")
        parquet, mapping_path = folder / "normalized.parquet", folder / "mapping.json"
        if not regular_output(parquet) or sha256(parquet) != descriptor.parquet_sha256:
            raise ValueError("parquet_integrity")
        if not regular_output(mapping_path) or sha256(mapping_path) != descriptor.mapping_sha256:
            raise ValueError("mapping_integrity")
        mappings[alias] = ImportMapping.model_validate(read_private_json(mapping_path))
        snapshots[alias], sources[alias] = parquet, source
        inputs.update({f"{alias}:source": descriptor.source_sha256,
                       f"{alias}:parquet": descriptor.parquet_sha256,
                       f"{alias}:mapping": descriptor.mapping_sha256})
    identities, exposed, seen_prior, alias_by_dataset = [], set(), set(), {}
    for alias, parquet in snapshots.items():
        descriptor = prior.inputs[alias]
        alias_by_dataset[descriptor.dataset_id] = alias
        # record_json is the current import's only business-key representation.
        # Decoding is mechanical; fields are not extracted or profiled here.
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["record_id", "record_json"], batch_size=2048):
            for item in batch.to_pylist():
                row = json.loads(item["record_json"])
                rid = UUID(item["record_id"])
                if row.get("schema_version") != "1.0" or row.get("id") != str(rid) or row.get("dataset_id") != str(descriptor.dataset_id):
                    raise ValueError("identity_version_or_dataset_mismatch")
                key = row.get("business_key") or {}
                parent = [key.get(k) for k in ("territory", "compartment", "stand")]
                if any(not isinstance(k, str) or not k for k in parent):
                    raise ValueError("missing_group_key")
                group = hash_value(["territory_compartment", *parent[:2]])
                identities.append(SplitIdentity(
                    schema_version="1.0", record_id=rid, dataset_id=descriptor.dataset_id,
                    parent_id=hash_value(["territory_compartment_stand", *parent]), group_id=group,
                ))
                if rid in prior_ids:
                    exposed.add(group)
                    seen_prior.add(rid)
    if seen_prior != prior_ids:
        raise ValueError("prior_records_missing")
    manifest = freeze_split(
        identities, seed=SEED, grouping_basis="Whole territory/compartment from imported BusinessKey; "
        "all stands, layers, elements and regeneration together; repeated keys stay together.",
        inputs=inputs, prior_exposed_groups=tuple(exposed), limitations=(
            "Unofficial internal split; no organizer test or label contract has been provided.",
            "Stage 3 technically profiled all imports. Holdout is reserved, not pristine external test.",
            "Quarter grouping is conservative; adjacency, temporal identifiers and aliases are unconfirmed.",
            "No coordinates or invented geography; hidden linkage across different codes remains possible.",
        ),
    )
    destination = private / "stage5"
    destination.mkdir(exist_ok=True)
    safe_directory(destination)
    manifest_path = destination / "split-manifest.json"
    receipt_path = destination / "receipt.json"
    if receipt_path.exists():
        receipt = read_private_json(receipt_path)
        if receipt.get("manifest_digest") != manifest.digest or receipt.get("inputs") != inputs:
            raise ValueError("existing_split_configuration_mismatch")
        for name, expected in receipt.get("output_hashes", {}).items():
            if name not in {"split-manifest.json", "structure-report.json"}:
                raise ValueError("unexpected_artifact")
            path = destination / name
            if not regular_output(path) or sha256(path) != expected:
                raise ValueError("existing_artifact_integrity")
        if set(receipt.get("output_hashes", {})) != {"split-manifest.json", "structure-report.json"}:
            raise ValueError("incomplete_receipt")
        return destination
    if manifest_path.exists() or (destination / "structure-report.json").exists():
        raise ValueError("incomplete_run_requires_local_review")
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(manifest.model_dump(mode="json")) + "\n")
    # Freeze has now reached durable disk, before any validation of feature objects.
    assignments = {a.record_id: a for a in manifest.assignments}
    selected, sample_counts = set(), Counter()
    sample_roles = {"development_cv"} if manifest.strategy == "group_cv" else {"train", "tune"}
    for a in manifest.assignments:
        alias = alias_by_dataset[a.dataset_id]
        if a.role in sample_roles and sample_counts[(alias, a.role)] < 100:
            selected.add(a.record_id)
            sample_counts[(alias, a.role)] += 1
    checked = set()
    for alias, parquet in snapshots.items():
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["record_id", "record_json"], batch_size=2048):
            for item in batch.to_pylist():
                rid = UUID(item["record_id"])
                if rid not in selected:
                    continue
                record = Record.model_validate_json(item["record_json"])
                if record.id != rid or record.dataset_id != prior.inputs[alias].dataset_id:
                    raise ValueError("sample_identity_mismatch")
                if {f.name for f in record.fields} != {c.field for c in mappings[alias].columns}:
                    raise ValueError("sample_mapping_field_mismatch")
                checked.add(rid)
    if checked != selected:
        raise ValueError("structure_sample_missing")
    if sha256(prior_path) != prior_hash or any(sha256(sources[a]) != prior.inputs[a].source_sha256
                                              or sha256(snapshots[a]) != prior.inputs[a].parquet_sha256
                                              or sha256(snapshots[a].parent / "mapping.json") != prior.inputs[a].mapping_sha256
                                              for a in snapshots):
        raise ValueError("inputs_changed_during_run")
    by_role, by_table, groups_by_role = Counter(), defaultdict(Counter), defaultdict(set)
    for a in assignments.values():
        by_role[a.role] += 1
        by_table[alias_by_dataset[a.dataset_id]][a.role] += 1
        groups_by_role[a.role].add(a.group_id)
    report = {
        "schema_version": "1.0", "rows": len(assignments),
        "groups": len({a.group_id for a in assignments.values()}),
        "parents": len({a.parent_id for a in assignments.values()}),
        "prior_exposed_groups": len(exposed), "strategy": manifest.strategy,
        "record_counts_by_role": dict(by_role), "record_counts_by_table": dict(by_table),
        "group_counts_by_role": {k: len(v) for k, v in groups_by_role.items()},
        "structure_validated_records": len(checked),
        "structure_samples": {f"{a}:{r}": count for (a, r), count in sample_counts.items()},
        "feature_validation_roles": sorted(sample_roles),
        "final_holdout_evaluated": False, "metrics_status": "NOT_RUN",
        "source_and_import_hashes_unchanged": True,
        "limitations": list(manifest.limitations),
    }
    report_path = destination / "structure-report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(report) + "\n")
    receipt = {
        "schema_version": "1.0", "runner_version": "evaluation-local-1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "manifest_digest": manifest.digest, "split_file_sha256": sha256(manifest_path),
        "inputs": inputs, "output_hashes": {p.name: sha256(p) for p in (manifest_path, report_path)},
        "measured_wall_seconds": time.perf_counter() - started,
        "peak_memory_status": "NOT_RUN",
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(receipt) + "\n")
    return destination
