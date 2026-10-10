"""Explicit local registration of verified stage 4–7 artifacts; never an HTTP input."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from forestcheck.api.contracts import AnalysisProfile, TrustedModel
from forestcheck.api.service import Service
from forestcheck.ingest.csv_import import sha256
from forestcheck.ml.data import LocalData
from forestcheck.ml.local import load_model_for_plan
from forestcheck.repair.local import RecommendationPlan, _load_rules, _model_report
from forestcheck.rules.local import read_private_json
from forestcheck.storage.application import WorkflowError, digest


def register_forestcheck(store, config_path: Path, *, workspace_root: Path):
    data = LocalData(config_path, workspace_root=workspace_root)
    path = data.private / "stage7/plan.json"
    plan = RecommendationPlan.model_validate(read_private_json(path))
    receipt = read_private_json(path.parent / "plan-receipt.json")
    rules, precision, rule_hash = _load_rules(data)
    report = _model_report(data)
    if (
        sha256(path) != receipt["sha256"]
        or digest(plan.model_dump(mode="json")) != receipt["canonical_sha256"]
        or plan.ml_plan_sha256 != data.plan.digest
        or plan.split_sha256 != data.manifest.digest
        or plan.rule_plan_sha256 != rule_hash
        or plan.model_run_id != report.run_id
        or plan.model_ids != report.model_ids
    ):
        raise WorkflowError("registered_development_plan_incompatible")
    models = []
    for target in data.plan.targets:
        registry = data.private / "stage6/models" / target.name
        train_hash = report.targets[target.name]["train_sha256"]
        artifact = load_model_for_plan(
            registry,
            plan.model_ids[target.name],
            data.plan,
            target,
            run_id=report.run_id,
            train_hash=train_hash,
        )
        models.append(
            TrustedModel(
                schema_version="1.0",
                model_id=plan.model_ids[target.name],
                target=target.name,
                registry=str(registry),
                run_id=report.run_id,
                train_sha256=train_hash,
                metadata_sha256=digest(artifact.metadata.model_dump(mode="json")),
            )
        )
    values = dict(
        schema_version="1.0",
        name="ForestCheck development",
        version="backend-8.1",
        mapping_sha256=data.plan.mapping_sha256,
        record_level=data.mapping.record_level,
        source_kind=data.plan.source_kind,
        rules=tuple(rules.rules),
        precision=tuple(precision.values()),
        numeric_fields={t.source_field: t.unit for t in data.plan.targets},
        models=tuple(models),
        ml_plan=data.plan,
        split_manifest_path=str(data.private / "stage5/split-manifest.json"),
        reference_parquet_path=str(data.folder / "normalized.parquet"),
        allowed_record_ids=plan.record_ids,
        limitations=(
            "previously_used_tune_cohort",
            "unverified_observations",
            "growth_tables_and_codebooks_unavailable",
            "related_aggregate_contract_unavailable",
            "unknown_semantic_duplicates",
        ),
    )
    prototype = AnalysisProfile(id=uuid5(NAMESPACE_URL, "forestcheck:profile"), **values)
    profile = prototype.model_copy(
        update={
            "id": uuid5(
                NAMESPACE_URL, "forestcheck:profile:" + digest(prototype.model_dump(mode="json"))
            )
        }
    )
    data.verify_inputs()
    return Service(store).register_profile(profile)
