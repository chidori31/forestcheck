"""Adapter around shipped engines. No new biological rules or trained estimators."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid5

from forestcheck.api.settings import ordinary_path
from forestcheck.domain.contracts import Evidence, Finding
from forestcheck.evaluation.split import SplitManifest
from forestcheck.growth import GrowthQuery, GrowthTableRegistry
from forestcheck.ingest.csv_import import sha256
from forestcheck.ml.data import extract_sample
from forestcheck.ml.local import load_model_for_plan
from forestcheck.ml.pipeline import ConditionalPipeline
from forestcheck.repair.engine import RecommendationEngine
from forestcheck.repair.local import _comparable, _query, affected_fields
from forestcheck.repair.providers import growth_source, model_source, stock_sources
from forestcheck.repair.references import ReferenceObservation, ReferencePool
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.models import EvaluationContext, UniqueKeyCheck
from forestcheck.storage.application import WorkflowError, digest


def checked_models(profile):
    if not profile.models:
        return []
    plan = profile.ml_plan
    path = ordinary_path(Path(profile.split_manifest_path))
    manifest = SplitManifest.model_validate_json(path.read_text("utf-8"))
    if manifest.digest != plan.split_sha256:
        raise WorkflowError("model_split_incompatible")
    for record_id in profile.allowed_record_ids:
        assignment = manifest.assert_allowed(record_id, roles={"tune"})
        if assignment.dataset_id != plan.dataset_id:
            raise WorkflowError("model_cohort_incompatible")
    result = []
    for binding in profile.models:
        target = next((t for t in plan.targets if t.name == binding.target), None)
        if target is None:
            raise WorkflowError("model_target_incompatible")
        artifact = load_model_for_plan(
            Path(binding.registry),
            binding.model_id,
            plan,
            target,
            run_id=binding.run_id,
            train_hash=binding.train_sha256,
        )
        if digest(artifact.metadata.model_dump(mode="json")) != binding.metadata_sha256:
            raise WorkflowError("model_metadata_changed")
        model = ConditionalPipeline.from_state(
            json.loads(artifact.model_files["state.json"]), artifact.path("model.cbm")
        )
        result.append((binding, target, model))
    return result


def context_for(profile, run_id):
    return EvaluationContext(
        analysis_run_id=UUID(str(run_id)),
        record_level=profile.record_level,
        category=profile.category,
        allow_synthetic=profile.source_kind == "synthetic",
    )


def reference_pools(profile, checkpoint):
    from forestcheck.api.execution import read_records

    if not profile.reference_parquet_path or not profile.ml_plan:
        return {}, {}, None
    plan = profile.ml_plan
    path = ordinary_path(Path(profile.reference_parquet_path))
    if sha256(path) != plan.snapshot_sha256:
        raise WorkflowError("reference_snapshot_changed")
    manifest = SplitManifest.model_validate_json(
        ordinary_path(Path(profile.split_manifest_path)).read_text("utf-8")
    )
    if manifest.digest != plan.split_sha256:
        raise WorkflowError("reference_split_changed")
    assignments = {a.record_id: a for a in manifest.assignments if a.dataset_id == plan.dataset_id}
    data = SimpleNamespace(plan=plan, mapping=SimpleNamespace(record_level=profile.record_level))
    observations = {t.name: [] for t in plan.targets}
    train_ids = {i for i, a in assignments.items() if a.role == "train"}
    for index, record in enumerate(read_records(path, train_ids)):
        a = assignments[record.id]
        sample = extract_sample(record, plan, group_id=a.group_id, role="train")
        for target in plan.targets:
            query = _query(record, sample, a, target, data)
            observations[target.name].append(
                ReferenceObservation(
                    **query.model_dump(),
                    target_value=sample.targets[target.name],
                    target_state=sample.target_states[target.name],
                    source_snapshot_sha256=plan.snapshot_sha256,
                )
            )
        if index % 512 == 0:
            checkpoint()
    pools = {}
    for target in plan.targets:
        pools[target.name] = ReferencePool(
            observations.pop(target.name),
            manifest=manifest,
            dataset_id=plan.dataset_id,
            target_field=target.source_field,
            target_unit=target.unit,
            age_unit=next(f.unit for f in plan.features if f.name == "age"),
            snapshot_input_key=f"{plan.table}:parquet",
            snapshot_sha256=plan.snapshot_sha256,
            mapping_input_key=f"{plan.table}:mapping",
            mapping_version=plan.mapping_version,
            mapping_sha256=plan.mapping_sha256,
            minimum_rows=20,
            minimum_groups=5,
        )
    if sha256(path) != plan.snapshot_sha256:
        raise WorkflowError("reference_snapshot_changed")
    return pools, assignments, data


def analyse(records, profile, binding, run_id, *, use_models, checkpoint):
    rules = RuleEngine(list(profile.rules), codebooks=list(profile.codebooks))
    context = context_for(profile, run_id)
    outcomes = rules.evaluate_many(records, context)
    by_record = {}
    findings = {}
    for outcome in outcomes:
        by_record.setdefault(outcome.record_id, []).append(outcome)
        if outcome.finding:
            findings[outcome.finding.id] = outcome.finding
    checkpoint()
    model_providers = {}
    model_count = 0
    pools, assignments, reference_data = (
        reference_pools(profile, checkpoint) if use_models else ({}, {}, None)
    )
    reference_counts = Counter()
    if use_models and profile.models and not assignments:
        manifest = SplitManifest.model_validate_json(
            ordinary_path(Path(profile.split_manifest_path)).read_text("utf-8")
        )
        assignments = {
            a.record_id: a
            for a in manifest.assignments
            if a.dataset_id == profile.ml_plan.dataset_id
        }
    for model_binding, target, model in checked_models(profile) if use_models else ():
        features = [
            extract_sample(
                r, profile.ml_plan, group_id=assignments[r.id].group_id, role="tune"
            ).features
            for r in records
        ]
        predictions, baseline = model.predict(features), model.predict(features, method="baseline")
        for record, predicted, base in zip(records, predictions, baseline, strict=True):
            providers = [
                model_source(
                    predicted,
                    model_id=model_binding.model_id,
                    plan=profile.ml_plan,
                    target=target,
                    method="conditional_model",
                ),
                model_source(
                    base,
                    model_id=model_binding.model_id,
                    plan=profile.ml_plan,
                    target=target,
                    method="baseline",
                ),
            ]
            if target.name in pools:
                assignment = assignments[record.id]
                sample = extract_sample(
                    record, profile.ml_plan, group_id=assignment.group_id, role="tune"
                )
                reference = pools[target.name].query(
                    _query(record, sample, assignment, target, reference_data)
                )
                reference_counts[reference.status] += 1
                providers.append(_comparable(reference, reference_data))
            model_providers[record.id, target.source_field] = providers
            observed = next(f for f in record.fields if f.name == target.source_field)
            missing = observed.state == "missing"
            unusual = (
                observed.state == "observed"
                and type(observed.normalized_value) in (int, float)
                and predicted.status == "ok"
                and predicted.lower is not None
                and not predicted.lower <= observed.normalized_value <= predicted.upper
            )
            if missing or unusual:
                finding = Finding(
                    schema_version="1.0",
                    id=uuid5(UUID(str(run_id)), f"model:{record.id}:{target.name}"),
                    record_id=record.id,
                    record_revision=record.revision,
                    analysis_run_id=UUID(str(run_id)),
                    field_name=target.source_field,
                    signal_kind="missing" if missing else "anomaly",
                    severity="warning",
                    priority=1,
                    review_status="unreviewed",
                    evidence=(
                        Evidence(
                            code="missing_observation"
                            if missing
                            else "conditional_interval_suspicion",
                            message="Наблюдение отсутствует; требуется проверка применимости поля."
                            if missing
                            else "Наблюдение вне условного диапазона модели; это подозрение, не подтверждённая ошибка.",
                            reference_id=str(model_binding.model_id),
                        ),
                    ),
                )
                findings[finding.id] = finding
        model_count += 1
        checkpoint()
    engine = RecommendationEngine(profile.numeric_fields)
    precision = {p.field_name: p for p in profile.precision}
    records_by_id = {r.id: r for r in records}
    recommendations = []
    for finding in findings.values():
        record = records_by_id[finding.record_id]
        fields = (finding.field_name,) if finding.field_name else tuple(profile.numeric_fields)
        for field_name in fields:
            if field_name not in profile.numeric_fields:
                continue
            sources, candidates = stock_sources(
                record,
                field_name,
                rules,
                context,
                data_root="observations:" + binding.snapshot_sha256,
            )
            sources, candidates = list(sources), list(candidates)
            # Available arithmetic sources are already carried by their Candidate.
            sources = [s for s in sources if s.status != "available"]
            for source, candidate in model_providers.get((record.id, field_name), ()):
                (candidates if candidate else sources).append(candidate if candidate else source)
            source, candidate = growth_source(
                GrowthTableRegistry(),
                GrowthQuery(
                    schema_version="1.0",
                    indicator=field_name,
                    unit=profile.numeric_fields[field_name],
                    age_unit="year",
                ),
            )
            sources.append(source)
            relevant_relations = tuple(
                a
                for a in profile.aggregates
                if record.id in a.component_record_ids or record.id == a.total_record_id
            )
            related_ids = {
                i for a in relevant_relations for i in (*a.component_record_ids, a.total_record_id)
            }
            if any(
                isinstance(r.check, UniqueKeyCheck) and field_name in r.dependencies
                for r in rules.rules
            ):
                related_ids.update(records_by_id)
            related = tuple(r for r in records if r.id in related_ids and r.id != record.id)
            result = engine.recommend(
                record,
                finding,
                field_name,
                candidates=tuple(candidates),
                sources=tuple(sources),
                precision=precision.get(field_name),
                binding=binding,
                rule_engine=rules,
                context=context,
                suspect_fields=affected_fields(by_record.get(record.id, ()), rules),
                aggregate_relations=relevant_relations,
                related_records=related,
                related_contexts={r.id: context for r in related},
            )
            recommendations.append(result)
        checkpoint()
    report = {
        "schema_version": "1.0",
        "analysed_records": len(records),
        "findings": len(findings),
        "suggestions": len(recommendations),
        "numeric_proposals": sum(r.patch is not None for r in recommendations),
        "rule_outcomes": dict(Counter(o.status for o in outcomes)),
        "model_count": model_count,
        "growth_reference": "unavailable_reference",
        "comparable_reference": "train_only_registered_snapshot"
        if pools
        else "NOT_RUN_backend_profile_no_reference_pool",
        "reference_queries": dict(reference_counts),
        "aggregate_relations": len(profile.aggregates),
        "expert_accuracy": "NOT_RUN",
        "patches_applied": 0,
        "limitations": list(profile.limitations)
        + ["unverified_observations", "unknown_reference_coverage"],
    }
    return list(findings.values()), recommendations, report
