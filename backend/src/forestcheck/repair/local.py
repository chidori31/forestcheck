"""Offline stage-7 dev experiment. Sources, labels, cases and proposed patches stay separate."""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4, uuid5

from forestcheck.domain.contracts import Contract, Digest, Evidence, Finding, Record
from forestcheck.evaluation.scoring import NumericCase, ScoringProtocol, score_frozen_cohort
from forestcheck.evaluation.split import canonical, hash_value
from forestcheck.growth import GrowthQuery, GrowthTableRegistry
from forestcheck.ingest.csv_import import regular_output, sha256
from forestcheck.ml.data import LocalData, extract_sample
from forestcheck.ml.local import RunReport, directory, load_model_for_plan, write_json
from forestcheck.ml.pipeline import ConditionalPipeline
from forestcheck.repair.engine import RecommendationEngine, verify_patch
from forestcheck.repair.models import Candidate, DatasetBinding, PrecisionPolicy, SourceSupport
from forestcheck.repair.providers import growth_source, model_source, stock_sources
from forestcheck.repair.references import ReferenceObservation, ReferencePool, ReferenceQuery
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.local import DevelopmentPlan, DevelopmentSplit, read_private_json
from forestcheck.rules.models import EvaluationContext, RoundingCheck


class RecommendationPlan(Contract):
    plan_version: Literal["recommendations-dev-1.0.0"] = "recommendations-dev-1.0.0"
    seed: Literal["recommendations-dev-v1"] = "recommendations-dev-v1"
    ml_plan_sha256: Digest
    split_sha256: Digest
    rule_plan_sha256: Digest
    model_run_id: UUID
    model_ids: dict[str, UUID]
    group_ids: tuple[Digest, ...]
    record_ids: tuple[UUID, ...]
    minimum_reference_rows: Literal[20] = 20
    minimum_reference_groups: Literal[5] = 5
    max_dev_groups: Literal[128] = 128
    role: Literal["tune"] = "tune"
    mode: Literal["masked_observation"] = "masked_observation"
    holdout: Literal["NOT_RUN"] = "NOT_RUN"


def affected_fields(outcomes, rules):
    dependencies = {rule.id: rule.dependencies for rule in rules.rules}
    return tuple(
        sorted(
            {
                field
                for outcome in outcomes
                if outcome.finding
                for field in dependencies[outcome.rule_id]
            }
        )
    )


def field_findings(outcomes, rules, field_name):
    dependencies = {rule.id: rule.dependencies for rule in rules.rules}
    return [o.finding for o in outcomes if o.finding and field_name in dependencies[o.rule_id]]


def _load_rules(data):
    stage = data.private / "stage4"
    if stage.resolve() != stage or not stage.is_dir():
        raise ValueError("rule_directory")
    path = stage / "rule-plan.json"
    plan = DevelopmentPlan.model_validate(read_private_json(path))
    splitpath = stage / "development-split.json"
    if not regular_output(splitpath) or sha256(splitpath) != plan.split_sha256:
        raise ValueError("rule_split_hash")
    split = DevelopmentSplit.model_validate(read_private_json(splitpath))
    p = data.plan
    descriptor = split.inputs[p.table]
    if (
        descriptor.dataset_id != p.dataset_id
        or descriptor.source_sha256 != p.source_sha256
        or descriptor.mapping_sha256 != p.mapping_sha256
        or descriptor.parquet_sha256 != p.snapshot_sha256
    ):
        raise ValueError("rule_mapping_snapshot_mismatch")
    rules = plan.rules_by_table[p.table]
    known = {c.field for c in data.mapping.columns}
    if any(not set(r.dependencies) <= known for r in rules):
        raise ValueError("rule_dependencies_not_mapped")
    precision = {}
    for t in p.targets:
        applicable = [
            r
            for r in rules
            if isinstance(r.check, RoundingCheck)
            and r.check.field == t.source_field
            and r.check.unit == t.unit
            and data.mapping.record_level in r.record_levels
            and not r.categories
        ]
        if len(applicable) != 1:
            raise ValueError("unambiguous_target_precision_required")
        r = applicable[0]
        precision[t.name] = PrecisionPolicy(
            field_name=t.source_field, unit=t.unit, step=r.check.step, source=r.source
        )
    return RuleEngine(list(rules)), precision, sha256(path)


def _model_report(data):
    receipt = read_private_json(data.private / "stage6/verification.json")
    run_id = UUID(receipt["run_id"])
    folder = data.private / "stage6/runs" / str(run_id)
    if folder.resolve() != folder or not folder.is_dir():
        raise ValueError("model_run_directory")
    report = RunReport.model_validate(read_private_json(folder / "report.json"))
    if report.run_id != run_id or report.plan_sha256 != data.plan.digest:
        raise ValueError("model_run_plan")
    for target in data.plan.targets:
        load_model_for_plan(
            data.private / "stage6/models" / target.name,
            report.model_ids[target.name],
            data.plan,
            target,
            run_id=run_id,
            train_hash=report.targets[target.name]["train_sha256"],
        )
    return report


def prepare(config_path: Path, *, workspace_root: Path) -> Path:
    data = LocalData(config_path, workspace_root=workspace_root)
    _, _, rule_hash = _load_rules(data)
    report = _model_report(data)
    assignments = [
        a
        for a in data.manifest.assignments
        if a.dataset_id == data.plan.dataset_id and a.role == "tune"
    ]
    groups = tuple(
        sorted(
            {a.group_id for a in assignments},
            key=lambda g: hash_value(["recommendations-dev-v1", g]),
        )[:128]
    )
    if not groups:
        raise ValueError("no_tune_groups")
    selected = set(groups)
    plan = RecommendationPlan(
        schema_version="1.0",
        ml_plan_sha256=data.plan.digest,
        split_sha256=data.manifest.digest,
        rule_plan_sha256=rule_hash,
        model_run_id=report.run_id,
        model_ids=report.model_ids,
        group_ids=groups,
        record_ids=tuple(
            sorted((a.record_id for a in assignments if a.group_id in selected), key=str)
        ),
    )
    stage = directory(data.private / "stage7")
    path = stage / "plan.json"
    write_json(path, plan.model_dump(mode="json"))
    write_json(
        stage / "plan-receipt.json",
        {
            "schema_version": "1.0",
            "sha256": sha256(path),
            "canonical_sha256": hash_value(plan.model_dump(mode="json")),
        },
    )
    return path


def _records(data, ids):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    allowed = pa.array([str(i) for i in ids], type=pa.string())
    seen = set()
    for batch in pq.ParquetFile(data.folder / "normalized.parquet").iter_batches(
        columns=["record_id", "record_json"], batch_size=2048
    ):
        for row in batch.filter(pc.is_in(batch.column(0), value_set=allowed)).to_pylist():
            record = Record.model_validate_json(row["record_json"])
            if (
                record.id not in ids
                or record.id in seen
                or record.dataset_id != data.plan.dataset_id
            ):
                raise ValueError("record_cohort_binding")
            seen.add(record.id)
            yield record
    if seen != set(ids):
        raise ValueError("incomplete_record_cohort")


def _fingerprint(record):
    # Opaque exclusion-only lineage; never a predictor or numeric search condition.
    return hash_value(
        {
            "business_key": record.business_key.model_dump() if record.business_key else None,
            "raw": [(f.name, f.original_value, f.unit) for f in record.fields],
        }
    )


def _query(record, sample, assignment, target, data):
    return ReferenceQuery(
        schema_version="1.0",
        record_id=record.id,
        dataset_id=record.dataset_id,
        group_id=assignment.group_id,
        parent_id=assignment.parent_id,
        duplicate_id=assignment.duplicate_id,
        role=assignment.role,
        record_level=data.mapping.record_level,
        region=sample.features["region"],
        species=sample.features["species"],
        age=sample.features["age"],
        age_unit=next(f.unit for f in data.plan.features if f.name == "age"),
        target_field=target.source_field,
        target_unit=target.unit,
        raw_record_sha256=_fingerprint(record),
    )


def _comparable(result, data):
    source = SourceSupport(
        source_id="comparable:" + result.target_field,
        version=result.procedure_version,
        method="comparable",
        status=result.status,
        basis=f"{result.reason}; {result.selection_basis}; rows={result.n_rows}; groups={result.n_groups}; "
        f"reference_ids_sha256={result.reference_ids_sha256}",
        dependency_roots=(f"observations:{result.source_snapshot_sha256}",),
        input_fields=tuple(f.source_field for f in data.plan.features),
        limitations=result.limitations,
    )
    candidate = (
        None
        if result.status != "available"
        else Candidate(
            support=source,
            value=result.point,
            lower=result.lower,
            upper=result.upper,
            unit=result.unit,
            uncertainty_basis="Descriptive interval of unverified group medians, not individual prediction coverage",
        )
    )
    return source, candidate


def _masked(record, target):
    fields = [f.model_dump() for f in record.fields]
    for f in fields:
        if f["name"] == target.source_field:
            f.update(original_value=None, normalized_value=None, state="missing")
    return Record.model_validate(
        {
            **record.model_dump(),
            "fields": fields,
            "revision": record.revision + 1,
            "dataset_revision": record.dataset_revision + 1,
        }
    )


def _finding(record, target, run_id, *, masked=False):
    return Finding(
        schema_version="1.0",
        id=uuid5(run_id, f"{record.id}:{target.name}:{masked}"),
        record_id=record.id,
        record_revision=record.revision,
        analysis_run_id=run_id,
        field_name=target.source_field,
        signal_kind="missing" if masked else "anomaly",
        severity="info" if masked else "warning",
        priority=1,
        review_status="unreviewed",
        evidence=(
            Evidence(
                code="evaluation_mask" if masked else "conditional_interval_suspicion",
                message="Наблюдение скрыто для оценки восстановления, это не обнаруженная ошибка."
                if masked
                else "Наблюдение вне условного диапазона или нарушает доступное правило; требуется проверка специалиста.",
                reference_id="recommendations-dev-1.0.0",
            ),
        ),
    )


def run(config_path: Path, *, workspace_root: Path, run_id: UUID | None = None) -> Path:
    started = time.perf_counter()
    plan_path = prepare(
        config_path, workspace_root=workspace_root
    )  # Freeze before reading feature values.
    data = LocalData(config_path, workspace_root=workspace_root)
    plan = RecommendationPlan.model_validate(read_private_json(plan_path))
    receipt = read_private_json(plan_path.parent / "plan-receipt.json")
    if sha256(plan_path) != receipt["sha256"]:
        raise ValueError("recommendation_plan_integrity")
    rules, precision, _ = _load_rules(data)
    report = _model_report(data)
    stage = directory(data.private / "stage7")
    runs = directory(stage / "runs")
    run_id = run_id or uuid4()
    destination = runs / str(run_id)
    destination.mkdir()
    directory(destination)
    write_json(destination / "plan.json", plan.model_dump(mode="json"))
    assignments = {
        a.record_id: a for a in data.manifest.assignments if a.dataset_id == data.plan.dataset_id
    }
    train_ids = {i for i, a in assignments.items() if a.role == "train"}
    observations = {t.name: [] for t in data.plan.targets}
    for record in _records(data, train_ids):
        a = assignments[record.id]
        sample = extract_sample(record, data.plan, group_id=a.group_id, role="train")
        for target in data.plan.targets:
            query = _query(record, sample, a, target, data)
            observations[target.name].append(
                ReferenceObservation(
                    **query.model_dump(),
                    target_value=sample.targets[target.name],
                    target_state=sample.target_states[target.name],
                    source_snapshot_sha256=data.plan.snapshot_sha256,
                )
            )
    pools = {}
    for target in data.plan.targets:
        pools[target.name] = ReferencePool(
            observations.pop(target.name),
            manifest=data.manifest,
            dataset_id=data.plan.dataset_id,
            target_field=target.source_field,
            target_unit=target.unit,
            age_unit=next(f.unit for f in data.plan.features if f.name == "age"),
            snapshot_input_key=f"{data.plan.table}:parquet",
            snapshot_sha256=data.plan.snapshot_sha256,
            mapping_input_key=f"{data.plan.table}:mapping",
            mapping_version=data.plan.mapping_version,
            mapping_sha256=data.plan.mapping_sha256,
            minimum_rows=20,
            minimum_groups=5,
        )
    records = list(_records(data, set(plan.record_ids)))
    records.sort(key=lambda r: str(r.id))
    if len({r.dataset_revision for r in records}) != 1:
        raise ValueError("mixed_dataset_revision")
    samples = {
        r.id: extract_sample(r, data.plan, group_id=assignments[r.id].group_id, role="tune")
        for r in records
    }
    engine = RecommendationEngine({t.source_field: t.unit for t in data.plan.targets})
    context = EvaluationContext(
        analysis_run_id=run_id,
        record_level=data.mapping.record_level,
        allow_synthetic=data.plan.source_kind == "synthetic",
    )
    source_binding = DatasetBinding(
        schema_version="1.0",
        dataset_id=data.plan.dataset_id,
        dataset_revision=records[0].dataset_revision,
        source_sha256=data.plan.source_sha256,
        mapping_sha256=data.plan.mapping_sha256,
        snapshot_sha256=data.plan.snapshot_sha256,
    )
    write_json(destination / "original-source.json", source_binding.model_dump(mode="json"))
    summaries = {}
    for target in data.plan.targets:
        artifact = load_model_for_plan(
            data.private / "stage6/models" / target.name,
            plan.model_ids[target.name],
            data.plan,
            target,
            run_id=report.run_id,
            train_hash=report.targets[target.name]["train_sha256"],
        )
        model = ConditionalPipeline.from_state(
            json.loads(artifact.model_files["state.json"]), artifact.path("model.cbm")
        )
        features = [samples[r.id].features for r in records]
        cb = model.predict(features)
        base = model.predict(features, method="baseline")
        eligible = [r for r in records if samples[r.id].targets[target.name] is not None]
        if not eligible:
            raise ValueError("no_observed_mask_targets")
        modified_path = destination / f"{target.name}-modified.jsonl"
        labels_path = destination / f"{target.name}-labels.jsonl"
        with (
            modified_path.open("x", encoding="utf-8") as changed,
            labels_path.open("x", encoding="utf-8") as labels,
        ):
            for r in eligible:
                changed.write(canonical(_masked(r, target).model_dump(mode="json")) + "\n")
                labels.write(
                    canonical(
                        {
                            "record_id": str(r.id),
                            "field": target.source_field,
                            "group_id": assignments[r.id].group_id,
                            "reference": samples[r.id].targets[target.name],
                            "unit": target.unit,
                            "basis": "unverified_hidden_observation",
                        }
                    )
                    + "\n"
                )
        masked_binding = source_binding.model_copy(
            update={
                "dataset_revision": source_binding.dataset_revision + 1,
                "snapshot_sha256": sha256(modified_path),
            }
        )
        counts, ref_counts, suspicion_counts = Counter(), Counter(), Counter()
        numeric = []
        issued = 0
        with (
            (destination / f"{target.name}-recommendations.jsonl").open(
                "x", encoding="utf-8"
            ) as outputs,
            (destination / f"{target.name}-references.jsonl").open("x", encoding="utf-8") as refs,
            (destination / f"{target.name}-suspicions.jsonl").open(
                "x", encoding="utf-8"
            ) as suspicions,
            (destination / f"{target.name}-requests.jsonl").open("x", encoding="utf-8") as requests,
            (destination / f"{target.name}-findings.jsonl").open(
                "x", encoding="utf-8"
            ) as finding_outputs,
        ):
            for index, r in enumerate(records):
                reference = samples[r.id].targets[target.name]
                if reference is None:
                    continue
                masked = _masked(r, target)
                # Numeric target is absent from predictors and matching dimensions.
                # Full raw fingerprint is opaque copy-exclusion lineage, never a predictor.
                sample = extract_sample(
                    masked, data.plan, group_id=assignments[r.id].group_id, role="tune"
                )
                if sample.targets[target.name] is not None or sample.features != features[index]:
                    raise ValueError("masked_target_or_feature_leakage")
                query = _query(r, sample, assignments[r.id], target, data)
                comparable = pools[target.name].query(query)
                refs.write(
                    canonical(
                        {"record_id": str(r.id), "result": comparable.model_dump(mode="json")}
                    )
                    + "\n"
                )
                ref_counts[comparable.status] += 1
                providers = [
                    model_source(
                        cb[index],
                        model_id=plan.model_ids[target.name],
                        plan=data.plan,
                        target=target,
                        method="conditional_model",
                    ),
                    model_source(
                        base[index],
                        model_id=plan.model_ids[target.name],
                        plan=data.plan,
                        target=target,
                        method="baseline",
                    ),
                    _comparable(comparable, data),
                    growth_source(
                        GrowthTableRegistry(),
                        GrowthQuery(
                            schema_version="1.0",
                            region=sample.features["region"],
                            species=sample.features["species"],
                            indicator=target.name,
                            unit=target.unit,
                            age_unit="year",
                            age=sample.features["age"]
                            if sample.features["age"] is not None and sample.features["age"] >= 0
                            else None,
                        ),
                    ),
                ]
                arithmetic_sources, arithmetic_candidates = stock_sources(
                    masked,
                    target.source_field,
                    rules,
                    context,
                    data_root=f"observations:{data.plan.snapshot_sha256}",
                )
                candidates = tuple(c for _, c in providers if c is not None) + arithmetic_candidates
                sources = tuple(s for s, c in providers if c is None) + tuple(
                    s for s in arithmetic_sources if s.status != "available"
                )
                before_outcomes = rules.evaluate(masked, context)
                suspect = affected_fields(before_outcomes, rules)
                request = _finding(masked, target, run_id, masked=True)
                requests.write(request.model_dump_json() + "\n")
                result = engine.recommend(
                    masked,
                    request,
                    target.source_field,
                    candidates=candidates,
                    sources=sources,
                    precision=precision[target.name],
                    binding=masked_binding,
                    rule_engine=rules,
                    context=context,
                    suspect_fields=suspect,
                )
                outputs.write(result.model_dump_json() + "\n")
                reason = result.suggestion.refusal_reason or result.suggestion.status
                counts[reason] += 1
                if result.patch:
                    verify_patch(result.patch, masked, masked_binding)
                    issued += 1
                value = result.suggestion.proposed_value if result.patch else None
                numeric.append(
                    NumericCase(
                        schema_version="1.0",
                        record_id=r.id,
                        group_id=assignments[r.id].group_id,
                        field_name=target.name,
                        unit=target.unit,
                        reference=reference,
                        estimate=value,
                    )
                )
                actual_outcomes = rules.evaluate(r, context)
                actual_findings = field_findings(actual_outcomes, rules, target.source_field)
                outside = (
                    cb[index].status == "ok"
                    and cb[index].lower is not None
                    and not cb[index].lower <= reference <= cb[index].upper
                )
                if actual_findings or outside:
                    suspicion_counts["triggered_field_cases"] += 1
                    suspicion_counts["rule_suspicion"] += bool(actual_findings)
                    suspicion_counts["conditional_interval_suspicion"] += bool(outside)
                    triggers = list(actual_findings)
                    if outside:
                        triggers.append(_finding(r, target, run_id))
                    for trigger in triggers:
                        finding_outputs.write(trigger.model_dump_json() + "\n")
                    actual_suspect = affected_fields(actual_outcomes, rules)
                    actual = engine.recommend(
                        r,
                        triggers[0],
                        target.source_field,
                        candidates=candidates,
                        sources=sources,
                        precision=precision[target.name],
                        binding=source_binding,
                        rule_engine=rules,
                        context=context,
                        suspect_fields=actual_suspect,
                    )
                    if actual.patch:
                        verify_patch(actual.patch, r, source_binding)
                    suspicions.write(actual.model_dump_json() + "\n")
                    suspicion_counts[
                        "numeric_proposals" if actual.patch else "no_numeric_proposal"
                    ] += 1
        protocol = ScoringProtocol(
            schema_version="1.0",
            mode="masked_observation",
            label_basis="hidden_observation",
            evaluated_fields=(target.name,),
            evaluated_method_id="recommendations-dev-1.0.0",
        )
        metrics = score_frozen_cohort(
            protocol,
            data.manifest,
            tuple(r.id for r in eligible),
            numeric_cases=tuple(numeric),
            role="tune",
        )
        write_json(destination / f"{target.name}-metrics.json", metrics.model_dump(mode="json"))
        summaries[target.name] = {
            "masked_cases": len(eligible),
            "excluded_unobserved_targets": len(records) - len(eligible),
            "issued": issued,
            "abstained": len(eligible) - issued,
            "issuance_coverage": issued / len(eligible),
            "abstention_rate": 1 - issued / len(eligible),
            "reasons": dict(counts),
            "references": dict(ref_counts),
            "numeric": [m.model_dump(mode="json") for m in metrics.numeric],
            "groups": metrics.sample_groups,
            "real_suspicions": dict(suspicion_counts),
            "proven_corrections": 0,
            "range_coverage": "NOT_RUN_recommendation_intervals_not_calibrated",
        }
    data.verify_inputs()
    _, _, current_rulehash = _load_rules(data)
    if current_rulehash != plan.rule_plan_sha256:
        raise ValueError("rules_changed_during_run")
    result = {
        "schema_version": "1.0",
        "run_id": str(run_id),
        "plan_sha256": hash_value(plan.model_dump(mode="json")),
        "source_kind": data.plan.source_kind,
        "selected_records": len(records),
        "selected_groups": len(plan.group_ids),
        "targets": summaries,
        "proven_corrections": 0,
        "expert_metrics": "NOT_RUN_no_expert_labels",
        "holdout": "NOT_RUN",
        "calibration": "NOT_READ",
        "wall_seconds": time.perf_counter() - started,
        "originals_unchanged": True,
        "new_expert_labels": False,
        "patches_applied": 0,
        "limitations": [
            "source_observations_not_true_labels",
            "tune_previously_used_for_development",
            "growth_tables_and_codebooks_unavailable",
            "related_aggregate_contract_unavailable",
            "unknown_semantic_duplicates_not_proven_absent",
            "representation_step_not_measurement_accuracy",
        ],
    }
    write_json(destination / "report.json", result)
    write_json(
        destination / "outputs-manifest.json",
        {
            "schema_version": "1.0",
            "files": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()},
        },
    )
    return destination
