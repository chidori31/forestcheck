"""Local observation-model experiments; explicit frozen cohorts, no holdout evaluation."""

import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Literal
from uuid import UUID

from forestcheck.domain.contracts import Contract, Digest
from forestcheck.evaluation.annotation import build_annotation_queue
from forestcheck.evaluation.scoring import NumericCase, ScoringProtocol, score_frozen_cohort
from forestcheck.evaluation.split import canonical, hash_value
from forestcheck.ingest.csv_import import regular_output
from forestcheck.ml.artifacts import (
    ArtifactCompatibility,
    ArtifactMetadata,
    initialize_registry,
    load_candidate,
    register_candidate,
)
from forestcheck.ml.data import LocalData, MLPlan
from forestcheck.ml.isolation import IsolationModel, eligible_isolation_indices, fit_isolation
from forestcheck.ml.pipeline import (
    ConditionalPipeline,
    TargetSpec,
    _category,
    eligible_training_indices,
    fit_pipeline,
    robust_deviation,
)


class RunReport(Contract):
    status: Literal["trained_candidate"]
    run_id: UUID
    plan_sha256: Digest
    model_ids: dict[str, UUID]
    targets: dict[str, dict]
    oof: dict
    holdout_status: Literal["NOT_RUN"]
    classification_metrics: Literal["NOT_RUN_no_expert_labels"]
    promoted: Literal[False]
    isolation: dict
    reload_max_abs_difference: float
    reload_absolute_tolerance: float
    worker_wall_seconds: float
    library_versions: dict[str, str]


class IsolationContext(Contract):
    support: dict[str, tuple[str, ...]]


def load_model_for_plan(registry, model_id, plan, target, *, run_id=None, train_hash=None):
    artifact = load_candidate(registry, model_id, expected=compatibility(plan, target))
    metadata = artifact.metadata
    if (
        metadata.split_sha256 != plan.split_sha256
        or MLPlan.model_validate(metadata.config.get("plan")).digest != plan.digest
        or metadata.seed != plan.seed
        or metadata.source_kind != plan.source_kind
        or metadata.feature_lineage_sha256
        != hash_value([f.model_dump(mode="json") for f in plan.features])
        or TargetSpec.model_validate(metadata.config.get("target_spec"))
        != target_spec(plan, target)
        or (run_id is not None and metadata.config.get("run_id") != str(run_id))
        or (train_hash is not None and metadata.train_sha256 != train_hash)
    ):
        raise ValueError("artifact_plan_binding")
    return artifact


def write_json(path: Path, value):
    content = canonical(value) + "\n"
    if path.exists():
        if not regular_output(path) or path.read_text("utf-8") != content:
            raise ValueError("existing_output_differs")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)


def directory(path):
    path.mkdir(exist_ok=True)
    if path.resolve() != path or path.is_symlink():
        raise ValueError("private_directory_link")
    return path


def libraries():
    return {
        "python": platform.python_version(),
        **{
            n: importlib.metadata.version(n)
            for n in ("catboost", "scikit-learn", "numpy", "pydantic", "pyarrow", "psutil")
        },
    }


def target_spec(plan, target):
    return TargetSpec(
        schema_version="1.0",
        target=target.name,
        unit=target.unit,
        numeric_features=tuple(f.name for f in plan.features if f.kind == "numeric"),
        categorical_features=tuple(f.name for f in plan.features if f.kind == "categorical"),
        context_fields=tuple(f.name for f in plan.features if f.kind == "context"),
        required_categories=tuple(f.name for f in plan.features if f.kind == "categorical"),
        min_train_rows=40,
        min_train_groups=3,
        min_conditional_rows=20,
    )


def compatibility(plan, target):
    return ArtifactCompatibility(
        schema_version="1.0",
        record_schema_version="1.0",
        mapping_version=plan.mapping_version,
        mapping_sha256=plan.mapping_sha256,
        features=target_spec(plan, target).feature_names,
        target=target.name,
        target_unit=target.unit,
        library_versions=libraries(),
    )


def _fit(samples, target, spec, plan, *, role="train", candidate=True):
    return fit_pipeline(
        [s.features for s in samples],
        [s.targets[target.name] for s in samples],
        [s.group_id for s in samples],
        spec,
        role=role,
        seed=plan.seed,
        iterations=plan.iterations,
        thread_count=plan.cpu_threads,
        train_candidate=candidate,
    )


def _metrics(data, samples, predictions, target, method, role, *, intervals=False):
    cases = tuple(
        NumericCase(
            schema_version="1.0",
            record_id=s.record_id,
            group_id=s.group_id,
            field_name=target.name,
            unit=target.unit,
            reference=s.targets[target.name],
            estimate=p.value,
            interval_lower=p.lower if intervals else None,
            interval_upper=p.upper if intervals else None,
        )
        for s, p in zip(samples, predictions, strict=True)
    )
    protocol = ScoringProtocol(
        schema_version="1.0",
        mode="masked_observation",
        label_basis="hidden_observation",
        evaluated_fields=(target.name,),
        evaluated_method_id=method,
    )
    report = score_frozen_cohort(
        protocol, data.manifest, tuple(s.record_id for s in samples), numeric_cases=cases, role=role
    )
    return report


def _summary(report):
    return {
        "records": report.sample_records,
        "groups": report.sample_groups,
        "numeric": [r.model_dump(mode="json") for r in report.numeric],
        "reference": report.reference_interpretation,
        "uncertainty": "NOT_RUN_independence_unconfirmed",
    }


def _isolation_rows(samples, plan):
    numeric = [f.name for f in plan.features if f.kind == "numeric"]
    names = tuple(numeric + [t.name for t in plan.targets])
    return names, [
        {name: (s.features[name] if name in s.features else s.targets[name]) for name in names}
        for s in samples
    ]


def build_isolation_context(samples, plan, eligible_indices):
    return IsolationContext(
        schema_version="1.0",
        support={
            f.name: tuple(
                sorted(
                    {
                        _category(samples[i].features[f.name])
                        for i in eligible_indices
                        if samples[i].features[f.name] is not None
                    }
                )
            )
            for f in plan.features
            if f.kind in {"categorical", "context"}
        },
    )


def guard_isolation_context(scores, samples, context):
    # Context support comes from precisely the rows used by this IF fit.
    output = []
    for score, sample in zip(scores, samples, strict=True):
        unknown = any(
            sample.features[n] is None or _category(sample.features[n]) not in values
            for n, values in context.support.items()
        )
        result = score.model_dump(mode="json")
        if unknown:
            result = {
                "status": "abstained",
                "score": None,
                "rank_quantile": None,
                "warnings": ["unknown_or_missing_region_or_species"],
            }
        output.append(result)
    return output


def train_run(config_path: Path, *, workspace_root: Path, run_id: UUID) -> dict:
    started = time.perf_counter()
    data = LocalData(config_path, workspace_root=workspace_root)
    plan = data.plan
    stage = directory(data.private / "stage6")
    runs = directory(stage / "runs")
    destination = runs / str(run_id)
    if destination.exists():
        raise ValueError("explicit_new_run_id_required")
    directory(destination)
    write_json(destination / "plan.json", plan.model_dump(mode="json"))
    samples = data.load({"train", "tune"})
    train = [s for s in samples if s.role == "train"]
    dev = [s for s in samples if s.role == "tune"]
    if not train or not dev:
        raise ValueError("training_and_tune_required")
    write_json(
        destination / "cohorts.json",
        {
            "schema_version": "1.0",
            "split_digest": data.manifest.digest,
            "train": [str(s.record_id) for s in train],
            "tune": [str(s.record_id) for s in dev],
            "train_sha256": hash_value([str(s.record_id) for s in train]),
            "tune_sha256": hash_value([str(s.record_id) for s in dev]),
            "calibration_fit_groups": list(plan.calibration_fit_groups),
            "calibration_check_groups": list(plan.calibration_check_groups),
            "holdout": "NOT_RUN",
        },
    )
    models, results, oof_by_target, fold_audit = {}, {}, {}, []
    for target in plan.targets:
        spec = target_spec(plan, target)
        begin = time.perf_counter()
        baseline = _fit(train, target, spec, plan, candidate=False)
        baseline_wall = time.perf_counter() - begin
        begin = time.perf_counter()
        candidate = _fit(train, target, spec, plan)
        candidate_wall = time.perf_counter() - begin
        accepted_indices = eligible_training_indices(
            [s.features for s in train], [s.targets[target.name] for s in train], spec
        )
        baseline_dev = baseline.baseline.predict([s.features for s in dev], calibrated=False)
        candidate_dev = candidate.predict([s.features for s in dev], calibrated=False)
        results[target.name] = {
            "unit": target.unit,
            "train_observations": candidate.train_count,
            "train_sha256": hash_value([str(train[i].record_id) for i in accepted_indices]),
            "baseline_fit_seconds": baseline_wall,
            "candidate_fit_seconds": candidate_wall,
            "reference": "unverified_source_observation",
            "dev": {},
        }
        for method, pred in (("baseline", baseline_dev), ("catboost", candidate_dev)):
            report = _metrics(data, dev, pred, target, method, "tune")
            results[target.name]["dev"][method] = _summary(report)
            write_json(
                destination / f"{target.name}-dev-{method}.json", report.model_dump(mode="json")
            )
        # Freeze candidate decision/configuration before loading calibration features.
        models[target.name] = candidate
        oof = {}
        begin = time.perf_counter()
        for fold in range(3):
            fitting = [s for s in train if plan.train_folds[s.group_id] != fold]
            validation = [s for s in train if plan.train_folds[s.group_id] == fold]
            if {s.group_id for s in fitting} & {s.group_id for s in validation}:
                raise ValueError("oof_group_overlap")
            fitted = _fit(fitting, target, spec, plan, role="train_fold")
            cb = fitted.predict([s.features for s in validation], calibrated=False)
            base = fitted.baseline.predict([s.features for s in validation], calibrated=False)
            for s, p, b in zip(validation, cb, base, strict=True):
                if s.record_id in oof:
                    raise ValueError("oof_duplicate")
                oof[s.record_id] = (p, b, fold)
            fold_audit.append(
                {
                    "target": target.name,
                    "fold": fold,
                    "train_groups": len({s.group_id for s in fitting}),
                    "validation_groups": len({s.group_id for s in validation}),
                    "input_record_hash": hash_value([str(s.record_id) for s in fitting]),
                    "fit_record_hash": hash_value(
                        [
                            str(fitting[i].record_id)
                            for i in eligible_training_indices(
                                [s.features for s in fitting],
                                [s.targets[target.name] for s in fitting],
                                spec,
                            )
                        ]
                    ),
                    "fit_records": fitted.train_count,
                    "validation_record_hash": hash_value([str(s.record_id) for s in validation]),
                    "overlap": False,
                }
            )
        if set(oof) != {s.record_id for s in train}:
            raise ValueError("oof_incomplete")
        results[target.name]["oof_seconds"] = time.perf_counter() - begin
        results[target.name]["oof"] = {}
        for method, index in (("catboost", 0), ("baseline", 1)):
            report = _metrics(
                data, train, [oof[s.record_id][index] for s in train], target, method, "train"
            )
            results[target.name]["oof"][method] = _summary(report)
            write_json(
                destination / f"{target.name}-oof-{method}.json", report.model_dump(mode="json")
            )
        oof_by_target[target.name] = oof
    write_json(
        destination / "candidate-selection.json",
        {
            "schema_version": "1.0",
            "policy": "One prespecified candidate; no search or auto-promotion",
            "plan_sha256": plan.digest,
            "dev_results": results,
            "calibration_not_yet_loaded": True,
        },
    )

    # Additional signal: numeric IF, with separate group-OOF scores for train.
    isolation_names, train_numeric = _isolation_rows(train, plan)
    begin = time.perf_counter()
    isolation = fit_isolation(
        train_numeric,
        groups=[s.group_id for s in train],
        feature_names=isolation_names,
        seed=plan.seed,
    )
    isolation_wall = time.perf_counter() - begin
    isolation_eligible = eligible_isolation_indices(train_numeric, isolation_names)
    isolation_context = build_isolation_context(train, plan, isolation_eligible)
    isolation_folds = []
    isolation_oof = {}
    for fold in range(3):
        fitting = [s for s in train if plan.train_folds[s.group_id] != fold]
        validation = [s for s in train if plan.train_folds[s.group_id] == fold]
        _, fitrows = _isolation_rows(fitting, plan)
        _, valrows = _isolation_rows(validation, plan)
        fitted = fit_isolation(
            fitrows,
            groups=[s.group_id for s in fitting],
            feature_names=isolation_names,
            seed=plan.seed,
            role="train_fold",
        )
        eligible = eligible_isolation_indices(fitrows, isolation_names)
        context = build_isolation_context(fitting, plan, eligible)
        guarded = guard_isolation_context(fitted.predict(valrows), validation, context)
        for s, value in zip(validation, guarded, strict=True):
            isolation_oof[s.record_id] = value
        isolation_folds.append(
            {
                "fold": fold,
                "fit_rows": fitted.train_rows,
                "fit_record_hash": hash_value([str(fitting[i].record_id) for i in eligible]),
                "validation_record_hash": hash_value([str(s.record_id) for s in validation]),
                "overlap": False,
            }
        )
    if set(isolation_oof) != {s.record_id for s in train}:
        raise ValueError("isolation_oof_incomplete")
    with (destination / "train-oof-predictions.jsonl").open("x", encoding="utf-8") as stream:
        for sample in train:
            row = {
                "record_id": str(sample.record_id),
                "group_id": sample.group_id,
                "fold": plan.train_folds[sample.group_id],
                "targets": {},
                "isolation": isolation_oof[sample.record_id],
            }
            for target in plan.targets:
                cb, base, _ = oof_by_target[target.name][sample.record_id]
                row["targets"][target.name] = {
                    "catboost": cb.model_dump(mode="json"),
                    "baseline": base.model_dump(mode="json"),
                    "robust_deviation": robust_deviation(sample.targets[target.name], base),
                }
            stream.write(canonical(row) + "\n")
    del oof_by_target, isolation_oof

    calibration = data.load({"calibration"})
    fit_groups = set(plan.calibration_fit_groups)
    check_groups = set(plan.calibration_check_groups)
    calfit = [s for s in calibration if s.group_id in fit_groups]
    calcheck = [s for s in calibration if s.group_id in check_groups]
    if not calfit or not calcheck:
        raise ValueError("independent_calibration_parts_required")
    registry_parent = directory(stage / "models")
    model_ids = {}
    reload_max = 0.0
    for target in plan.targets:
        model = models[target.name]
        results[target.name]["calibration"] = {}
        results[target.name]["calibration_check"] = {}
        for method in ("baseline", "catboost"):
            info = model.calibrate(
                [s.features for s in calfit],
                [s.targets[target.name] for s in calfit],
                [s.group_id for s in calfit],
                method=method,
                min_groups=20,
            )
            results[target.name]["calibration"][method] = info
            pred = model.predict([s.features for s in calcheck], method=method)
            report = _metrics(data, calcheck, pred, target, method, "calibration", intervals=True)
            results[target.name]["calibration_check"][method] = _summary(report)
            write_json(
                destination / f"{target.name}-calcheck-{method}.json",
                report.model_dump(mode="json"),
            )
        native = destination / f"{target.name}.cbm"
        model.save_catboost(native)
        registry = registry_parent / target.name
        if not registry.exists():
            directory(registry)
            initialize_registry(registry)
        metadata = ArtifactMetadata(
            schema_version="1.0",
            method_id="catboost_quantiles_with_separate_baseline_and_if",
            config={
                "plan": plan.model_dump(mode="json"),
                "target_spec": model.spec.model_dump(mode="json"),
                "run_id": str(run_id),
            },
            library_versions=libraries(),
            seed=plan.seed,
            record_schema_version="1.0",
            mapping_version=plan.mapping_version,
            mapping_sha256=plan.mapping_sha256,
            split_sha256=plan.split_sha256,
            train_sha256=results[target.name]["train_sha256"],
            features=model.spec.feature_names,
            feature_lineage_sha256=hash_value([f.model_dump(mode="json") for f in plan.features]),
            target=target.name,
            target_unit=target.unit,
            dev_cohort_sha256=hash_value([str(s.record_id) for s in dev]),
            dev_report=results[target.name]["dev"],
            source_kind=plan.source_kind,
        )
        model_id = register_candidate(
            registry,
            metadata,
            {
                "model.cbm": native.read_bytes(),
                "state.json": canonical(model.export_state()).encode(),
                "isolation.json": isolation.model_dump_json().encode(),
                "isolation_context.json": isolation_context.model_dump_json().encode(),
            },
        )
        model_ids[target.name] = str(model_id)
        loaded = load_candidate(registry, model_id, expected=compatibility(plan, target))
        restored = ConditionalPipeline.from_state(
            json.loads(loaded.model_files["state.json"]), loaded.path("model.cbm")
        )
        original_predictions = model.predict([s.features for s in dev])
        repeated_predictions = restored.predict([s.features for s in dev])
        for a, b in zip(original_predictions, repeated_predictions, strict=True):
            if a.status != b.status or a.warnings != b.warnings:
                raise ValueError("reload_status_mismatch")
            for key in ("value", "lower", "upper"):
                av, bv = getattr(a, key), getattr(b, key)
                if av is None:
                    if bv is not None:
                        raise ValueError("reload_missing_mismatch")
                else:
                    reload_max = max(reload_max, abs(av - bv))
        if reload_max > 1e-9:
            raise ValueError("reload_prediction_tolerance")
    data.verify_inputs()
    _, dev_numeric = _isolation_rows(dev, plan)
    restored_isolation = IsolationModel.model_validate_json(isolation.model_dump_json())
    isolation_reload_max = 0.0
    for a, b in zip(
        isolation.predict(dev_numeric), restored_isolation.predict(dev_numeric), strict=True
    ):
        if a.status != b.status or a.warnings != b.warnings or a.rank_quantile != b.rank_quantile:
            raise ValueError("isolation_reload_mismatch")
        if a.score is not None:
            isolation_reload_max = max(isolation_reload_max, abs(a.score - b.score))
    if isolation_reload_max > 1e-10:
        raise ValueError("isolation_reload_tolerance")
    report = {
        "schema_version": "1.0",
        "status": "trained_candidate",
        "run_id": str(run_id),
        "plan_sha256": plan.digest,
        "model_ids": model_ids,
        "targets": results,
        "oof": {"all_train_records_once": True, "folds": fold_audit},
        "holdout_status": "NOT_RUN",
        "classification_metrics": "NOT_RUN_no_expert_labels",
        "promoted": False,
        "isolation": {
            "method": "IsolationForest",
            "fit_seconds": isolation_wall,
            "train_rows": isolation.train_rows,
            "train_sha256": hash_value([str(train[i].record_id) for i in isolation_eligible]),
            "oof_folds": isolation_folds,
            "reload_max_abs_difference": isolation_reload_max,
            "reload_absolute_tolerance": 1e-10,
            "features": list(isolation_names),
            "score_meaning": "negative sklearn score_samples; empirical rank, not error probability",
            "threshold": "NONE; explicit queue depth only",
            "train_inference": "group_OOF",
        },
        "reload_max_abs_difference": reload_max,
        "reload_absolute_tolerance": 1e-9,
        "worker_wall_seconds": time.perf_counter() - started,
        "library_versions": libraries(),
    }
    RunReport.model_validate(report)
    write_json(destination / "report.json", report)
    return report


def predict_run(config_path: Path, *, workspace_root: Path, run_id: UUID, role="tune") -> Path:
    if role not in {"tune", "calibration_check"}:
        raise ValueError("holdout_forbidden_or_train_requires_stored_oof")
    data = LocalData(config_path, workspace_root=workspace_root)
    plan = data.plan
    destination = data.private / "stage6/runs" / str(run_id)
    if destination.resolve() != destination or not destination.is_dir():
        raise ValueError("run_directory")
    from forestcheck.rules.local import read_private_json

    report = RunReport.model_validate(read_private_json(destination / "report.json"))
    if (
        report.plan_sha256 != plan.digest
        or report.run_id != run_id
        or set(report.model_ids) != {t.name for t in plan.targets}
        or set(report.targets) != set(report.model_ids)
    ):
        raise ValueError("run_plan_mismatch")
    samples = data.load({"tune"} if role == "tune" else {"calibration"})
    if role == "calibration_check":
        allowed = set(plan.calibration_check_groups)
        samples = [s for s in samples if s.group_id in allowed]
    per_target = {}
    context = None
    isolation = None
    for target in plan.targets:
        artifact = load_model_for_plan(
            data.private / "stage6/models" / target.name,
            report.model_ids[target.name],
            plan,
            target,
            run_id=run_id,
            train_hash=report.targets[target.name]["train_sha256"],
        )
        model = ConditionalPipeline.from_state(
            json.loads(artifact.model_files["state.json"]), artifact.path("model.cbm")
        )
        if model.spec != target_spec(plan, target):
            raise ValueError("model_spec_binding")
        per_target[target.name] = (
            model.predict([s.features for s in samples]),
            model.baseline.predict([s.features for s in samples]),
        )
        if isolation is None:
            isolation = IsolationModel.model_validate_json(artifact.model_files["isolation.json"])
            context = IsolationContext.model_validate_json(
                artifact.model_files["isolation_context.json"]
            )
            if set(context.support) != {
                f.name for f in plan.features if f.kind in {"categorical", "context"}
            }:
                raise ValueError("isolation_context_binding")
    _, numeric = _isolation_rows(samples, plan)
    scores = guard_isolation_context(isolation.predict(numeric), samples, context)
    out = destination / f"{role}-predictions.jsonl"
    # Compare repeated bytes; never replace an existing prediction file silently.
    temporary = destination / f"{role}-predictions.check.jsonl"
    with temporary.open("x", encoding="utf-8") as stream:
        for i, s in enumerate(samples):
            stream.write(
                canonical(
                    {
                        "record_id": str(s.record_id),
                        "group_id": s.group_id,
                        "targets": {
                            t.name: {
                                "catboost": per_target[t.name][0][i].model_dump(mode="json"),
                                "baseline": per_target[t.name][1][i].model_dump(mode="json"),
                                "robust_deviation": robust_deviation(
                                    s.targets[t.name], per_target[t.name][1][i]
                                ),
                            }
                            for t in plan.targets
                        },
                        "isolation": scores[i],
                        "warning": "Measurement provenance unverified; not an error probability",
                    }
                )
                + "\n"
            )
    if out.exists():
        from forestcheck.ingest.csv_import import sha256

        if not regular_output(out) or sha256(out) != sha256(temporary):
            raise ValueError("repeat_prediction_mismatch")
        temporary.unlink()
    else:
        temporary.rename(out)
    ranked = sorted(
        ((s, p) for s, p in zip(samples, scores, strict=True) if p["status"] == "scored"),
        key=lambda pair: (-pair[1]["score"], str(pair[0].record_id)),
    )[: plan.queue_depth]
    write_json(
        destination / f"{role}-queue.json",
        {
            "schema_version": "1.0",
            "depth": plan.queue_depth,
            "meaning": "Review workload, not inferred fraction of errors",
            "expert_answers": "NOT_PROVIDED",
            "items": [
                {
                    "record_id": str(s.record_id),
                    "score": p["score"],
                    "rank_quantile": p["rank_quantile"],
                }
                for s, p in ranked
            ],
        },
    )
    if role == "tune":
        # Leave at least one random control even for a very small permitted cohort.
        ranked_ids = tuple(s.record_id for s, _ in ranked[: max(0, len(samples) - 1)])
        control_count = min(20, len(samples) - len(ranked_ids))
        if control_count < 1:
            raise ValueError("annotation_controls_unavailable")
        queue = build_annotation_queue(
            tuple(s.record_id for s in samples),
            flagged=ranked_ids,
            manifest=data.manifest,
            seed=f"stage6:{plan.seed}:{plan.digest}",
            control_count=control_count,
            max_flagged=plan.queue_depth,
        )
        write_json(
            destination / "tune-annotation-forms.json",
            [f.model_dump(mode="json") for f in queue.forms],
        )
        write_json(
            destination / "tune-annotation-sampling.json",
            {
                "schema_version": "1.0",
                "split_digest": queue.split_digest,
                "seed": queue.seed,
                "sampling": [s.model_dump(mode="json") for s in queue.sampling],
                "expert_status": queue.expert_status,
            },
        )
    data.verify_inputs()
    return out


def evaluate_run(config_path: Path, *, workspace_root: Path, run_id: UUID, role="tune") -> Path:
    path = predict_run(config_path, workspace_root=workspace_root, run_id=run_id, role=role)
    data = LocalData(config_path, workspace_root=workspace_root)
    samples = data.load({"tune"} if role == "tune" else {"calibration"})
    if role == "calibration_check":
        selected = set(data.plan.calibration_check_groups)
        samples = [s for s in samples if s.group_id in selected]
    from forestcheck.ml.pipeline import Prediction

    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    if [r["record_id"] for r in rows] != [str(s.record_id) for s in samples]:
        raise ValueError("evaluation_cohort_mismatch")
    summary = {"schema_version": "1.0", "role": role, "holdout": "NOT_RUN", "targets": {}}
    for target in data.plan.targets:
        summary["targets"][target.name] = {}
        for method in ("catboost", "baseline"):
            predictions = [
                Prediction.model_validate(r["targets"][target.name][method]) for r in rows
            ]
            report = _metrics(
                data,
                samples,
                predictions,
                target,
                method,
                "tune" if role == "tune" else "calibration",
                intervals=role == "calibration_check",
            )
            summary["targets"][target.name][method] = _summary(report)
    output = path.parent / f"{role}-evaluation.json"
    write_json(output, summary)
    return output
