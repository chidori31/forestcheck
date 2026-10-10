"""One-shot private stage-10 evaluation; never invoked by training or the API."""

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from forestcheck.api.runtime import reference_pools
from forestcheck.evaluation.final import claim_final, verify_release
from forestcheck.evaluation.scoring import NumericCase, ScoringProtocol, score_evaluation
from forestcheck.evaluation.split import hash_value
from forestcheck.growth import GrowthQuery, GrowthTableRegistry
from forestcheck.ingest.csv_import import sha256
from forestcheck.ml.data import LocalData
from forestcheck.ml.local import load_model_for_plan, write_json
from forestcheck.ml.pipeline import ConditionalPipeline
from forestcheck.ml.resources import run_bounded
from forestcheck.repair.engine import RecommendationEngine
from forestcheck.repair.local import (
    _comparable,
    _finding,
    _load_rules,
    _masked,
    _model_report,
    _query,
    _records,
    affected_fields,
)
from forestcheck.repair.models import DatasetBinding
from forestcheck.repair.providers import growth_source, model_source, stock_sources
from forestcheck.rules.models import EvaluationContext

ROOT = Path(__file__).resolve().parents[1]


def candidate():
    paths = []
    for folder in ("backend/src", "frontend/src", "scripts", "tests", "frontend/e2e"):
        paths.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.suffix in {".py", ".ts", ".tsx", ".css", ".mjs", ".sql"}
        )
    paths.extend(
        ROOT / p
        for p in (
            "pyproject.toml",
            "uv.lock",
            "frontend/package.json",
            "frontend/package-lock.json",
        )
    )
    paths.extend(
        p for p in (ROOT / "configs").iterdir() if p.is_file() and not p.name.startswith("local.")
    )
    paths.extend((ROOT / "frontend").glob("*.ts"))
    paths.extend(ROOT / p for p in ("frontend/index.html", "frontend/eslint.config.js"))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def inputs(data):
    report = _model_report(data)
    rules, precision, rulehash = _load_rules(data)
    models, hashes = {}, {}
    train_groups = {a.group_id for a in data.manifest.assignments if a.role == "train"}
    for target in data.plan.targets:
        art = load_model_for_plan(
            data.private / "stage6/models" / target.name,
            report.model_ids[target.name],
            data.plan,
            target,
            run_id=report.run_id,
            train_hash=report.targets[target.name]["train_sha256"],
        )
        state = json.loads(art.model_files["state.json"])
        if not set(state["train_groups"]) <= train_groups:
            raise ValueError("model_trained_outside_train")
        models[target.name] = ConditionalPipeline.from_state(state, art.path("model.cbm"))
        hashes[target.name] = {
            "metadata": hash_value(art.metadata.model_dump(mode="json")),
            "state": hash_value(state),
            "native": sha256(art.path("model.cbm")),
        }
    assignments = {
        a.record_id: a for a in data.manifest.assignments if a.dataset_id == data.plan.dataset_id
    }
    ids = sorted(str(i) for i, a in assignments.items() if a.role == "holdout")
    groups = {assignments[UUID(i)].group_id for i in ids}
    if not ids or groups & (set(data.manifest.prior_exposed_groups) | train_groups):
        raise ValueError("holdout_exposure_or_overlap")
    release = {
        "schema_version": "1.0",
        "candidate": candidate(),
        "split": data.manifest.digest,
        "cohort": ids,
        "groups": sorted(groups),
        "plan": data.plan.digest,
        "rules": rulehash,
        "models": hashes,
        "mode": "masked_observation",
        "methods": ["rules", "rules_growth", "baseline", "full_pipeline"],
        "seed": data.plan.seed,
        "wall_seconds": 900,
        "memory_mib": 4096,
        "cpu_threads": 2,
        "scope": "all reserved internal holdout rows in the trained dataset; finite observed targets only",
        "growth": "unavailable_reference",
        "expert_metrics": "NOT_RUN_no_expert_labels",
        "uncertainty": "NOT_RUN_spatial_group_independence_unconfirmed",
    }
    return release, models, report, rules, precision, assignments


def features(record, plan):
    """Only frozen allowlisted predictors; no target, ID or evaluation label."""
    fields = {f.name: f for f in record.fields}
    result = {}
    if record.dataset_id != plan.dataset_id:
        raise ValueError("dataset_mismatch")
    for binding in plan.features:
        f = fields[binding.source_field]
        if f.unit != binding.unit:
            raise ValueError("unit_mismatch")
        v = f.normalized_value if f.state == "observed" else None
        if binding.kind == "numeric" and (type(v) not in (int, float) or not math.isfinite(v)):
            v = None
        if binding.kind != "numeric" and v is not None and not isinstance(v, str):
            raise ValueError("category_type_mismatch")
        result[binding.name] = v
    return result


def evaluate(data, stage, release, models, report, rules, precision, assignments):
    started = time.perf_counter()
    # Marker precedes any holdout record decoding, including on a failed run.
    claim_final(stage, hash_value(release))
    profile = SimpleNamespace(
        reference_parquet_path=str(data.folder / "normalized.parquet"),
        ml_plan=data.plan,
        split_manifest_path=str(data.private / "stage5/split-manifest.json"),
        record_level=data.mapping.record_level,
    )
    pools, _, _ = reference_pools(profile, lambda: None)
    records = sorted(_records(data, {UUID(i) for i in release["cohort"]}), key=lambda r: str(r.id))
    fs = [features(r, data.plan) for r in records]
    context = EvaluationContext(
        analysis_run_id=uuid4(),
        record_level=data.mapping.record_level,
        allow_synthetic=data.plan.source_kind == "synthetic",
    )
    engine = RecommendationEngine({t.source_field: t.unit for t in data.plan.targets})
    raw_counts = Counter(o.status for r in records for o in rules.evaluate(r, context))
    result = {
        "schema_version": "1.0",
        "freeze_sha256": hash_value(release),
        "reference_interpretation": "reconstruction_of_unverified_observation",
        "expert_metrics": "NOT_RUN_no_expert_labels",
        "growth": "unavailable_reference",
        "records": len(records),
        "groups": len(release["groups"]),
        "raw_rule_outcomes": dict(raw_counts),
        "targets": {},
    }
    for target in data.plan.targets:
        model = models[target.name]
        cb = model.predict(fs)
        base = model.predict(fs, method="baseline")
        # Repeat prediction stability is checked on precisely the same frozen input.
        if [p.model_dump() for p in cb] != [p.model_dump() for p in model.predict(fs)]:
            raise ValueError("repeat_prediction_changed")
        eligible = []
        for index, r in enumerate(records):
            field = next((f for f in r.fields if f.name == target.source_field), None)
            if (
                field
                and field.state == "observed"
                and field.unit == target.unit
                and type(field.normalized_value) in (int, float)
                and math.isfinite(field.normalized_value)
            ):
                eligible.append((index, r, field.normalized_value))
        modified = stage / f"{target.name}-modified.jsonl"
        labels = stage / f"{target.name}-labels.jsonl"
        with modified.open("x", encoding="utf-8") as m, labels.open("x", encoding="utf-8") as lab:
            for _, r, truth in eligible:
                m.write(_masked(r, target).model_dump_json() + "\n")
                lab.write(
                    json.dumps(
                        {
                            "record_id": str(r.id),
                            "reference": truth,
                            "unit": target.unit,
                            "group_id": assignments[r.id].group_id,
                            "basis": "unverified_hidden_observation",
                        }
                    )
                    + "\n"
                )
        sha_modified = sha256(modified)
        cases = {m: [] for m in release["methods"]}
        reasons = Counter()
        with (stage / f"{target.name}-predictions.jsonl").open("x", encoding="utf-8") as out:
            for index, r, truth in eligible:
                masked = _masked(r, target)
                if features(masked, data.plan) != fs[index]:
                    raise ValueError("masked_target_feature_leakage")
                a = assignments[r.id]
                q = _query(r, SimpleNamespace(features=fs[index]), a, target, data)
                comparable = pools[target.name].query_final(q, release=release)
                providers = [
                    model_source(
                        cb[index],
                        model_id=report.model_ids[target.name],
                        plan=data.plan,
                        target=target,
                        method="conditional_model",
                    ),
                    model_source(
                        base[index],
                        model_id=report.model_ids[target.name],
                        plan=data.plan,
                        target=target,
                        method="baseline",
                    ),
                    _comparable(comparable, data),
                    growth_source(
                        GrowthTableRegistry(),
                        GrowthQuery(
                            schema_version="1.0",
                            region=fs[index]["region"],
                            species=fs[index]["species"],
                            indicator=target.name,
                            unit=target.unit,
                            age_unit="year",
                            age=fs[index]["age"]
                            if fs[index]["age"] is not None and fs[index]["age"] >= 0
                            else None,
                        ),
                    ),
                ]
                ar_sources, ar_candidates = stock_sources(
                    masked,
                    target.source_field,
                    rules,
                    context,
                    data_root=f"observations:{data.plan.snapshot_sha256}",
                )
                binding = DatasetBinding(
                    schema_version="1.0",
                    dataset_id=r.dataset_id,
                    dataset_revision=masked.dataset_revision,
                    source_sha256=data.plan.source_sha256,
                    mapping_sha256=data.plan.mapping_sha256,
                    snapshot_sha256=sha_modified,
                )
                request = _finding(masked, target, context.analysis_run_id, masked=True)
                suspects = affected_fields(rules.evaluate(masked, context), rules)

                def recommend(candidates, sources):
                    return engine.recommend(
                        masked,
                        request,
                        target.source_field,
                        candidates=tuple(candidates),
                        sources=tuple(sources),
                        precision=precision[target.name],
                        binding=binding,
                        rule_engine=rules,
                        context=context,
                        suspect_fields=suspects,
                    )

                rule_answer = recommend(
                    ar_candidates, [s for s in ar_sources if s.status != "available"]
                )
                full = recommend(
                    tuple(c for _, c in providers if c is not None) + ar_candidates,
                    tuple(s for s, c in providers if c is None)
                    + tuple(s for s in ar_sources if s.status != "available"),
                )
                reasons[full.suggestion.refusal_reason or full.suggestion.status] += 1
                vals = {
                    "rules": rule_answer.suggestion.proposed_value if rule_answer.patch else None,
                    "rules_growth": rule_answer.suggestion.proposed_value
                    if rule_answer.patch
                    else None,
                    "baseline": base[index].value if base[index].status == "ok" else None,
                    "full_pipeline": full.suggestion.proposed_value if full.patch else None,
                }
                for method, value in vals.items():
                    low = base[index].lower if method == "baseline" and value is not None else None
                    high = base[index].upper if method == "baseline" and value is not None else None
                    if (
                        method == "full_pipeline"
                        and value is not None
                        and full.suggestion.candidate_range
                    ):
                        low = full.suggestion.candidate_range.lower
                        high = full.suggestion.candidate_range.upper
                    cases[method].append(
                        NumericCase(
                            schema_version="1.0",
                            record_id=r.id,
                            group_id=a.group_id,
                            field_name=target.name,
                            unit=target.unit,
                            reference=truth,
                            estimate=value,
                            interval_lower=low,
                            interval_upper=high,
                        )
                    )
                out.write(
                    json.dumps(
                        {
                            "record_id": str(r.id),
                            "values": vals,
                            "recommendation": full.model_dump(mode="json"),
                            "conditional_prediction": cb[index].model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summaries = {}
        for method, numeric in cases.items():
            protocol = ScoringProtocol(
                schema_version="1.0",
                mode="masked_observation",
                label_basis="hidden_observation",
                evaluated_fields=(target.name,),
                evaluated_method_id=method,
            )
            summaries[method] = score_evaluation(protocol, numeric_cases=tuple(numeric)).model_dump(
                mode="json"
            )
        result["targets"][target.name] = {
            "eligible": len(eligible),
            "excluded": len(records) - len(eligible),
            "recommendation_reasons": dict(reasons),
            "methods": summaries,
            "rules_growth_status": "BLOCKED_REFERENCE_same_available_rules_only",
        }
    data.verify_inputs()
    result["wall_seconds"] = time.perf_counter() - started
    result["libraries"] = {
        n: importlib.metadata.version(n)
        for n in ("catboost", "numpy", "pyarrow", "scikit-learn", "pydantic")
    }
    result["hardware"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    write_json(stage / "evaluation.json", result)
    print("FINAL_EVALUATION_PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "evaluate", "_worker"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    data = LocalData(args.config, workspace_root=ROOT)
    stage = data.private / "stage10"
    stage.mkdir(exist_ok=True)
    release, models, report, rules, precision, assignments = inputs(data)
    frozen = stage / "freeze.json"
    if args.command == "prepare":
        if (stage / "consumed.json").exists():
            raise ValueError("holdout_already_consumed")
        write_json(frozen, release)
        print("CANDIDATE_FROZEN")
    else:
        verify_release(json.loads(frozen.read_text("utf-8")), release)
        if args.command == "_worker":
            evaluate(data, stage, release, models, report, rules, precision, assignments)
        else:
            logs = stage / "resource-logs"
            logs.mkdir()
            measured = run_bounded(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "_worker",
                    "--config",
                    str(args.config.resolve()),
                ],
                cwd=ROOT,
                log_dir=logs,
                wall_seconds=900,
                memory_mib=4096,
                cpu_threads=2,
            )
            write_json(stage / "resources.json", measured)
            print("FINAL_" + measured["status"].upper())
            return 0 if measured["status"] == "success" else 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Private diagnostic, no source values or paths on stdout.
        import traceback

        cfg = Path(sys.argv[sys.argv.index("--config") + 1])
        private = LocalData(cfg, workspace_root=ROOT).private / "stage10"
        (private / "failure.txt").write_text(traceback.format_exc(), "utf-8")
        print("AUDIT_FAILED")
        raise SystemExit(3)
