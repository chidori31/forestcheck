import json
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest
from forestcheck.evaluation.split import canonical
from forestcheck.ingest.csv_import import sha256
from forestcheck.ml.data import LocalData
from forestcheck.ml.local import train_run
from forestcheck.repair.local import prepare, run
from forestcheck.rules.local import DevelopmentPlan, DevelopmentSplit, SplitInput
from forestcheck.rules.models import RuleDefinition
from test_ml_local import prepared


def environment(tmp_path):
    config, repo, private, source = prepared(tmp_path)
    data = LocalData(config, workspace_root=repo)
    p = data.plan
    stage4 = private / "stage4"
    stage4.mkdir()
    split = DevelopmentSplit(
        schema_version="1.0",
        split_id="synthetic",
        inputs={
            p.table: SplitInput(
                dataset_id=p.dataset_id,
                source_sha256=p.source_sha256,
                mapping_sha256=p.mapping_sha256,
                parquet_sha256=p.snapshot_sha256,
            )
        },
        assignments={p.table: ()},
    )
    splitpath = stage4 / "development-split.json"
    splitpath.write_text(canonical(split.model_dump(mode="json")))
    rules = []
    for t in p.targets:
        rules.append(
            RuleDefinition(
                schema_version="1.0",
                id=t.name + "_step",
                version="1",
                record_levels=("source_row",),
                dependencies=(t.source_field,),
                source=dict(reference_id="synthetic", locator="fixture", kind="synthetic"),
                severity="warning",
                priority=1,
                explanation="Synthetic rounding",
                check=dict(kind="rounding", field=t.source_field, unit=t.unit, step="1"),
            )
        )
    ruleplan = DevelopmentPlan(
        schema_version="1.0",
        version="synthetic",
        split_sha256=sha256(splitpath),
        rules_by_table={p.table: tuple(rules)},
        composition_fields={},
        blocked_requirements=("No growth table",),
    )
    (stage4 / "rule-plan.json").write_text(ruleplan.model_dump_json())
    run_id = UUID(int=12345)
    report = train_run(config, workspace_root=repo, run_id=run_id)
    (private / "stage6/verification.json").write_text(
        json.dumps({"run_id": str(run_id), "plan_sha256": p.digest})
    )
    return config, repo, private, source, report


def test_masked_dev_roundtrip_keeps_raw_and_labels_separate(tmp_path):
    config, repo, private, source, _ = environment(tmp_path)
    before = sha256(source)
    plan = prepare(config, workspace_root=repo)
    assert plan.is_file()
    assert prepare(config, workspace_root=repo) == plan
    result = run(config, workspace_root=repo)
    report = json.loads((result / "report.json").read_text("utf-8"))
    assert report["holdout"] == "NOT_RUN" and report["proven_corrections"] == 0
    assert report["expert_metrics"] == "NOT_RUN_no_expert_labels"
    assert set(report["targets"]) == {"height", "diameter"}
    for name, value in report["targets"].items():
        assert value["issued"] + value["abstained"] == value["masked_cases"] > 0
        predictions = [
            json.loads(line)
            for line in (result / f"{name}-recommendations.jsonl").read_text("utf-8").splitlines()
        ]
        assert len(predictions) == value["masked_cases"]
        assert all(r["suggestion"]["before"]["original_value"] is None for r in predictions)
        assert all(r["suggestion"]["before"]["state"] == "missing" for r in predictions)
        labels = [
            json.loads(line)
            for line in (result / f"{name}-labels.jsonl").read_text("utf-8").splitlines()
        ]
        assert len(labels) == len(predictions) and all(r["reference"] is not None for r in labels)
        assert all("labels" not in r["suggestion"] for r in predictions)
        requests = [
            json.loads(line)
            for line in (result / f"{name}-requests.jsonl").read_text("utf-8").splitlines()
        ]
        assert len(requests) == len(predictions)
        assert all(
            f["signal_kind"] == "missing" and f["review_status"] == "unreviewed" for f in requests
        )
        assert (result / f"{name}-findings.jsonl").is_file()
    assert sha256(source) == before


def test_frozen_cohort_rejects_tamper_before_evaluation(tmp_path):
    config, repo, private, _, _ = environment(tmp_path)
    path = prepare(config, workspace_root=repo)
    content = json.loads(path.read_text("utf-8"))
    content["seed"] = "changed_after_freeze"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        run(config, workspace_root=repo)


def test_cli_invalid_path_has_no_traceback_or_private_values(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/recommendations_local.py"
    output = subprocess.run(
        [sys.executable, str(script), "evaluate", "--config", str(tmp_path / "PRIVATE")],
        capture_output=True,
        text=True,
    )
    assert output.returncode == 3 and output.stdout.splitlines() == ["RECOMMENDATIONS_FAILED"]
    assert output.stderr == ""


def test_multifield_findings_mark_all_dependencies_suspicious():
    from forestcheck.repair.local import affected_fields, field_findings
    from forestcheck.rules.engine import RuleEngine
    from forestcheck.rules.models import EvaluationContext
    from test_recommendation_providers import relation_record, stock_rule

    rules = RuleEngine([stock_rule()])
    context = EvaluationContext(
        analysis_run_id=UUID(int=99), record_level="stand", allow_synthetic=True
    )
    outcomes = rules.evaluate(relation_record(), context)
    assert outcomes[0].finding.field_name is None
    assert set(affected_fields(outcomes, rules)) == {"per", "area", "total"}
    assert field_findings(outcomes, rules, "total") == [outcomes[0].finding]
