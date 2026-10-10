import json
import subprocess
import sys
from pathlib import Path

import pytest
from forestcheck.evaluation.local import prepare_evaluation
from forestcheck.ingest.csv_import import normalize_dataset, sha256
from forestcheck.ingest.mapping import ImportMapping


def prepare(tmp_path, groups=80):
    source = tmp_path / "input" / "synthetic.csv"
    source.parent.mkdir()
    source.write_text("region;quarter;stand;height\n" + "".join(
        f"01;{g:03d};001;{g % 10}\n" for g in range(groups)
    ), encoding="utf-8")
    private, repo = tmp_path / "private", tmp_path / "repo"
    private.mkdir()
    repo.mkdir()
    mapping = ImportMapping.model_validate({
        "schema_version": "1.0", "mapping_version": "synthetic-1", "source_format": "csv",
        "encoding": "utf-8", "delimiter": ";", "table": "synthetic",
        "record_level": "source_row", "grain_basis": "Synthetic only",
        "columns": [
            {"position": i, "source_header": name, "field": name,
             "unit": "m" if name == "height" else None,
             "transform": "decimal_point" if name == "height" else "text",
             "key_role": role, "basis": "Synthetic only"}
            for i, (name, role) in enumerate([
                ("region", "territory"), ("quarter", "compartment"),
                ("stand", "stand"), ("height", None),
            ], 1)
        ],
    })
    imported = normalize_dataset(source, mapping, private, workspace_root=repo)
    import pyarrow.parquet as pq
    ids = pq.read_table(imported / "normalized.parquet", columns=["record_id"])["record_id"].to_pylist()
    manifest = json.loads((imported / "manifest.json").read_text())
    stage4 = private / "stage4"
    stage4.mkdir()
    split = {
        "schema_version": "1.0", "split_id": "synthetic-stage4",
        "inputs": {"synthetic": {
            "dataset_id": manifest["dataset_id"], "source_sha256": sha256(source),
            "mapping_sha256": sha256(imported / "mapping.json"),
            "parquet_sha256": sha256(imported / "normalized.parquet"),
        }},
        "assignments": {"synthetic": [{"record_id": ids[0], "group_digest": "a" * 64}]},
    }
    splitpath = stage4 / "development-split.json"
    splitpath.write_text(json.dumps(split), encoding="utf-8")
    (stage4 / "rule-plan.json").write_text(json.dumps({"schema_version": "1.0",
        "version": "synthetic", "split_sha256": sha256(splitpath), "rules_by_table": {"synthetic": []},
        "composition_fields": {}, "blocked_requirements": []}), encoding="utf-8")
    config = repo / "local.json"
    config.write_text(json.dumps({"source_dir": str(source.parent), "private_dir": str(private), "imports": []}))
    return config, repo, private, source


def test_freeze_before_structure_smoke_and_idempotent(tmp_path):
    config, repo, private, source = prepare(tmp_path)
    before = sha256(source)
    destination = prepare_evaluation(config, workspace_root=repo)
    assert destination == private / "stage5"
    receipt = json.loads((destination / "receipt.json").read_text())
    report = json.loads((destination / "structure-report.json").read_text())
    assert receipt["split_file_sha256"] == sha256(destination / "split-manifest.json")
    assert report["rows"] == 80
    assert report["groups"] == 80
    assert report["prior_exposed_groups"] == 1
    assert report["feature_validation_roles"] == ["train", "tune"]
    assert report["metrics_status"] == "NOT_RUN"
    assert report["structure_validated_records"] > 0
    assert report["final_holdout_evaluated"] is False
    assert sha256(source) == before
    assert prepare_evaluation(config, workspace_root=repo) == destination


def test_existing_manifest_tamper_source_change_or_stale_prior_refused(tmp_path):
    config, repo, private, source = prepare(tmp_path)
    destination = prepare_evaluation(config, workspace_root=repo)
    (destination / "split-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="existing_artifact_integrity"):
        prepare_evaluation(config, workspace_root=repo)
    source.write_text(source.read_text() + "01;081;001;3\n")
    with pytest.raises(ValueError, match="source_integrity"):
        prepare_evaluation(config, workspace_root=repo)


def test_missing_group_blocks_freeze(tmp_path):
    config, repo, private, _ = prepare(tmp_path)
    splitpath = private / "stage4" / "development-split.json"
    data = json.loads(splitpath.read_text())
    data["assignments"]["synthetic"][0]["record_id"] = "00000000-0000-0000-0000-000000000999"
    splitpath.write_text(json.dumps(data))
    planpath = private / "stage4" / "rule-plan.json"
    plan = json.loads(planpath.read_text())
    plan["split_sha256"] = sha256(splitpath)
    planpath.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="prior_records_missing"):
        prepare_evaluation(config, workspace_root=repo)
    assert not (private / "stage5" / "split-manifest.json").exists()


def test_cli_status_only_on_invalid_config(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/evaluation_local.py"
    run = subprocess.run([sys.executable, str(script), "--config", str(tmp_path / "PRIVATE")],
                         capture_output=True, text=True)
    assert run.returncode == 3
    assert run.stdout.splitlines() == ["EVALUATION_STARTED", "EVALUATION_FAILED"]
    assert run.stderr == ""


def test_small_cv_structure_smoke_is_explicit(tmp_path):
    config, repo, _, _ = prepare(tmp_path, groups=9)
    destination = prepare_evaluation(config, workspace_root=repo)
    report = json.loads((destination / "structure-report.json").read_text())
    assert report["strategy"] == "group_cv"
    assert report["structure_validated_records"] == 9
    assert report["feature_validation_roles"] == ["development_cv"]
