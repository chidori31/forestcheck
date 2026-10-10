"""Integration with invented imported datasets; no private fixtures."""

import json
from pathlib import Path

import pytest
from forestcheck.ingest.csv_import import normalize_dataset, sha256
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.rules.local import run_development


def prepare(tmp_path):
    source = tmp_path / "input" / "invented.csv"
    source.parent.mkdir()
    source.write_text("id;height\n001;4\n002;8\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    mapping = ImportMapping.model_validate(
        {
            "schema_version": "1.0",
            "mapping_version": "synthetic-1",
            "source_format": "csv",
            "encoding": "utf-8",
            "delimiter": ";",
            "table": "invented",
            "record_level": "source_row",
            "grain_basis": "synthetic",
            "columns": [
                {
                    "position": 1,
                    "source_header": "id",
                    "field": "id",
                    "unit": None,
                    "transform": "text",
                    "key_role": "stand",
                    "basis": "synthetic",
                },
                {
                    "position": 2,
                    "source_header": "height",
                    "field": "h",
                    "unit": "m",
                    "transform": "decimal_point",
                    "basis": "synthetic",
                },
            ],
        }
    )
    imported = normalize_dataset(source, mapping, private, workspace_root=repo)
    import pyarrow.parquet as pq

    ids = pq.read_table(imported / "normalized.parquet", columns=["record_id"])[
        "record_id"
    ].to_pylist()
    manifest = json.loads((imported / "manifest.json").read_text())
    split = {
        "schema_version": "1.0",
        "split_id": "synthetic-split",
        "assignments": {"invented": [{"record_id": ids[0], "group_digest": "f" * 64}]},
        "inputs": {
            "invented": {
                "dataset_id": manifest["dataset_id"],
                "source_sha256": manifest["source_sha256"],
                "mapping_sha256": manifest["mapping_sha256"],
                "parquet_sha256": manifest["output_hashes"]["normalized.parquet"],
            }
        },
    }
    (private / "stage4").mkdir()
    splitpath = private / "stage4" / "development-split.json"
    splitpath.write_text(json.dumps(split), encoding="utf-8")
    plan = {
        "schema_version": "1.0",
        "version": "synthetic-1",
        "split_sha256": sha256(splitpath),
        "rules_by_table": {
            "invented": [
                {
                    "schema_version": "1.0",
                    "id": "synthetic.height",
                    "version": "1",
                    "record_levels": ["source_row"],
                    "dependencies": ["h"],
                    "source": {
                        "reference_id": "synthetic",
                        "locator": "fixture section1",
                        "kind": "explicit_configuration",
                    },
                    "severity": "warning",
                    "priority": 2,
                    "explanation": "Synthetic metres contract",
                    "check": {
                        "kind": "field_contract",
                        "field": "h",
                        "value_type": "number",
                        "unit": "m",
                    },
                }
            ]
        },
        "composition_fields": {},
        "blocked_requirements": ["no_growth_reference"],
    }
    (private / "stage4" / "rule-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    config = repo / "local.json"
    config.write_text(
        json.dumps({"source_dir": str(source.parent), "private_dir": str(private), "imports": []}),
        encoding="utf-8",
    )
    return config, repo, private, ids, source


def test_run_uses_only_frozen_development_ids_and_repeat(tmp_path):
    config, repo, private, ids, source = prepare(tmp_path)
    before = sha256(source)
    result = run_development(config, workspace_root=repo)
    report = json.loads((result / "report.json").read_text())
    assert report["records_evaluated"] == 1
    outcomes = [json.loads(line) for line in (result / "outcomes.jsonl").read_text().splitlines()]
    assert {r["record_id"] for r in outcomes} == {ids[0]}
    assert report["rule_statuses"] == {"passed": 1}
    assert report["growth_statuses"] == {"unavailable_reference": 1}
    assert sha256(source) == before
    assert run_development(config, workspace_root=repo) == result


def test_frozen_split_change_refused(tmp_path):
    config, repo, private, _, _ = prepare(tmp_path)
    (private / "stage4" / "development-split.json").write_text("{}")
    with pytest.raises(ValueError, match="split_integrity"):
        run_development(config, workspace_root=repo)


def test_selected_id_missing_cannot_be_reported_complete(tmp_path):
    config, repo, private, _, _ = prepare(tmp_path)
    p = private / "stage4" / "development-split.json"
    split = json.loads(p.read_text())
    split["assignments"]["invented"][0]["record_id"] = "00000000-0000-0000-0000-000000000099"
    p.write_text(json.dumps(split))
    planpath = private / "stage4" / "rule-plan.json"
    plan = json.loads(planpath.read_text())
    plan["split_sha256"] = sha256(p)
    planpath.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="selected_records_missing"):
        run_development(config, workspace_root=repo)
    assert not list((private / "stage4" / "runs").glob("*/manifest.json"))


def test_cli_does_not_print_private_exceptions(tmp_path):
    import subprocess
    import sys

    p = tmp_path / "PRIVATE_SENTINEL.json"
    p.write_text("PRIVATE_CONTENT")
    out = subprocess.run(
        [sys.executable, "scripts/rules_local.py", "--config", str(p)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 3 and out.stderr == ""
    assert out.stdout.splitlines() == ["RULES_STARTED", "RULES_FAILED"]


@pytest.mark.parametrize("location", ["root", "input", "assignment"])
def test_unknown_frozen_split_fields_rejected(tmp_path, location):
    from pydantic import ValidationError

    config, repo, private, _, _ = prepare(tmp_path)
    path = private / "stage4" / "development-split.json"
    split = json.loads(path.read_text())
    target = (
        split
        if location == "root"
        else split["inputs"]["invented"]
        if location == "input"
        else split["assignments"]["invented"][0]
    )
    target["future_semantics"] = "test"
    path.write_text(json.dumps(split))
    plan_path = private / "stage4" / "rule-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["split_sha256"] = sha256(path)
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValidationError):
        run_development(config, workspace_root=repo)
