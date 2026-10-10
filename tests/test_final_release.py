from pathlib import Path

import pytest
from forestcheck.evaluation.final import claim_final, verify_release


def test_release_binds_candidate_and_cohort():
    frozen = {"candidate": "a", "split": "b", "cohort": ["one"], "schema_version": "1.0"}
    verify_release(frozen, dict(frozen))
    for key, value in [
        ("candidate", "changed"),
        ("split", "changed"),
        ("cohort", ["two"]),
        ("schema_version", "2.0"),
    ]:
        with pytest.raises(ValueError):
            verify_release(frozen, {**frozen, key: value})


def test_final_read_is_one_shot_even_after_failure(tmp_path: Path):
    claim_final(tmp_path, "digest")
    with pytest.raises(FileExistsError):
        claim_final(tmp_path, "digest")
    assert (tmp_path / "consumed.json").is_file()


def test_final_evaluator_synthetic_actual_models(tmp_path):
    import importlib.util
    import json

    from forestcheck.ingest.csv_import import sha256
    from forestcheck.ml.data import LocalData
    from test_recommendation_local import environment

    spec = importlib.util.spec_from_file_location("audit_local", Path("scripts/audit_local.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    config, repo, private, source, _ = environment(tmp_path)
    data = LocalData(config, workspace_root=repo)
    before = sha256(source)
    release, models, report, rules, precision, assignments = audit.inputs(data)
    stage = private / "stage10"
    stage.mkdir()
    audit.evaluate(data, stage, release, models, report, rules, precision, assignments)
    result = json.loads((stage / "evaluation.json").read_text("utf-8"))
    assert result["records"] > 0
    assert result["expert_metrics"] == "NOT_RUN_no_expert_labels"
    for target in result["targets"].values():
        assert target["eligible"] > 0
        assert set(target["methods"]) == {"rules", "rules_growth", "baseline", "full_pipeline"}
        assert len({m["sample_records"] for m in target["methods"].values()}) == 1
    assert sha256(source) == before
    with pytest.raises(FileExistsError):
        audit.evaluate(data, stage, release, models, report, rules, precision, assignments)
