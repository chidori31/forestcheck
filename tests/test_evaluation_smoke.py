import json

from forestcheck.evaluation.smoke import run_synthetic_smoke


def test_smoke_separates_payloads_and_does_not_claim_model_quality(tmp_path):
    destination = run_synthetic_smoke(tmp_path / "synthetic")
    report = json.loads((destination / "smoke-report.json").read_text())
    assert report["source_kind"] == "synthetic"
    assert report["corruption_cases"] == 6
    assert report["rare_valid_cases"] == 2
    assert report["real_metrics"] == "NOT_RUN"
    assert report["model_training"] == "NOT_RUN"
    assert report["peak_traced_python_bytes"] > 0
    for folder in (destination / "cases").iterdir():
        assert {p.name for p in folder.iterdir()} == {
            "original.json", "modified.json", "labels.json", "lineage.json"
        }
    forms = json.loads((destination / "annotation-forms.json").read_text())
    assert forms and all(f["error_assessment"] is None for f in forms)
