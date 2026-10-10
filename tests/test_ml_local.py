"""Synthetic end-to-end training, never competition fixtures."""

import json
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from forestcheck.evaluation.split import SplitIdentity, canonical, freeze_split, hash_value
from forestcheck.ingest.csv_import import normalize_dataset, sha256
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.ml.data import FeatureBinding, LocalData, MLPlan, TargetBinding, freeze_subsets
from forestcheck.ml.local import predict_run, train_run


def prepared(tmp_path):
    source = tmp_path / "source"
    private = tmp_path / "private"
    repo = tmp_path / "repo"
    for p in (source, private, repo):
        p.mkdir()
    csv = source / "invented.csv"
    csv.write_text(
        "region;quarter;stand;age;species;height;diameter\n"
        + "".join(
            f"01;{g:03d};{i:03d};{i + g % 5 + 1};A;{2 * (i + g % 5 + 1) + g % 3};{3 * (i + g % 5 + 1)}\n"
            for g in range(80)
            for i in range(6)
        ),
        encoding="utf-8",
    )
    mapping = ImportMapping(
        schema_version="1.0",
        mapping_version="synthetic-v1",
        source_format="csv",
        encoding="utf-8",
        delimiter=";",
        table="invented",
        record_level="source_row",
        grain_basis="synthetic",
        columns=[
            dict(
                position=i,
                source_header=name,
                field=name,
                unit=unit,
                transform="decimal_point" if unit else "text",
                key_role=key,
                basis="synthetic",
            )
            for i, (name, unit, key) in enumerate(
                [
                    ("region", None, "territory"),
                    ("quarter", None, "compartment"),
                    ("stand", None, "stand"),
                    ("age", "year", None),
                    ("species", None, None),
                    ("height", "m", None),
                    ("diameter", "cm", None),
                ],
                1,
            )
        ],
    )
    folder = normalize_dataset(csv, mapping, private, workspace_root=repo)
    import pyarrow.parquet as pq

    identities = []
    for text in pq.read_table(folder / "normalized.parquet", columns=["record_json"])[
        "record_json"
    ].to_pylist():
        r = json.loads(text)
        k = r["business_key"]
        identities.append(
            SplitIdentity(
                schema_version="1.0",
                record_id=r["id"],
                dataset_id=r["dataset_id"],
                group_id=hash_value([k["territory"], k["compartment"]]),
                parent_id=hash_value([k["territory"], k["compartment"], k["stand"]]),
            )
        )
    sourcehash, maphash, pqhash = (
        sha256(csv),
        sha256(folder / "mapping.json"),
        sha256(folder / "normalized.parquet"),
    )
    manifest = freeze_split(
        identities,
        seed="synthetic-ml",
        grouping_basis="Invented quarter",
        inputs={
            "invented:source": sourcehash,
            "invented:mapping": maphash,
            "invented:parquet": pqhash,
        },
    )
    (private / "stage5").mkdir()
    (private / "stage6").mkdir()
    splitfile = private / "stage5/split-manifest.json"
    splitfile.write_text(canonical(manifest.model_dump(mode="json")))
    (private / "stage5/receipt.json").write_text(
        json.dumps({"split_file_sha256": sha256(splitfile), "manifest_digest": manifest.digest})
    )
    folds, fit, check = freeze_subsets(manifest, seed=1701)
    plan = MLPlan(
        schema_version="1.0",
        plan_version="ml-plan-1.0.0",
        seed=1701,
        table="invented",
        dataset_id=UUID(folder.name),
        mapping_version=mapping.mapping_version,
        mapping_sha256=maphash,
        split_sha256=manifest.digest,
        snapshot_sha256=pqhash,
        source_sha256=sourcehash,
        source_kind="synthetic",
        features=tuple(
            FeatureBinding(
                schema_version="1.0",
                name=name,
                source_field=name,
                unit=unit,
                kind=kind,
                provenance="Invented independent inputs",
                dependency_status="independent_measurement_confirmed",
            )
            for name, unit, kind in [
                ("age", "year", "numeric"),
                ("species", None, "categorical"),
                ("region", None, "context"),
            ]
        ),
        targets=tuple(
            TargetBinding(
                schema_version="1.0",
                name=name,
                source_field=name,
                unit=unit,
                provenance="Invented truth",
            )
            for name, unit in [("height", "m"), ("diameter", "cm")]
        ),
        excluded_fields=("quarter", "stand"),
        exclusion_rationale="No identifier features",
        train_folds=folds,
        calibration_fit_groups=fit,
        calibration_check_groups=check,
        limitations=("Synthetic fixture only",),
        iterations=5,
    )
    (private / "stage6/plan.json").write_text(plan.model_dump_json())
    config = repo / "local.json"
    config.write_text(json.dumps({"source_dir": str(source), "private_dir": str(private)}))
    return config, repo, private, csv


def test_train_evaluate_and_reload_repeat_do_not_touch_holdout(tmp_path):
    config, repo, private, source = prepared(tmp_path)
    run_id = uuid4()
    before = sha256(source)
    report = train_run(config, workspace_root=repo, run_id=run_id)
    assert report["status"] == "trained_candidate"
    assert report["holdout_status"] == "NOT_RUN"
    assert report["promoted"] is False
    assert set(report["targets"]) == {"height", "diameter"}
    assert report["reload_max_abs_difference"] <= 1e-9
    assert report["oof"]["all_train_records_once"] is True
    assert sha256(source) == before
    output = predict_run(config, workspace_root=repo, run_id=run_id, role="tune")
    assert output.is_file()
    assert predict_run(config, workspace_root=repo, run_id=run_id, role="tune") == output
    for role in ("holdout", "train"):
        with pytest.raises(ValueError):
            predict_run(config, workspace_root=repo, run_id=run_id, role=role)


def test_loader_rejects_mapping_tamper_before_features(tmp_path):
    config, repo, private, _ = prepared(tmp_path)
    folder = next((private / "imports").iterdir())
    p = folder / "mapping.json"
    data = json.loads(p.read_text())
    data["mapping_version"] = "changed"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mapping_version_or_table"):
        LocalData(config, workspace_root=repo)


def test_utf8_bom_local_config_supported(tmp_path):
    config, repo, _, _ = prepared(tmp_path)
    config.write_text(config.read_text(), encoding="utf-8-sig")
    assert LocalData(config, workspace_root=repo).plan.table == "invented"


def test_linked_stage_directory_refused_before_metadata(tmp_path, monkeypatch):
    config, repo, private, _ = prepared(tmp_path)
    original = Path.resolve

    def resolved(path, *args, **kwargs):
        if path == private / "stage5":
            return private / "redirected"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolved)
    with pytest.raises(ValueError, match="directory"):
        LocalData(config, workspace_root=repo)


def test_loader_never_decodes_holdout_or_unrequested_record_json(tmp_path, monkeypatch):
    config, repo, _, _ = prepared(tmp_path)
    data = LocalData(config, workspace_root=repo)
    allowed = {a.record_id for a in data.manifest.assignments if a.role == "train"}
    import forestcheck.ml.data as module

    original = module.Record.model_validate_json
    seen = []

    def validate(text):
        record = original(text)
        assert record.id in allowed
        seen.append(record.id)
        return record

    monkeypatch.setattr(module.Record, "model_validate_json", validate)
    assert len(data.load({"train"})) == len(seen) == len(allowed)
    with pytest.raises(ValueError, match="holdout_forbidden"):
        data.load({"holdout"})


def test_cli_failure_emits_only_status(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/ml_local.py"
    run = subprocess.run(
        [sys.executable, str(script), "train", "--config", str(tmp_path / "PRIVATE")],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 3 and run.stdout.splitlines() == ["ML_FAILED"] and run.stderr == ""


def test_run_binding_queue_and_manual_lifecycle(tmp_path, monkeypatch, capsys):
    import importlib.util

    from forestcheck.ml.artifacts import load_candidate, register_candidate
    from forestcheck.ml.local import compatibility

    config, repo, private, _ = prepared(tmp_path)
    run_id = uuid4()
    report = train_run(config, workspace_root=repo, run_id=run_id)
    destination = private / "stage6/runs" / str(run_id)
    predict_run(config, workspace_root=repo, run_id=run_id)
    forms = json.loads((destination / "tune-annotation-forms.json").read_text())
    sampling = json.loads((destination / "tune-annotation-sampling.json").read_text())
    assert {s["stratum"] for s in sampling["sampling"]} == {"flagged", "control"}
    assert all(f["error_assessment"] is None and f["correction_assessment"] is None for f in forms)
    assert report["isolation"]["reload_max_abs_difference"] <= 1e-10
    plan = LocalData(config, workspace_root=repo).plan
    target = plan.targets[0]
    registry = private / "stage6/models" / target.name
    first_id = UUID(report["model_ids"][target.name])
    loaded = load_candidate(registry, first_id, expected=compatibility(plan, target))
    mismatched = loaded.metadata.model_copy(update={"split_sha256": "f" * 64})
    other = register_candidate(registry, mismatched, loaded.model_files)
    changed = json.loads(json.dumps(report))
    changed["model_ids"][target.name] = str(other)
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="artifact_plan_binding"):
        predict_run(config, workspace_root=repo, run_id=run_id)
    for key, value in (("schema_version", "2.0"), ("run_id", str(uuid4())), ("extra", True)):
        report_path.write_text(json.dumps({**report, key: value}))
        with pytest.raises(ValueError):
            predict_run(config, workspace_root=repo, run_id=run_id)
    report_path.write_text(json.dumps(report))
    second_id = register_candidate(registry, loaded.metadata, loaded.model_files)
    script = Path(__file__).resolve().parents[1] / "scripts/ml_local.py"
    module_spec = importlib.util.spec_from_file_location("tested_ml_cli", script)
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    # Use this synthetic workspace boundary, not the real project containing pytest scratch.
    monkeypatch.setattr(cli, "__file__", str(repo / "scripts/ml_local.py"))
    for command, selected, current in (
        ("promote", first_id, None),
        ("promote", second_id, first_id),
        ("rollback", first_id, second_id),
    ):
        arguments = [
            sys.executable,
            str(script),
            command,
            "--config",
            str(config),
            "--target",
            target.name,
            "--model-id",
            str(selected),
            "--reason",
            "Synthetic lifecycle test",
            "--confirm-manual-promotion",
        ]
        if current:
            arguments.extend(["--expected-current", str(current)])
        monkeypatch.setattr(sys, "argv", arguments[1:])
        assert cli.main() == 0
        output = capsys.readouterr()
        assert output.err == "" and output.out.splitlines() == ["ML_STARTED", "ML_COMPLETE"]


def test_isolation_context_uses_only_actual_fit_rows():
    from types import SimpleNamespace

    from forestcheck.ml.isolation import IsolationPrediction
    from forestcheck.ml.local import build_isolation_context, guard_isolation_context

    plan = SimpleNamespace(features=(SimpleNamespace(name="region", kind="context"),))
    samples = [
        SimpleNamespace(features={"region": "known"}),
        SimpleNamespace(features={"region": "excluded"}),
    ]
    context = build_isolation_context(samples, plan, (0,))
    scores = [IsolationPrediction(status="scored", score=0.4, rank_quantile=0.5, warnings=())] * 2
    guarded = guard_isolation_context(scores, samples, context)
    assert guarded[0]["status"] == "scored"
    assert guarded[1]["status"] == "abstained"


def test_cli_preserves_existing_failure_without_leaking_traceback(tmp_path, monkeypatch, capsys):
    import importlib.util

    import forestcheck.ml.local as local

    config, repo, private, _ = prepared(tmp_path)
    run_id = uuid4()
    destination = private / "stage6/runs" / str(run_id)
    destination.mkdir(parents=True)
    diagnostic = destination / "failure.json"
    diagnostic.write_text('{"original":true}')
    script = Path(__file__).resolve().parents[1] / "scripts/ml_local.py"
    module_spec = importlib.util.spec_from_file_location("failed_ml_cli", script)
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "__file__", str(repo / "scripts/ml_local.py"))

    def failed(*args, **kwargs):
        raise ValueError("private diagnostics must remain local")

    monkeypatch.setattr(local, "train_run", failed)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "train", "--config", str(config), "--run-id", str(run_id), "--worker"],
    )
    assert cli.main() == 3
    output = capsys.readouterr()
    assert output.out.splitlines() == ["ML_STARTED", "ML_FAILED"] and output.err == ""
    assert diagnostic.read_text() == '{"original":true}'
