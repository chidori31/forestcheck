"""Invented CSV fixtures only. No competition paths or schemas."""
import hashlib
import json

import pytest
from forestcheck.ingest.csv_import import inspect_source, normalize_dataset, validate_mapping
from forestcheck.ingest.mapping import ImportMapping
from pydantic import ValidationError


def mapping_payload():
    return {
        "schema_version": "1.0", "mapping_version": "synthetic-1",
        "source_format": "csv", "encoding": "cp1251", "delimiter": ";",
        "table": "synthetic", "record_level": "source_row", "grain_basis": "Synthetic fixture",
        "analysis_blockers": ["units_unconfirmed"],
        "columns": [
            {"position": 1, "source_header": "id", "field": "stand_id", "unit": None,
             "transform": "text", "key_role": "stand", "basis": "Synthetic identifier"},
            {"position": 2, "source_header": "value", "field": "measure", "unit": "m",
             "transform": "decimal_comma", "basis": "Synthetic declared metres",
             "not_applicable_tokens": ["N/A"]},
            {"position": 3, "source_header": "value", "field": "unresolved", "unit": None,
             "transform": "unknown", "basis": "Synthetic ambiguous header"},
        ],
        "service_rows": [{"position": 1, "values": ["TOTAL"], "reason": "synthetic_total"}],
    }


def test_import_balances_states_raw_positions_and_repeat(tmp_path):
    source = tmp_path / "input" / "source.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n001;1,25;raw\n002;0;\n003;;?\n004;N/A;other\n005;bad;kept\nTOTAL;9;\n006;2;extra;ragged\n\n", encoding="cp1251")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    repo = tmp_path / "repo"
    repo.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    mapping = ImportMapping.model_validate(mapping_payload())
    inspection = inspect_source(source, encoding=mapping.encoding, delimiter=mapping.delimiter)
    assert inspection["headers"] == ["id", "value", "value"]
    validate_mapping(inspection, mapping)
    result = normalize_dataset(source, mapping, private, workspace_root=repo)
    report = json.loads((result / "report.json").read_text())
    assert report["balance"] == {"input_records": 8, "accepted": 4, "quarantined": 2, "excluded": 2, "normalized_records": 4}
    import pyarrow.parquet as pq
    records = pq.read_table(result / "normalized.parquet").to_pylist()
    assert records[0]["fields"][0]["original_value"] == "001"
    assert records[0]["fields"][1]["normalized_number"] == 1.25
    assert records[1]["fields"][1]["normalized_number"] == 0
    assert [r["fields"][1]["state"] for r in records] == ["observed", "observed", "missing", "not_applicable"]
    assert records[0]["fields"][2]["state"] == "unknown"
    assert records[0]["fields"][1]["column_position"] == 2
    assert records[0]["fields"][2]["column_position"] == 3
    assert all(r["analysis_blocked"] for r in records)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    manifest = (result / "manifest.json").read_bytes()
    assert normalize_dataset(source, mapping, private, workspace_root=repo) == result
    assert (result / "manifest.json").read_bytes() == manifest
    assert len(list((private / "imports").iterdir())) == 1


def test_mapping_unknown_versions_fields_and_header_changes_rejected(tmp_path):
    payload = mapping_payload()
    for version in ("0.9", "1.1", "2.0"):
        with pytest.raises(ValidationError):
            ImportMapping.model_validate(dict(payload, schema_version=version))
    with pytest.raises(ValidationError):
        ImportMapping.model_validate(dict(payload, guessed_unit=True))
    mapping = ImportMapping.model_validate(payload)
    with pytest.raises(ValueError, match="header_mismatch"):
        validate_mapping({"headers": ["id", "value", "changed"]}, mapping)


def test_utf8_multiline_and_unconfigured_total_never_silently_dropped(tmp_path):
    payload = mapping_payload()
    payload.update(encoding="utf-8-sig", service_rows=[])
    source = tmp_path / "input" / "source.csv"
    source.parent.mkdir()
    source.write_text('id;value;value\n"00\n1";2;raw\nTOTAL;3;raw\n', encoding="utf-8-sig")
    private = tmp_path / "private"
    private.mkdir()
    result = normalize_dataset(source, ImportMapping.model_validate(payload), private, workspace_root=tmp_path / "repo")
    import pyarrow.parquet as pq
    records = pq.read_table(result / "normalized.parquet").to_pylist()
    assert [r["source_line_start"] for r in records] == [2, 4]
    assert [r["source_line_end"] for r in records] == [3, 4]
    assert records[1]["fields"][0]["original_value"] == "TOTAL"


def test_corrupt_existing_output_refused_and_private_boundary(tmp_path):
    source = tmp_path / "input" / "a.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n001;2;raw\n", encoding="cp1251")
    mapping = ImportMapping.model_validate(mapping_payload())
    with pytest.raises(ValueError):
        normalize_dataset(source, mapping, tmp_path, workspace_root=tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    result = normalize_dataset(source, mapping, private, workspace_root=tmp_path / "repo")
    (result / "report.json").write_text("corrupt")
    with pytest.raises(ValueError, match="existing_output_integrity"):
        normalize_dataset(source, mapping, private, workspace_root=tmp_path / "repo")


def test_same_business_key_does_not_merge_rows(tmp_path):
    source = tmp_path / "input" / "a.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n001;2;one\n001;2;two\n", encoding="cp1251")
    private = tmp_path / "private"
    private.mkdir()
    result = normalize_dataset(source, ImportMapping.model_validate(mapping_payload()), private, workspace_root=tmp_path / "repo")
    import pyarrow.parquet as pq
    records = pq.read_table(result / "normalized.parquet").to_pylist()
    assert len({r["record_id"] for r in records}) == 2
    assert [r["fields"][0]["original_value"] for r in records] == ["001", "001"]


@pytest.mark.parametrize("raw", ["1.2", "NaN", "inf", "1e999", "1 000", "=1+2"])
def test_ambiguous_numbers_are_parse_errors(raw):
    from forestcheck.ingest.csv_import import convert
    column = ImportMapping.model_validate(mapping_payload()).columns[1]
    normalized, state, _ = convert(raw, column)
    assert normalized is None and state == "parse_error"


def test_text_formula_is_literal_and_not_executed():
    from forestcheck.ingest.csv_import import convert
    column = ImportMapping.model_validate(mapping_payload()).columns[0]
    assert convert('=1+2', column)[:2] == ('=1+2', 'observed')


def test_mapping_rejects_coercion_of_identifiers_and_ambiguous_tokens():
    payload = mapping_payload()
    payload["columns"][0]["transform"] = "integer"
    with pytest.raises(ValidationError):
        ImportMapping.model_validate(payload)
    payload = mapping_payload()
    payload["columns"][1]["not_applicable_tokens"] = [""]
    with pytest.raises(ValidationError):
        ImportMapping.model_validate(payload)


def test_import_cli_failure_does_not_leak_private_paths_or_content(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    sentinel = "PRIVATE_SENTINEL_DO_NOT_PRINT"
    bad_config = tmp_path / f"{sentinel}.json"
    bad_config.write_text(sentinel)
    result = subprocess.run([sys.executable, "scripts/import_local.py", "--config", str(bad_config)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 3
    assert result.stdout.splitlines() == ["IMPORT_STARTED", "IMPORT_FAILED"]
    assert result.stderr == ""


def test_header_only_csv_balances_zero_records(tmp_path):
    source = tmp_path / "input" / "empty.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n", encoding="cp1251")
    private = tmp_path / "private"
    private.mkdir()
    result = normalize_dataset(source, ImportMapping.model_validate(mapping_payload()), private,
                               workspace_root=tmp_path / "repo")
    report = json.loads((result / "report.json").read_text())
    assert all(n == 0 for n in report["balance"].values())


def test_fatal_csv_error_leaves_no_completed_snapshot(tmp_path):
    source = tmp_path / "input" / "bad.csv"
    source.parent.mkdir()
    source.write_text('id;value;value\n001;2;"unclosed', encoding="cp1251")
    private = tmp_path / "private"
    private.mkdir()
    import csv
    with pytest.raises(csv.Error):
        normalize_dataset(source, ImportMapping.model_validate(mapping_payload()), private,
                          workspace_root=tmp_path / "repo")
    assert not list((private / "imports").glob("*/manifest.json"))


def test_source_change_during_import_not_published(tmp_path, monkeypatch):
    import forestcheck.ingest.csv_import as importer
    source = tmp_path / "input" / "changing.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n001;2;raw\n", encoding="cp1251")
    private = tmp_path / "private"
    private.mkdir()
    original_sha = importer.sha256
    calls = 0
    def changed_hash(path):
        nonlocal calls
        if path == source:
            calls += 1
            if calls > 1:
                return "f" * 64
        return original_sha(path)
    monkeypatch.setattr(importer, "sha256", changed_hash)
    with pytest.raises(ValueError, match="source_changed_during_import"):
        normalize_dataset(source, ImportMapping.model_validate(mapping_payload()), private,
                          workspace_root=tmp_path / "repo")
    assert not list((private / "imports").glob("*/manifest.json"))


def test_extreme_integer_becomes_parse_error():
    from forestcheck.ingest.csv_import import convert
    payload = mapping_payload()
    payload["columns"][1]["transform"] = "integer"
    column = ImportMapping.model_validate(payload).columns[1]
    assert convert("9" * 4301, column)[:2] == (None, "parse_error")


def test_parent_segments_cannot_bypass_source_output_boundary(tmp_path):
    source = tmp_path / "input" / "source.csv"
    source.parent.mkdir()
    (source.parent / "child").mkdir()
    private = source.parent / "private"
    private.mkdir()
    source.write_text("id;value;value\n001;2;raw\n", encoding="cp1251")
    with pytest.raises(ValueError, match="source_and_output_must_not_overlap"):
        normalize_dataset(source.parent / "child" / ".." / source.name,
                          ImportMapping.model_validate(mapping_payload()), private,
                          workspace_root=tmp_path / "repo")
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("key,value", [("schema_version", "2.0"), ("pipeline_version", "future"),
                                     ("dataset_id", "wrong"), ("source_sha256_after", "wrong"),
                                     ("unexpected", True)])
def test_manifest_version_and_identity_tampering_refused(tmp_path, key, value):
    source = tmp_path / "input" / "source.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n001;2;raw\n", encoding="cp1251")
    private = tmp_path / "private"
    private.mkdir()
    mapping = ImportMapping.model_validate(mapping_payload())
    result = normalize_dataset(source, mapping, private, workspace_root=tmp_path / "repo")
    path = result / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="existing_output_integrity"):
        normalize_dataset(source, mapping, private, workspace_root=tmp_path / "repo")


def test_missing_unknown_column_still_blocks_analysis(tmp_path):
    source = tmp_path / "input" / "source.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n001;2;\n", encoding="cp1251")
    private = tmp_path / "private"
    private.mkdir()
    mapping = ImportMapping.model_validate(dict(mapping_payload(), analysis_blockers=[]))
    result = normalize_dataset(source, mapping, private, workspace_root=tmp_path / "repo")
    import pyarrow.parquet as pq
    row = pq.read_table(result / "normalized.parquet").to_pylist()[0]
    assert row["fields"][2]["state"] == "missing"
    assert row["analysis_blocked"]
    assert "unknown:unresolved" in row["blockers"]


def test_explicit_decimal_point_transform_no_separator_guessing():
    from forestcheck.ingest.csv_import import convert
    payload = mapping_payload()
    payload["columns"][1]["transform"] = "decimal_point"
    column = ImportMapping.model_validate(payload).columns[1]
    assert convert("1.25", column)[:2] == (1.25, "observed")
    assert convert("0.0", column)[:2] == (0.0, "observed")
    assert convert("1,25", column)[:2] == (None, "parse_error")
    assert convert(" 12.0 ", column)[2] == ["strip_outer_whitespace", "decimal_point_to_float"]


@pytest.mark.parametrize("transform", ["decimal_point", "decimal_comma"])
def test_nonzero_float_underflow_is_not_observed_zero(transform):
    from forestcheck.ingest.csv_import import convert
    payload = mapping_payload()
    payload["columns"][1]["transform"] = transform
    column = ImportMapping.model_validate(payload).columns[1]
    for raw in ("1e-999", "-1e-999"):
        assert convert(raw, column)[:2] == (None, "parse_error")
    for raw in ("0e-999", "-0"):
        assert convert(raw, column)[:2] == (0.0, "observed")


def test_exclusion_flag_is_strict_boolean():
    for value in (1, "yes", "false"):
        with pytest.raises(ValidationError):
            ImportMapping.model_validate(dict(mapping_payload(), exclude_empty_records=value))


def test_literal_empty_identifier_preserved_with_analysis_blocker(tmp_path):
    payload = mapping_payload()
    payload["columns"][0]["missing_tokens"] = []
    source = tmp_path / "input" / "source.csv"
    source.parent.mkdir()
    source.write_text("id;value;value\n;2;raw\n", encoding="cp1251")
    private = tmp_path / "private"
    private.mkdir()
    result = normalize_dataset(source, ImportMapping.model_validate(payload), private,
                               workspace_root=tmp_path / "repo")
    import pyarrow.parquet as pq
    row = pq.read_table(result / "normalized.parquet").to_pylist()[0]
    assert row["fields"][0]["original_value"] == ""
    assert row["fields"][0]["state"] == "observed"
    assert "business_key_unavailable" in row["blockers"]
