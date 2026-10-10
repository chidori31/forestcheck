"""Position-aware, deterministic CSV ingestion to a private immutable snapshot."""
import csv
import hashlib
import json
import math
import re
import stat
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from forestcheck.domain.contracts import BusinessKey, FieldValue, Record, SourceCoordinates
from forestcheck.ingest.mapping import ImportMapping

PIPELINE_VERSION = "csv-import-1.0.2"
INTEGER = re.compile(r"[+-]?[0-9]+\Z")
COMMA_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:,[0-9]*)?|,[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
POINT_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def checked_source(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
        raise ValueError("absolute_local_source_required")
    for item in (path, *path.parents):
        st = item.lstat()
        if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & 0x400:
            raise ValueError("source_link_forbidden")
    st = path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > 512 * 1024**2:
        raise ValueError("source_type_or_size_limit")
    if path.suffix.lower() != ".csv":
        raise ValueError("unsupported_source_format")
    return path.resolve(strict=True)


def inspect_source(path: Path, *, encoding: str, delimiter: str) -> dict:
    path = checked_source(path)
    csv.field_size_limit(1024**2)
    with path.open("r", encoding=encoding, newline="") as stream:
        headers = next(csv.reader(stream, delimiter=delimiter, strict=True), [])
    if not headers or len(headers) > 2048:
        raise ValueError("missing_or_excessive_columns")
    return {"format": "csv", "sha256": sha256(path), "size_bytes": path.stat().st_size,
            "headers": headers, "encoding": encoding, "delimiter": delimiter}


def validate_mapping(inspection: dict, mapping: ImportMapping) -> None:
    if inspection["headers"] != [c.source_header for c in mapping.columns]:
        raise ValueError("header_mismatch")


def representable_float(number: float, text: str) -> bool:
    if not math.isfinite(number):
        return False
    coefficient = text.lower().split("e", 1)[0]
    return number != 0 or not any(c in "123456789" for c in coefficient)


def convert(raw: str, column) -> tuple[object, str, list[str]]:
    # State markers are exact matches; whitespace handling is an explicit transform.
    if raw in column.missing_tokens:
        return None, "missing", ["exact_missing_token"]
    if raw in column.not_applicable_tokens:
        return None, "not_applicable", ["exact_not_applicable_token"]
    if column.transform == "unknown":
        return None, "unknown", ["preserve_uninterpreted"]
    if column.transform == "text":
        return raw, "observed", ["identity_text"]
    value = raw.strip()
    if column.transform == "integer" and INTEGER.fullmatch(value):
        try:
            number = int(value)
        except ValueError:
            return None, "parse_error", ["integer_conversion_limit"]
        if -(2**63) <= number < 2**63:
            return number, "observed", ["strip_numeric_whitespace", "parse_int64"]
    if column.transform == "decimal_comma" and COMMA_NUMBER.fullmatch(value):
        number = float(value.replace(",", "."))
        if representable_float(number, value):
            return number, "observed", ["strip_numeric_whitespace", "decimal_comma_to_float64"]
    if column.transform == "decimal_point" and POINT_NUMBER.fullmatch(value):
        number = float(value)
        if representable_float(number, value):
            return number, "observed", ["strip_outer_whitespace", "decimal_point_to_float"]
    return None, "parse_error", ["parse_failed_no_coercion"]


def parquet_schema():
    import pyarrow as pa
    field = pa.struct([
        ("name", pa.string()), ("original_value", pa.string()),
        ("normalized_string", pa.string()), ("normalized_integer", pa.int64()),
        ("normalized_number", pa.float64()), ("state", pa.string()), ("unit", pa.string()),
        ("column_position", pa.int32()), ("transformation_history", pa.list_(pa.string())),
    ])
    return pa.schema([
        ("record_id", pa.string()), ("dataset_id", pa.string()), ("record_level", pa.string()),
        ("source_record_ordinal", pa.int64()), ("source_line_start", pa.int64()),
        ("source_line_end", pa.int64()), ("fields", pa.list_(field)),
        ("record_json", pa.string()), ("analysis_blocked", pa.bool_()),
        ("blockers", pa.list_(pa.string())),
    ])


def checked_private(private: Path, source: Path, workspace_root: Path) -> Path:
    private = Path(private)
    if not private.is_absolute() or str(private).startswith(("\\\\", "//")):
        raise ValueError("absolute_local_private_directory_required")
    for item in (private, *private.parents):
        st = item.lstat()
        if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & 0x400:
            raise ValueError("private_link_forbidden")
    private = private.resolve(strict=True)
    if not private.is_dir() or private.is_relative_to(workspace_root.resolve()):
        raise ValueError("private_directory_must_be_outside_workspace")
    if private.is_relative_to(source.parent) or source.is_relative_to(private):
        raise ValueError("source_and_output_must_not_overlap")
    return private


def regular_output(path: Path) -> bool:
    st = path.lstat()
    return (stat.S_ISREG(st.st_mode) and st.st_nlink == 1
            and not getattr(st, "st_file_attributes", 0) & 0x400)


def normalize_dataset(source: Path, mapping: ImportMapping, private: Path, *, workspace_root: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = checked_source(source)
    private = checked_private(private, source, workspace_root)
    inspected = inspect_source(source, encoding=mapping.encoding, delimiter=mapping.delimiter)
    validate_mapping(inspected, mapping)
    mapping_json = canonical(mapping.model_dump(mode="json"))
    mapping_hash = hashlib.sha256(mapping_json.encode("utf-8")).hexdigest()
    identity = f"forestcheck:{PIPELINE_VERSION}:{inspected['sha256']}:{mapping_hash}"
    dataset_id = uuid5(NAMESPACE_URL, identity)
    artifact_id = uuid5(NAMESPACE_URL, "sha256:" + inspected["sha256"])
    parent = private / "imports"
    parent.mkdir(exist_ok=True)
    if parent.is_symlink() or getattr(parent.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("output_link_forbidden")
    destination = parent / str(dataset_id)
    if destination.exists():
        if destination.is_symlink() or getattr(destination.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("output_link_forbidden")
        manifest_path = destination / "manifest.json"
        if not regular_output(manifest_path) or manifest_path.stat().st_size > 1024**2:
            raise ValueError("existing_output_integrity")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        expected = {"normalized.parquet", "quarantine.jsonl", "excluded.jsonl", "report.json", "mapping.json"}
        manifest_fields = {"schema_version", "pipeline_version", "dataset_id", "source_name",
                           "source_sha256", "source_sha256_after", "mapping_sha256", "encoding", "output_hashes"}
        if (not isinstance(manifest, dict) or set(manifest) != manifest_fields
                or manifest.get("schema_version") != "1.0"
                or manifest.get("pipeline_version") != PIPELINE_VERSION
                or manifest.get("dataset_id") != str(dataset_id)
                or manifest.get("source_sha256") != inspected["sha256"]
                or manifest.get("source_sha256_after") != inspected["sha256"]
                or manifest.get("encoding") != mapping.encoding
                or not isinstance(manifest.get("source_name"), str)
                or manifest.get("mapping_sha256") != mapping_hash
                or not isinstance(manifest.get("output_hashes"), dict)
                or set(manifest.get("output_hashes", {})) != expected):
            raise ValueError("existing_output_integrity")
        for name in expected:
            item = destination / name
            if not regular_output(item) or sha256(item) != manifest["output_hashes"][name]:
                raise ValueError("existing_output_integrity")
        return destination
    staging = parent / (".incomplete-" + str(dataset_id) + "-" + uuid4().hex)
    staging.mkdir(mode=0o700)
    (staging / "mapping.json").write_text(mapping_json, encoding="utf-8")
    balance = dict(input_records=0, accepted=0, quarantined=0, excluded=0, normalized_records=0)
    state_counts, reasons = Counter(), Counter()
    blocked = 0
    schema, batch = parquet_schema(), []
    with (source.open("r", encoding=mapping.encoding, newline="") as stream,
          (staging / "quarantine.jsonl").open("w", encoding="utf-8") as quarantine,
          (staging / "excluded.jsonl").open("w", encoding="utf-8") as excluded,
          pq.ParquetWriter(staging / "normalized.parquet", schema, compression="zstd") as writer):
        reader = csv.reader(stream, delimiter=mapping.delimiter, strict=True)
        next(reader)
        end_line = reader.line_num
        for ordinal, raw in enumerate(reader, 1):
            start_line, end_line = end_line + 1, reader.line_num
            if ordinal > 2000000:
                raise ValueError("record_limit")
            balance["input_records"] += 1
            provenance = {"source_record_ordinal": ordinal, "source_line_start": start_line,
                          "source_line_end": end_line, "raw_values": raw,
                          "record_id": str(uuid5(dataset_id, f"record:{ordinal}")),
                          "artifact_id": str(artifact_id), "table": mapping.table}
            exclusion = "blank_record" if not raw and mapping.exclude_empty_records else None
            for policy in mapping.service_rows:
                if len(raw) >= policy.position and raw[policy.position - 1] in policy.values:
                    exclusion = policy.reason
            if exclusion:
                balance["excluded"] += 1
                reasons[exclusion] += 1
                excluded.write(canonical(dict(provenance, reason=exclusion)) + "\n")
                continue
            if len(raw) != len(mapping.columns):
                balance["quarantined"] += 1
                reasons["ragged_record"] += 1
                quarantine.write(canonical(dict(provenance, reason="ragged_record")) + "\n")
                continue
            fields, analytics, keys, errors = [], [], {}, []
            blockers = list(mapping.analysis_blockers)
            for value, column in zip(raw, mapping.columns, strict=True):
                normalized, state, history = convert(value, column)
                state_counts[f"{column.field}:{state}"] += 1
                if state == "parse_error":
                    errors.append(column.field)
                if column.transform == "unknown":
                    blockers.append("unknown:" + column.field)
                if column.key_role:
                    if state == "observed" and normalized != "":
                        keys[column.key_role] = normalized
                    else:
                        blockers.append("business_key_component_unavailable:" + column.key_role)
                fields.append(FieldValue(
                    schema_version="1.0", name=column.field, original_value=value,
                    normalized_value=normalized, unit=column.unit, state=state,
                    source=SourceCoordinates(artifact_id=artifact_id, table=mapping.table,
                                             row=start_line, column=f"column_{column.position:04d}"),
                ))
                analytics.append({"name": column.field, "original_value": value,
                                  "normalized_string": normalized if isinstance(normalized, str) else None,
                                  "normalized_integer": normalized if type(normalized) is int else None,
                                  "normalized_number": normalized if type(normalized) is float else None,
                                  "state": state, "unit": column.unit, "column_position": column.position,
                                  "transformation_history": history})
            record = Record(schema_version="1.0", id=uuid5(dataset_id, f"record:{ordinal}"),
                            dataset_id=dataset_id, business_key=BusinessKey(**keys) if "stand" in keys else None,
                            fields=tuple(fields))
            if errors:
                balance["quarantined"] += 1
                reasons["parse_error"] += 1
                quarantine.write(canonical(dict(provenance, reason="parse_error", fields=errors,
                                                 record=record.model_dump(mode="json"),
                                                 transformations=analytics)) + "\n")
                continue
            if record.business_key is None:
                blockers.append("business_key_unavailable")
            blockers = sorted(set(blockers))
            blocked += bool(blockers)
            balance["accepted"] += 1
            balance["normalized_records"] += 1
            batch.append({"record_id": str(record.id), "dataset_id": str(dataset_id),
                          "record_level": mapping.record_level, "source_record_ordinal": ordinal,
                          "source_line_start": start_line, "source_line_end": end_line,
                          "fields": analytics, "record_json": record.model_dump_json(),
                          "analysis_blocked": bool(blockers), "blockers": blockers})
            if len(batch) >= 500:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    source_after = sha256(checked_source(source))
    if source_after != inspected["sha256"]:
        raise ValueError("source_changed_during_import")
    if balance["input_records"] != balance["accepted"] + balance["quarantined"] + balance["excluded"]:
        raise ValueError("unbalanced_import")
    report = {"schema_version": "1.0", "pipeline_version": PIPELINE_VERSION, "balance": balance,
              "state_counts": dict(sorted(state_counts.items())), "reasons": dict(sorted(reasons.items())),
              "analysis_blocked_records": blocked, "row_expansion": "one_to_one_no_join_no_aggregation",
              "source_unchanged": True, "incomplete": False}
    (staging / "report.json").write_text(canonical(report), encoding="utf-8")
    names = ["normalized.parquet", "quarantine.jsonl", "excluded.jsonl", "report.json", "mapping.json"]
    manifest = {"schema_version": "1.0", "pipeline_version": PIPELINE_VERSION,
                "dataset_id": str(dataset_id), "source_name": source.name,
                "source_sha256": inspected["sha256"], "source_sha256_after": source_after,
                "mapping_sha256": mapping_hash, "encoding": mapping.encoding,
                "output_hashes": {name: sha256(staging / name) for name in names}}
    (staging / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
    staging.rename(destination)
    return destination
