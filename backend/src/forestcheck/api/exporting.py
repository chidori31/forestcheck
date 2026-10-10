"""Internal JSON/CSV/XLSX exchange; no formulas, macros or source writes."""

import base64
import csv
import json
import math
import time
import zipfile
from uuid import uuid4
from xml.sax.saxutils import escape

from forestcheck.api.execution import read_records
from forestcheck.api.service import Service
from forestcheck.ingest.csv_import import canonical, convert
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.storage.application import WorkflowError


def safe_text(value):
    if not isinstance(value, str):
        return value
    stripped = value
    while stripped and (stripped[0].isspace() or stripped[0] == "\ufeff"):
        stripped = stripped[1:]
    if stripped.startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def column_name(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell(value, address):
    if value is None:
        return f'<c r="{address}"/>'
    if type(value) is bool:
        return f'<c r="{address}" t="b"><v>{int(value)}</v></c>'
    if type(value) in (int, float):
        if not math.isfinite(value) or (type(value) is int and abs(value) > 2**53):
            raise WorkflowError("xlsx_numeric_precision_limit", 422)
        return f'<c r="{address}" t="n"><v>{value}</v></c>'
    text = safe_text(str(value))
    if any(
        not (
            c in "\t\n\r"
            or 0x20 <= ord(c) <= 0xD7FF
            or 0xE000 <= ord(c) <= 0xFFFD
            or 0x10000 <= ord(c) <= 0x10FFFF
        )
        for c in text
    ):
        raise WorkflowError("xlsx_invalid_xml_character", 422)
    if len(text) > 32767:
        raise WorkflowError("xlsx_text_limit", 422)
    return f'<c r="{address}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def write_xlsx(path, rows, checkpoint):
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="ForestCheck" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        with archive.open("xl/worksheets/sheet1.xml", "w") as stream:
            stream.write(
                b'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            )
            for row_number, row in enumerate(rows, 1):
                if row_number > 1048576 or len(row) > 16384:
                    raise WorkflowError("xlsx_dimensions_limit", 422)
                content = (
                    f'<row r="{row_number}">'
                    + "".join(
                        cell(v, column_name(i) + str(row_number)) for i, v in enumerate(row, 1)
                    )
                    + "</row>"
                )
                stream.write(content.encode("utf-8"))
                if row_number % 1000 == 0:
                    checkpoint()
            stream.write(b"</sheetData></worksheet>")


def tabular_rows(store, version, mapping, kind):
    if kind == "original":
        dataset = store.one("datasets", version["dataset_id"])
        source = store.resolve(dataset["source_artifact_id"])
        with source.open(encoding=mapping.encoding, newline="") as stream:
            reader = csv.reader(stream, delimiter=mapping.delimiter, strict=True)
            yield next(reader, [])
            for row in reader:
                values = []
                for i, text in enumerate(row):
                    parsed, state, _ = (
                        convert(text, mapping.columns[i])
                        if i < len(mapping.columns)
                        else (None, "unknown", ())
                    )
                    values.append(
                        parsed
                        if state == "observed" and type(parsed) in (int, float, bool)
                        else text
                    )
                yield values
    else:
        yield [c.source_header for c in mapping.columns]
        for record in read_records(store.resolve(version["artifact_id"])):
            fields = {f.name: f for f in record.fields}
            yield [
                fields[c.field].normalized_value
                if fields[c.field].state == "observed"
                else fields[c.field].original_value
                for c in mapping.columns
            ]


def report_snapshot(store, analysis_id):
    with store.transaction() as cx:
        findings = [
            json.loads(r[0])
            for r in cx.execute(
                "SELECT content_json FROM findings WHERE analysis_id=? ORDER BY priority,id",
                (analysis_id,),
            )
        ]
        suggestions = [
            json.loads(r[0])
            for r in cx.execute(
                "SELECT content_json FROM suggestions WHERE analysis_id=? ORDER BY id",
                (analysis_id,),
            )
        ]
        reviews = [
            json.loads(r[0])
            for r in cx.execute(
                "SELECT r.content_json FROM review_events r JOIN findings f ON f.id=r.finding_id WHERE f.analysis_id=? ORDER BY r.created_at,r.id",
                (analysis_id,),
            )
        ]
    return {
        "schema_version": "1.0",
        "kind": "report",
        "created_at": time.time(),
        "analysis": Service(store).job(analysis_id),
        "findings": findings,
        "suggestions": suggestions,
        "reviews": reviews,
        "official_submission_format": False,
    }


def report_rows(report):
    yield [
        "finding_id",
        "record_id",
        "field",
        "signal_kind",
        "severity",
        "priority",
        "evidence",
        "suggestions",
        "review_events",
    ]
    for finding in report["findings"]:
        suggestions = [
            s for s in report["suggestions"] if s["suggestion"]["finding_id"] == finding["id"]
        ]
        reviews = [r for r in report["reviews"] if r["finding_id"] == finding["id"]]
        yield [
            finding["id"],
            finding["record_id"],
            finding["field_name"],
            finding["signal_kind"],
            finding["severity"],
            finding["priority"],
            canonical(finding["evidence"]),
            canonical(suggestions),
            canonical(reviews),
        ]


def create_export(store, claim, checkpoint):
    request = claim["request"]
    version = store.one("versions", request["version_id"])
    dataset = store.one("datasets", version["dataset_id"])
    mapping = ImportMapping.model_validate_json(dataset["mapping_json"])
    source = store.resolve(dataset["source_artifact_id"])
    normalized = store.resolve(version["artifact_id"])
    fmt, kind = request["format"], request["kind"]
    path = store.root / "exports" / (str(uuid4()) + "." + fmt)
    report = report_snapshot(store, request["analysis_id"]) if kind == "report" else None
    if fmt == "json":
        with path.open("x", encoding="utf-8") as stream:
            if report:
                stream.write(canonical(report))
            else:
                metadata = {
                    "schema_version": "1.0",
                    "kind": kind,
                    "version_id": version["id"],
                    "mapping": mapping.model_dump(mode="json"),
                    "import_report": json.loads(version["import_report_json"]),
                    "official_submission_format": False,
                    "source": {"sha256": dataset["source_sha256"], "encoding": mapping.encoding},
                }
                if kind == "original":
                    metadata["source"]["content_base64"] = base64.b64encode(
                        source.read_bytes()
                    ).decode("ascii")
                stream.write(canonical(metadata)[:-1] + ',"records":[')
                for i, record in enumerate(read_records(normalized)):
                    if i:
                        stream.write(",")
                    stream.write(record.model_dump_json())
                    if i % 1000 == 0:
                        checkpoint()
                stream.write("]}")
    else:
        rows = report_rows(report) if report else tabular_rows(store, version, mapping, kind)
        if fmt == "csv":
            with path.open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, delimiter=";")
                for i, row in enumerate(rows):
                    writer.writerow([safe_text(v) for v in row])
                    if i % 1000 == 0:
                        checkpoint()
        elif fmt == "xlsx":
            write_xlsx(path, rows, checkpoint)
        else:
            raise WorkflowError("unsupported_export_format", 415)
    store.resolve(dataset["source_artifact_id"])
    store.resolve(version["artifact_id"])
    media = {
        "json": "application/json",
        "csv": "text/csv; charset=utf-8",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }[fmt]
    artifact_id = store.artifact(path, media)
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "format": fmt,
        "version_id": version["id"],
        "text_formula_protection": fmt in {"csv", "xlsx"},
        "raw_preserved_in_system": True,
    }
