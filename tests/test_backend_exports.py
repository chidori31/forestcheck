import csv
import io
import json
import zipfile
from xml.etree import ElementTree as ET

import pytest
from forestcheck.api.contracts import ExportRequest, ReviewRequest, VersionRequest
from forestcheck.api.execution import Worker
from forestcheck.api.exporting import safe_text
from test_backend_workflow import analysed


@pytest.mark.parametrize("text", ["=1+1", " +SUM(A1)", "\t@x", "\ufeff -1", '\r\n=HYPERLINK("x")'])
def test_formula_text_is_protected_without_mutating_source(text):
    assert safe_text(text) == "'" + text
    assert safe_text("ordinary") == "ordinary"
    assert safe_text(-3.0) == -3.0 and safe_text(0) == 0


def export_file(service, store, version, kind, fmt, analysis_id=None):
    job = service.export(
        ExportRequest(
            schema_version="1.0", version_id=version, kind=kind, format=fmt, analysis_id=analysis_id
        ),
        "operator",
        kind + fmt,
    )
    Worker(store).run_once()
    assert service.job(job["id"])["status"] == "succeeded"
    return service.download(job["id"])[0]


def test_original_report_and_corrected_export_real_workflow(tmp_path):
    _, store, service, dataset, version, _, run, detail, suggestion = analysed(tmp_path)
    original = export_file(service, store, version, "original", "csv")
    rows = list(csv.reader(io.StringIO(original.read_text("utf-8-sig")), delimiter=";"))
    assert rows[1] == ["001", "10.0", "2.0", "", "'=1+1"]
    assert rows[2][1:4] == ["-3.0", "0.0", "0.0"]
    original_json = json.loads(
        export_file(service, store, version, "original", "json").read_text("utf-8")
    )
    assert original_json["kind"] == "original" and original_json["source"]["content_base64"]
    value = original_json["records"][0]["fields"][-1]
    assert value["original_value"] == "=1+1"
    report = json.loads(
        export_file(service, store, version, "report", "json", run["id"]).read_text("utf-8")
    )
    assert len(report["findings"]) == 1 and report["analysis"]["status"] == "succeeded"
    event = service.review(
        ReviewRequest(
            schema_version="1.0",
            finding_id=detail["finding"]["id"],
            expected_version=0,
            error_assessment="confirmed_error",
            suggestion_id=suggestion["suggestion"]["id"],
            suggestion_decision="accepted",
        ),
        "operator",
        "review",
    )
    job = service.create_version(
        dataset["id"],
        VersionRequest(
            schema_version="1.0",
            base_version_id=version,
            analysis_id=run["id"],
            expected_version=service.dataset(dataset["id"])["etag"],
            review_event_ids=(event["id"],),
        ),
        "operator",
        "version",
    )
    Worker(store).run_once()
    changed = service.job(job["id"])["result"]["version_id"]
    xlsx = export_file(service, store, changed, "corrected", "xlsx")
    with zipfile.ZipFile(xlsx) as book:
        assert book.testzip() is None
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cells = {c.attrib["r"]: c for c in sheet.findall(".//x:c", ns)}
    assert cells["D2"].find("x:v", ns).text == "20"
    assert cells["B3"].find("x:v", ns).text == "-3.0"
    assert (
        cells["A2"].attrib["t"] == "inlineStr" and cells["E2"].find("x:is/x:t", ns).text == "'=1+1"
    )
    assert not sheet.findall(".//x:f", ns)
    unchanged = service.records(version)["items"][0]
    assert next(f for f in unchanged["fields"] if f["name"] == "total")["state"] == "missing"


def test_unaccepted_version_and_wrong_export_kind_are_rejected(tmp_path):
    _, store, service, _, version, _, run, _, _ = analysed(tmp_path)
    from forestcheck.storage.application import WorkflowError

    with pytest.raises(WorkflowError):
        service.export(
            ExportRequest(schema_version="1.0", version_id=version, kind="corrected", format="csv"),
            "operator",
            "invalid",
        )
    assert service.job(run["id"])["status"] == "succeeded"


@pytest.mark.parametrize("text", ["\ufffe", "\uffff", "\ud800"])
def test_xlsx_rejects_xml_forbidden_characters(tmp_path, text):
    from forestcheck.api.exporting import write_xlsx
    from forestcheck.storage.application import WorkflowError

    with pytest.raises(WorkflowError, match="xlsx_invalid_xml_character"):
        write_xlsx(tmp_path / "invalid.xlsx", [[text]], lambda: None)
