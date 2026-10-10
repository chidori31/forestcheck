"""Only generated fixtures under work/: never point these tests at owner data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile_local.py"


def xlsx(path, *, formula=False, unsafe=False):
    """Minimal OOXML fixture; synthetic strings have no relation to contest data."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", '<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Synthetic" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        cell = '<c r="B2"><f>1+1</f><v>2</v></c>' if formula else '<c r="B2"><v>2</v></c>'
        z.writestr("xl/worksheets/sheet1.xml", '<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>synthetic_id</t></is></c><c r="B1" t="inlineStr"><is><t>height</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>ROW_CANARY</t></is></c>' + cell + '</row><row r="3"><c r="A3" t="inlineStr"><is><t>x</t></is></c></row></sheetData></worksheet>')
        if unsafe:
            z.writestr("../escape.txt", "do not extract")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic-", dir=ROOT / "work")
        self.base = Path(self.tmp.name)
        self.data = self.base / "input"
        self.private = self.base / "private"
        self.data.mkdir()
        self.private.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *extra, raw=None):
        args = raw if raw is not None else ["--input", str(self.data), "--private-dir", str(self.private), "--confirm-private-dir", *extra]
        result = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], capture_output=True, text=True, timeout=30)
        self.assertNotIn(str(self.base), result.stdout + result.stderr)
        self.assertNotIn("ROW_CANARY", result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        return result

    def report(self):
        reports = sorted(self.private.glob("*/profile.json"))
        self.assertEqual(len(reports), 1)
        return json.loads(reports[0].read_text(encoding="utf-8"))

    def test_csv_hash_missing_types_no_values(self):
        raw = b"id,height,note\n001,2,ROW_CANARY\n002,,\n003,0,NA\n"
        (self.data / "sample.csv").write_bytes(raw)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0)
        report = self.report()
        file = report["files"][0]
        self.assertEqual(file["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(file["size_bytes"], len(raw))
        table = file["tables"][0]
        self.assertEqual(table["rows_profiled"], 3)
        self.assertTrue(table["complete"])
        self.assertEqual(table["columns"][0]["types"], {"string": 3})
        self.assertEqual(table["columns"][1]["missing_count"], 1)
        self.assertAlmostEqual(table["columns"][1]["missing_fraction"], 1 / 3)
        self.assertEqual(table["columns"][2]["missing_count"], 1)
        self.assertNotIn("ROW_CANARY", json.dumps(report))

    def test_tsv_missing_count(self):
        (self.data / "sample.tsv").write_text("id\tvalue\na\t\nb\t4\n", encoding="utf-8")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.report()["files"][0]["tables"][0]["columns"][1]["missing_count"], 1)

    def test_json_late_keys_and_jsonl_nulls(self):
        (self.data / "sample.json").write_text('[{"a":1},{"a":null,"b":true},{"b":false}]', encoding="utf-8")
        (self.data / "sample.jsonl").write_text('{"x":0}\n{"x":null}\n', encoding="utf-8")
        self.assertEqual(self.run_cli().returncode, 0)
        by_name = {f["relative_path"]: f for f in self.report()["files"]}
        cols = by_name["sample.json"]["tables"][0]["columns"]
        self.assertEqual([c["missing_count"] for c in cols], [2, 1])
        self.assertEqual(cols[1]["types"], {"boolean": 2})

    def test_geojson_layer_no_coordinate_values(self):
        (self.data / "map.geojson").write_text(json.dumps({"type":"FeatureCollection", "features":[{"type":"Feature","properties":{"age":None},"geometry":{"type":"Point","coordinates":[123.456789,65.4321]}},{"type":"Feature","properties":{"age":3},"geometry":None}]}), encoding="utf-8")
        self.assertEqual(self.run_cli().returncode, 0)
        report = self.report()
        table = report["files"][0]["tables"][0]
        self.assertEqual(table["kind"], "layer")
        self.assertEqual(table["geometry_types"], {"Point": 1})
        self.assertEqual(table["geometry_missing_count"], 1)
        self.assertNotIn("123.456789", json.dumps(report))

    def test_xlsx_sheets_formulas_and_sparse_missing(self):
        xlsx(self.data / "sample.xlsx", formula=True)
        self.assertEqual(self.run_cli().returncode, 0)
        report = self.report()
        table = report["files"][0]["tables"][0]
        self.assertEqual(table["name"], "Synthetic")
        self.assertEqual(table["columns"][1]["types"], {"formula": 1})
        self.assertEqual(table["columns"][1]["missing_count"], 1)
        self.assertNotIn("1+1", json.dumps(report))

    def test_archives_and_unknown_formats_explicitly_unsupported(self):
        with zipfile.ZipFile(self.data / "sample.zip", "w") as z:
            z.writestr("../../escape.txt", "ROW_CANARY")
        (self.data / "sample.shp").write_bytes(b"synthetic unsupported")
        (self.data / "sample.xlsm").write_bytes(b"synthetic macro file")
        self.assertEqual(self.run_cli().returncode, 4)
        report = self.report()
        self.assertEqual([f["status"] for f in report["files"]], ["unsupported"] * 3)
        self.assertFalse((self.base / "escape.txt").exists())
        self.assertEqual(len(list(self.private.rglob("*"))), 2)

    def test_xlsx_traversal_and_compression_limits(self):
        xlsx(self.data / "bad.xlsx", unsafe=True)
        with zipfile.ZipFile(self.data / "bomb.xlsx", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("xl/workbook.xml", "a" * 100000)
        self.assertEqual(self.run_cli().returncode, 4)
        self.assertTrue(all(f["status"] == "error" for f in self.report()["files"]))
        self.assertFalse((self.base / "escape.txt").exists())

    def test_partial_rows_not_reported_as_complete(self):
        (self.data / "sample.csv").write_text("a\n1\n2\n3\n", encoding="utf-8")
        self.assertEqual(self.run_cli("--max-rows", "2").returncode, 4)
        table = self.report()["files"][0]["tables"][0]
        self.assertFalse(table["complete"])
        self.assertEqual(table["rows_profiled"], 2)
        self.assertEqual(table["columns"][0]["missing_fraction"], 0)

    def test_empty_header_only_duplicate_and_ragged_csv(self):
        (self.data / "empty.csv").write_bytes(b"")
        (self.data / "header.csv").write_text("a,b\n", encoding="utf-8")
        (self.data / "bad.csv").write_text("a,a\n1,2,3\n", encoding="utf-8")
        self.assertEqual(self.run_cli().returncode, 4)
        files = {f["relative_path"]: f for f in self.report()["files"]}
        self.assertEqual(files["bad.csv"]["status"], "error")
        self.assertEqual(files["empty.csv"]["status"], "ok")
        self.assertIsNone(files["header.csv"]["tables"][0]["columns"][0]["missing_fraction"])

    def test_config_errors_never_echo_paths_or_values(self):
        cases = [[], ["--input", str(self.data)], ["--unknown", "ROW_CANARY"],
                 ["--input", str(self.data), "--private-dir", str(self.data), "--confirm-private-dir"],
                 ["--input", "https://example.invalid/ROW_CANARY", "--private-dir", str(self.private), "--confirm-private-dir"],
                 ["--input", "\\\\server\\share", "--private-dir", str(self.private), "--confirm-private-dir"]]
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(raw=args).returncode, 2)
        self.assertFalse(list(self.private.iterdir()))

    def test_file_and_entry_limits_are_explicit(self):
        (self.data / "a.csv").write_text("a\n" + "x\n" * 20, encoding="utf-8")
        (self.data / "b.csv").write_text("a\n1\n", encoding="utf-8")
        self.assertEqual(self.run_cli("--max-file-bytes", "10", "--max-files", "1").returncode, 4)
        report = self.report()
        self.assertFalse(report["inventory_complete"])
        self.assertEqual(len(report["files"]), 1)

    def test_hardlinks_not_read(self):
        target = self.base / "private-canary.csv"
        target.write_text("secret\nROW_CANARY\n", encoding="utf-8")
        os.link(target, self.data / "linked.csv")
        self.assertEqual(self.run_cli().returncode, 4)
        file = self.report()["files"][0]
        self.assertEqual(file["status"], "skipped_link")
        self.assertIsNone(file["sha256"])

    def test_repeated_run_does_not_overwrite(self):
        (self.data / "a.csv").write_text("a\n1\n", encoding="utf-8")
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual(len(list(self.private.glob("*/profile.json"))), 2)

    def test_bad_encoding_and_broken_json_have_no_exception_values(self):
        (self.data / "bad.csv").write_bytes(b"a\n\xff\n")
        (self.data / "bad.json").write_text('{"ROW_CANARY":', encoding="utf-8")
        self.assertEqual(self.run_cli().returncode, 4)
        report = self.report()
        self.assertTrue(all(f["status"] == "error" for f in report["files"]))
        self.assertNotIn("ROW_CANARY", json.dumps(report))

    def test_explicit_cp1251_semicolon(self):
        (self.data / "sample.csv").write_text("код;высота\nА;2,5\nБ;\n", encoding="cp1251")
        self.assertEqual(self.run_cli("--encoding", "cp1251", "--delimiter", ";").returncode, 0)
        table = self.report()["files"][0]["tables"][0]
        self.assertEqual(table["columns"][0]["name"], "код")
        self.assertEqual(table["columns"][1]["types"], {"string": 1})
        self.assertEqual(table["columns"][1]["missing_count"], 1)

    def test_json_duplicate_keys_and_nan_rejected(self):
        (self.data / "dup.json").write_text('[{"a":1,"a":2}]', encoding="utf-8")
        (self.data / "nan.json").write_text('[{"a":NaN}]', encoding="utf-8")
        self.assertEqual(self.run_cli().returncode, 4)
        self.assertTrue(all(f["status"] == "error" for f in self.report()["files"]))

    def test_symlink_or_junction_not_followed(self):
        target = self.base / "outside-input"
        target.mkdir()
        (target / "never-read.csv").write_text("secret\nROW_CANARY\n", encoding="utf-8")
        link = self.data / "directory-link"
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
        else:
            link.symlink_to(target, target_is_directory=True)
        self.assertEqual(self.run_cli().returncode, 4)
        report = self.report()
        self.assertEqual(len(report["files"]), 1)
        self.assertEqual(report["files"][0]["status"], "skipped_link")
        self.assertNotIn("never-read.csv", json.dumps(report))
        args = ["--input", str(link), "--private-dir", str(self.private), "--confirm-private-dir"]
        self.assertEqual(self.run_cli(raw=args).returncode, 2)

    def test_private_output_must_exist_and_not_overlap(self):
        inside = self.data / "private"
        inside.mkdir()
        for private in (inside, self.base, self.base / "does-not-exist"):
            args = ["--input", str(self.data), "--private-dir", str(private), "--confirm-private-dir"]
            self.assertNotEqual(self.run_cli(raw=args).returncode, 0)
        self.assertFalse(list(inside.iterdir()))

    def test_structured_and_column_limits(self):
        (self.data / "small.json").write_text('[{"a":1,"b":2}]', encoding="utf-8")
        self.assertEqual(self.run_cli("--max-columns", "1").returncode, 4)
        self.assertEqual(self.report()["files"][0]["reason"], "column_limit")

    def test_row_at_exact_limit_complete(self):
        (self.data / "a.csv").write_text("a\n1\n2\n", encoding="utf-8")
        self.assertEqual(self.run_cli("--max-rows", "2").returncode, 0)
        self.assertTrue(self.report()["files"][0]["tables"][0]["complete"])

    def test_no_network_or_writes_outside_private(self):
        (self.data / "a.csv").write_text("a\n1\n", encoding="utf-8")
        before = (self.data / "a.csv").read_bytes()
        # Audit the actual CLI in a fresh process; reject network and all file
        # writes outside the explicitly selected private directory.
        wrapper = '''import os, pathlib, runpy, sys
script, source, private = sys.argv[1:]
allowed = pathlib.Path(private).resolve()
def audit(event, args):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system"):
        raise RuntimeError("forbidden network or child process")
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not isinstance(path, int) and not pathlib.Path(path).resolve().is_relative_to(allowed):
            raise RuntimeError("write outside private")
sys.addaudithook(audit)
sys.argv = [script, "--input", source, "--private-dir", private, "--confirm-private-dir"]
runpy.run_path(script, run_name="__main__")
'''
        result = subprocess.run([sys.executable, "-I", "-B", "-c", wrapper, str(SCRIPT), str(self.data), str(self.private)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(before, (self.data / "a.csv").read_bytes())
        self.assertEqual(len(list(self.private.rglob("*"))), 2)

    def rewrite_xlsx(self, path, member, replacement):
        with zipfile.ZipFile(path) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        members[member] = replacement.encode("utf-8")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)

    def test_broken_xlsx_root_not_complete_empty_sheet(self):
        path = self.data / "broken.xlsx"
        xlsx(path)
        self.rewrite_xlsx(path, "xl/worksheets/sheet1.xml", "<broken/>")
        self.assertEqual(self.run_cli().returncode, 4)
        self.assertEqual(self.report()["files"][0]["status"], "error")

    def test_xlsx_chartsheet_explicitly_unsupported(self):
        path = self.data / "chart.xlsx"
        xlsx(path)
        self.rewrite_xlsx(path, "xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chartsheet" Target="worksheets/sheet1.xml"/></Relationships>')
        self.assertEqual(self.run_cli().returncode, 4)
        self.assertEqual(self.report()["files"][0]["reason"], "unsupported_sheet_kind")

    def test_xlsx_long_reference_rejected_as_invalid_syntax(self):
        path = self.data / "long-ref.xlsx"
        xlsx(path)
        # Even when max-columns is relaxed, cell addresses must have a bounded
        # Excel reference syntax BEFORE integer conversion.
        ref = "Z" * 64 + "1"
        self.rewrite_xlsx(path, "xl/worksheets/sheet1.xml", '<worksheet><sheetData><row r="1"><c r="' + ref + '"><v>1</v></c></row></sheetData></worksheet>')
        self.assertEqual(self.run_cli().returncode, 4)
        self.assertEqual(self.report()["files"][0]["reason"], "invalid_cell_reference")

    def test_csv_positional_headers_preserve_duplicate_columns(self):
        (self.data / "duplicate.csv").write_text("value,value\n1,\n2,3\n", encoding="utf-8")
        self.assertEqual(self.run_cli("--duplicate-headers", "positions").returncode, 0)
        table = self.report()["files"][0]["tables"][0]
        self.assertEqual([c["name"] for c in table["columns"]], ["column_0001", "column_0002"])
        self.assertEqual(table["source_headers"], ["value", "value"])
        self.assertEqual([c["missing_count"] for c in table["columns"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
