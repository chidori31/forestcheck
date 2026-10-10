#!/usr/bin/env python3
"""Offline inventory for owner use. See docs/data_contract.md before running.

Python 3.11+, standard library only. Never evaluates cells, imports data code,
extracts archives, sends requests, copies rows or prints paths/exceptions.
Private JSON contains identifying metadata: it is NOT safe to upload by default.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile

VERSION = "1.1.0"
SUPPORTED = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".geojson", ".xlsx"}
FORMULA = object()
INTEGER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z")
NUMBER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class ProfileError(Exception):
    """Only constant reason codes belong here, never a parser's exception text."""


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ProfileError("invalid_arguments")


def arguments(argv):
    p = QuietParser(add_help=False, allow_abbrev=False)
    p.add_argument("--input", required=True)
    p.add_argument("--private-dir", required=True)
    p.add_argument("--confirm-private-dir", action="store_true", required=True)
    p.add_argument("--encoding", choices=["utf-8-sig", "utf-8", "cp1251", "utf-16"], default="utf-8-sig")
    p.add_argument("--delimiter", choices=["auto", ",", ";", "tab", "|"], default="auto")
    p.add_argument("--duplicate-headers", choices=["reject", "positions"], default="reject")
    defaults = {"max-files": 10000, "max-entries": 20000, "max-depth": 32,
                "max-file-bytes": 256 * 1024**2, "max-total-bytes": 2 * 1024**3,
                "max-structured-bytes": 16 * 1024**2, "max-rows": 100000,
                "max-columns": 2048, "max-field-chars": 1024**2,
                "max-zip-members": 1000, "max-zip-member-bytes": 16 * 1024**2,
                "max-zip-total-bytes": 64 * 1024**2, "max-zip-ratio": 100}
    for key, value in defaults.items():
        p.add_argument("--" + key, type=int, default=value)
    args = p.parse_args(argv)
    if any(getattr(args, k.replace("-", "_")) < 1 for k in defaults):
        raise ProfileError("invalid_limit")
    return args


def is_link(st):
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def local_directory(raw):
    # Reject UNC and device namespaces before touching the filesystem.
    if raw.startswith(("\\\\", "//")) or "://" in raw or "\x00" in raw:
        raise ProfileError("not_local_path")
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ProfileError("absolute_path_required")
    if os.name == "nt":
        import ctypes
        if ":" in str(path)[2:] or ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) != 3:
            raise ProfileError("fixed_local_drive_required")
    for part in reversed((path, *path.parents)):
        if is_link(part.lstat()):
            raise ProfileError("linked_directory")
    if not path.is_dir():
        raise ProfileError("directory_required")
    return path.resolve(strict=True)


def signature(st):
    # Windows Python versions expose different ctime meanings via stat/fstat.
    # Identity, length and mtime are comparable through both interfaces.
    stable = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    return stable if os.name == "nt" else stable + (st.st_ctime_ns,)


@contextmanager
def open_regular(path, expected, root):
    # Static links/junctions/hardlinks are excluded. Use a quiescent local input;
    # these checks are not an OS sandbox against a concurrent hostile writer.
    for parent in (path.parent, *path.parent.parents):
        if is_link(parent.lstat()):
            raise ProfileError("linked_parent")
        if parent == root:
            break
    before = path.lstat()
    if is_link(before) or before.st_nlink != 1 or not stat.S_ISREG(before.st_mode):
        raise ProfileError("unsafe_file")
    if signature(before) != signature(expected):
        raise ProfileError("changed_during_run")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        if signature(os.fstat(stream.fileno())) != signature(expected):
            raise ProfileError("changed_during_run")
        yield stream
        if signature(os.fstat(stream.fileno())) != signature(expected) or signature(path.lstat()) != signature(expected):
            raise ProfileError("changed_during_run")


def value_type(value):
    if value is FORMULA:
        return "formula"
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    text = value.strip()
    if text.lower() in ("true", "false"):
        return "boolean"
    if INTEGER.fullmatch(text):
        return "integer"
    if NUMBER.fullmatch(text):
        return "number"
    return "string"


class Table:
    def __init__(self, name, kind, args, headers=None):
        self.name, self.kind, self.args = name, kind, args
        self.rows = 0
        self.columns = {}
        self.complete = True
        if headers is not None:
            if any(not isinstance(h, str) or not h.strip() for h in headers) or len(headers) != len(set(headers)):
                raise ProfileError("ambiguous_header")
            for h in headers:
                self.add_column(h)

    def add_column(self, key):
        if len(self.columns) >= self.args.max_columns or len(key) > self.args.max_field_chars:
            raise ProfileError("column_limit")
        self.columns[key] = Counter()

    def add(self, row):
        if not isinstance(row, dict):
            raise ProfileError("object_row_required")
        for key, value in row.items():
            if key not in self.columns:
                self.add_column(key)
            if isinstance(value, str) and len(value) > self.args.max_field_chars:
                raise ProfileError("field_limit")
            typ = value_type(value)
            if typ:
                self.columns[key][typ] += 1
        self.rows += 1

    def result(self):
        columns = []
        for i, (key, types) in enumerate(self.columns.items(), 1):
            missing = self.rows - sum(types.values())
            columns.append({"position": i, "name": key, "types": dict(types),
                            "missing_count": missing,
                            "missing_fraction": missing / self.rows if self.rows else None})
        return {"name": self.name, "kind": self.kind, "rows_profiled": self.rows,
                "complete": self.complete, "columns": columns}


def profile_csv(stream, suffix, args):
    csv.field_size_limit(args.max_field_chars)
    text = io.TextIOWrapper(stream, encoding=args.encoding, newline="")
    try:
        sample = text.read(65536)
        if "\x00" in sample:
            raise ProfileError("binary_text")
        text.seek(0)
        if args.delimiter != "auto":
            delimiter = "\t" if args.delimiter == "tab" else args.delimiter
            policy = "explicit"
        elif suffix == ".tsv":
            delimiter, policy = "\t", "tsv_default"
        else:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                policy = "sniffed_verify_locally"
            except csv.Error:
                delimiter, policy = ",", "comma_fallback_verify_locally"
        reader = csv.reader(text, delimiter=delimiter, strict=True)
        headers = next(reader, [])
        source_headers = list(headers)
        if args.duplicate_headers == "positions":
            headers = [f"column_{i:04d}" for i in range(1, len(headers) + 1)]
        table = Table("table", "table", args, headers)
        blank = 0
        for row in reader:
            if not row:
                blank += 1
                continue
            if table.rows >= args.max_rows:
                table.complete = False
                break
            if len(row) != len(headers):
                raise ProfileError("ragged_row")
            table.add(dict(zip(headers, row)))
        result = table.result()
        result.update(encoding=args.encoding, delimiter=delimiter, delimiter_policy=policy,
                      skipped_blank_records=blank, header_policy="first_record",
                      source_headers=source_headers, column_identity=args.duplicate_headers)
        return [result]
    finally:
        text.detach()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError("duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value):
    raise ProfileError("nonstandard_json_number")


def load_json(data):
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def profile_json(stream, suffix, args):
    if suffix in (".jsonl", ".ndjson"):
        table = Table("table", "table", args)
        text = io.TextIOWrapper(stream, encoding=args.encoding)
        try:
            while True:
                line = text.readline(args.max_field_chars + 1)
                if not line:
                    break
                if len(line) > args.max_field_chars:
                    raise ProfileError("line_limit")
                if not line.strip():
                    continue
                if table.rows >= args.max_rows:
                    table.complete = False
                    break
                table.add(load_json(line))
        finally:
            text.detach()
        return [table.result()]
    raw = stream.read(args.max_structured_bytes + 1)
    if len(raw) > args.max_structured_bytes:
        raise ProfileError("structured_byte_limit")
    obj = load_json(raw.decode(args.encoding))
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        rows = obj.get("features")
        if not isinstance(rows, list):
            raise ProfileError("invalid_feature_collection")
        table = Table("features", "layer", args)
        geometry = Counter()
        missing = 0
        for row in rows[:args.max_rows]:
            if not isinstance(row, dict) or row.get("type") != "Feature":
                raise ProfileError("invalid_feature")
            props = row.get("properties")
            table.add({} if props is None else props)
            geom = row.get("geometry")
            if geom is None:
                missing += 1
            elif isinstance(geom, dict) and geom.get("type") in {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"}:
                geometry[geom["type"]] += 1
            else:
                raise ProfileError("invalid_geometry_type")
        table.complete = len(rows) <= args.max_rows
        result = table.result()
        result.update(geometry_types=dict(geometry), geometry_missing_count=missing,
                      geometry_validation="not_performed", crs="not_inferred")
        return [result]
    if suffix == ".geojson" or not isinstance(obj, list):
        raise ProfileError("unsupported_json_structure")
    table = Table("table", "table", args)
    for row in obj[:args.max_rows]:
        table.add(row)
    table.complete = len(obj) <= args.max_rows
    return [table.result()]


def local_tag(element):
    return element.tag.rsplit("}", 1)[-1]


def xml_read(archive, name, args):
    raw = archive.read(name)
    if len(raw) > args.max_zip_member_bytes:
        raise ProfileError("zip_member_limit")
    # Deliberately only UTF-8 OOXML. Never resolve DTDs/entities/external resources.
    text = raw.decode("utf-8-sig")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ProfileError("xml_entity_forbidden")
    return ET.fromstring(text)


def profile_xlsx(stream, args):
    with zipfile.ZipFile(stream) as z:
        entries = z.infolist()
        if len(entries) > args.max_zip_members:
            raise ProfileError("zip_member_count_limit")
        total = 0
        names = set()
        for e in entries:
            name = PurePosixPath(e.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in e.filename or ":" in e.filename or e.filename in names:
                raise ProfileError("unsafe_zip_path")
            names.add(e.filename)
            if e.flag_bits & 1 or stat.S_ISLNK(e.external_attr >> 16):
                raise ProfileError("unsafe_zip_member")
            if e.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ProfileError("unsupported_zip_compression")
            total += e.file_size
            if e.file_size > args.max_zip_member_bytes or total > args.max_zip_total_bytes or e.file_size > max(1, e.compress_size) * args.max_zip_ratio:
                raise ProfileError("zip_expansion_limit")
            if e.filename.lower().endswith("vbaproject.bin"):
                raise ProfileError("macro_content_forbidden")
        book = xml_read(z, "xl/workbook.xml", args)
        rels = xml_read(z, "xl/_rels/workbook.xml.rels", args)
        if local_tag(book) != "workbook" or local_tag(rels) != "Relationships":
            raise ProfileError("invalid_workbook_structure")
        targets = {}
        for rel in rels:
            if rel.attrib.get("TargetMode", "").lower() == "external":
                continue
            target = rel.attrib.get("Target", "")
            if target.startswith("/xl/"):
                target = target[1:]
            else:
                target = "xl/" + target
            if ".." in PurePosixPath(target).parts or "\\" in target or ":" in target:
                raise ProfileError("unsafe_relationship")
            targets[rel.attrib.get("Id")] = (target, rel.attrib.get("Type"))
        shared = []
        if "xl/sharedStrings.xml" in names:
            for si in xml_read(z, "xl/sharedStrings.xml", args):
                shared.append("".join(t.text or "" for t in si.iter() if local_tag(t) == "t"))
        results = []
        for sheet in book.iter():
            if local_tag(sheet) != "sheet":
                continue
            rid = next((v for k, v in sheet.attrib.items() if k.rsplit("}", 1)[-1] == "id"), None)
            if rid not in targets:
                raise ProfileError("missing_sheet_relationship")
            target, relationship_type = targets[rid]
            if relationship_type not in {
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
                "http://purl.oclc.org/ooxml/officeDocument/relationships/worksheet",
            }:
                raise ProfileError("unsupported_sheet_kind")
            root = xml_read(z, target, args)
            sheet_data = [c for c in root if local_tag(c) == "sheetData"]
            if local_tag(root) != "worksheet" or len(sheet_data) != 1:
                raise ProfileError("invalid_worksheet_structure")
            table = None
            headers = []
            header_row = last_row = None
            truncated = False
            for row in sheet_data[0]:
                if local_tag(row) != "row":
                    continue
                row_number = int(row.attrib["r"])
                if row_number < 1 or (last_row is not None and row_number <= last_row):
                    raise ProfileError("invalid_sheet_row_order")
                last_row = row_number
                cells = {}
                for cell in row:
                    if local_tag(cell) != "c":
                        continue
                    # Bound the address syntax BEFORE arbitrary-precision math.
                    ref = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]{0,6})", cell.attrib.get("r", ""))
                    if not ref or int(ref[2]) != row_number:
                        raise ProfileError("invalid_cell_reference")
                    column = 0
                    for letter in ref[1]:
                        column = column * 26 + ord(letter) - 64
                        if column > min(args.max_columns, 16384):
                            raise ProfileError("invalid_sheet_column")
                    if column > args.max_columns or column in cells:
                        raise ProfileError("invalid_sheet_column")
                    children = {local_tag(c): c for c in cell}
                    if "f" in children:
                        value = FORMULA
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter() if local_tag(t) == "t")
                    else:
                        value = children["v"].text if "v" in children else None
                        typ = cell.attrib.get("t")
                        if typ == "s" and value is not None:
                            index = int(value)
                            if index < 0 or index >= len(shared):
                                raise ProfileError("invalid_shared_string")
                            value = shared[index]
                        elif typ == "b" and value is not None:
                            if value not in ("0", "1"):
                                raise ProfileError("invalid_boolean")
                            value = value == "1"
                    cells[column] = value
                if table is None:
                    if not cells or all(v is None for v in cells.values()):
                        continue
                    width = max(cells)
                    headers = [cells.get(c) for c in range(1, width + 1)]
                    table = Table(sheet.attrib.get("name", "sheet"), "sheet", args, headers)
                    header_row = row_number
                    continue
                if cells and max(cells) > len(headers):
                    raise ProfileError("sheet_beyond_header")
                # Count blank physical rows between the header and last XML row;
                # ignore misleading worksheet dimension declarations.
                wanted = row_number - header_row
                while table.rows < min(wanted, args.max_rows):
                    current = table.rows + header_row + 1
                    table.add({h: cells.get(i) for i, h in enumerate(headers, 1)} if current == row_number else {})
                if wanted > args.max_rows:
                    truncated = True
                    break
            if table is None:
                table = Table(sheet.attrib.get("name", "sheet"), "sheet", args, [])
            table.complete = not truncated
            result = table.result()
            result.update(header_policy="first_nonempty_xml_row", header_row=header_row,
                          formula_policy="count_only_never_evaluate", date_policy="serials_not_converted",
                          sheet_state=sheet.attrib.get("state", "visible"))
            results.append(result)
        if not results:
            raise ProfileError("no_sheets")
        return results


def file_profile(path, st, root, args, budget):
    suffix = path.suffix.lower()
    result = {"relative_path": path.relative_to(root).as_posix(), "size_bytes": st.st_size,
              "sha256": None, "format": suffix[1:] or "unknown", "format_basis": "extension",
              "status": "pending", "tables": []}
    if is_link(st) or st.st_nlink != 1:
        result.update(status="skipped_link", reason="link_or_reparse_or_hardlink")
        return result
    if not stat.S_ISREG(st.st_mode):
        result.update(status="skipped_special", reason="not_regular_file")
        return result
    if st.st_size > args.max_file_bytes or st.st_size > budget[0]:
        result.update(status="skipped_limit", reason="file_or_total_byte_limit")
        return result
    budget[0] -= st.st_size
    try:
        with open_regular(path, st, root) as stream:
            hasher = hashlib.sha256()
            count = 0
            while chunk := stream.read(1024**2):
                count += len(chunk)
                if count > st.st_size:
                    raise ProfileError("changed_during_run")
                hasher.update(chunk)
            if count != st.st_size:
                raise ProfileError("changed_during_run")
            result["sha256"] = hasher.hexdigest()
            stream.seek(0)
            if suffix not in SUPPORTED:
                result.update(status="unsupported", reason="no_safe_adapter")
            else:
                if suffix in (".csv", ".tsv"):
                    tables = profile_csv(stream, suffix, args)
                elif suffix == ".xlsx":
                    tables = profile_xlsx(stream, args)
                else:
                    tables = profile_json(stream, suffix, args)
                result.update(tables=tables, status="ok" if all(t["complete"] for t in tables) else "partial")
                result["format_basis"] = "extension_and_parser"
    except ProfileError as exc:
        result.update(status="error", reason=str(exc), tables=[])
        if str(exc) == "changed_during_run":
            result["sha256"] = None
    except Exception:
        result.update(status="error", reason="read_or_parse_error", tables=[])
    return result


def inventory(root, args):
    files, issues = [], []
    budget = [args.max_total_bytes]
    visited = 0
    stopped = False

    def walk(directory, depth):
        nonlocal visited, stopped
        if depth > args.max_depth:
            issues.append({"relative_path": directory.relative_to(root).as_posix(), "reason": "depth_limit"})
            return
        try:
            if is_link(directory.lstat()):
                raise ProfileError("linked_directory")
            with os.scandir(directory) as entries:
                for entry in entries:
                    if visited >= args.max_entries or len(files) >= args.max_files:
                        issues.append({"reason": "inventory_limit"})
                        stopped = True
                        return
                    visited += 1
                    path = Path(entry.path)
                    try:
                        # Windows DirEntry.stat can report st_nlink/st_ino as 0.
                        # A fresh lstat supplies the identity/link checks we need.
                        st = path.lstat()
                        if stat.S_ISDIR(st.st_mode) and not is_link(st):
                            walk(path, depth + 1)
                        else:
                            files.append(file_profile(path, st, root, args, budget))
                    except OSError:
                        issues.append({"relative_path": path.relative_to(root).as_posix(), "reason": "entry_unreadable"})
                    if stopped:
                        return
        except (OSError, ProfileError):
            issues.append({"relative_path": directory.relative_to(root).as_posix(), "reason": "directory_unreadable_or_linked"})

    walk(root, 0)
    files.sort(key=lambda f: f["relative_path"])
    return files, issues


def run(args):
    source = local_directory(args.input)
    private = local_directory(args.private_dir)
    if source.is_relative_to(private) or private.is_relative_to(source):
        raise ProfileError("overlapping_directories")
    # Owner attests that this directory has appropriate ACLs/encryption and no sync.
    # Windows mode bits do not enforce ACLs; files inherit the owner's chosen ACL.
    run_dir = private / ("profile-" + uuid.uuid4().hex)
    run_dir.mkdir(mode=0o700)
    print("PROFILE_STARTED", flush=True)
    files, issues = inventory(source, args)
    complete = not issues and all(f["status"] == "ok" for f in files)
    report = {"schema_version": "forestcheck.profile.v1", "script_version": VERSION,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "confidential": True, "input_root": str(source),
              "inventory_complete": not issues, "profile_complete": complete,
              "limits": {k: v for k, v in vars(args).items() if k.startswith("max_")},
              "missing_policy": "null_or_absent_or_empty_or_whitespace; zero/NA/NULL/dash are not missing",
              "type_policy": "observed_lexical_or_json_types_not_domain_semantics",
              "files": files, "inventory_issues": issues}
    # Exclusive write; incomplete writes keep a .partial name in this same private run.
    temporary = run_dir / "profile.json.partial"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=True, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(run_dir / "profile.json")
    print("PROFILE_COMPLETE" if complete else "PROFILE_PARTIAL", flush=True)
    return 0 if complete else 4


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["--help"], ["-h"]):
        print("PROFILE_HELP_SEE_DOCS")
        return 0
    try:
        args = arguments(argv)
        # Validate everything before creating a run directory.
        return run(args)
    except ProfileError:
        print("PROFILE_INVALID_ARGUMENTS")
        return 2
    except KeyboardInterrupt:
        print("PROFILE_INTERRUPTED")
        return 130
    except Exception:
        print("PROFILE_FAILED")
        return 3


if __name__ == "__main__":
    sys.exit(main())
