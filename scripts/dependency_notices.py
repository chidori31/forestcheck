"""Installed dependency versions and upstream license files, not a project license."""

import importlib.metadata
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "third_party"
OUT.mkdir(exist_ok=True)
rows = []
for d in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
    name = d.metadata["Name"]
    if name.lower() in {"forestcheck", "uv"}:
        continue
    notices = []
    for f in d.files or ():
        rel = str(f).replace("\\", "/")
        if ".dist-info/" in rel and re.search(
            r"(^|/)(licenses?/|licen[sc]e|copying|notice)", rel, re.I
        ):
            path = Path(d.locate_file(f))
            if path.is_file():
                suffix = rel.split(".dist-info/", 1)[1]
                target = OUT / "python" / name / suffix
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                notices.append(target.relative_to(ROOT).as_posix())
    rows.append(
        {
            "ecosystem": "python",
            "name": name,
            "version": d.version,
            "license": d.metadata.get("License-Expression")
            or d.metadata.get("License")
            or "UNSPECIFIED_METADATA",
            "notices": notices,
        }
    )
lock = json.loads((ROOT / "frontend/package-lock.json").read_text("utf-8"))
for key, meta in lock["packages"].items():
    if not key:
        continue
    directory = ROOT / "frontend" / key
    if not directory.is_dir():
        continue
    name = key.split("node_modules/")[-1]
    notices = []
    for p in directory.iterdir():
        if p.is_file() and re.match(r"^(licen[sc]e|copying|notice)([.-]|$)", p.name, re.I):
            target = OUT / "npm" / name / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
            notices.append(target.relative_to(ROOT).as_posix())
    rows.append(
        {
            "ecosystem": "npm",
            "name": name,
            "version": meta.get("version"),
            "license": meta.get("license", "UNSPECIFIED_METADATA"),
            "notices": notices,
        }
    )
(OUT / "dependencies.json").write_text(
    json.dumps(
        {
            "schema_version": "1.0",
            "scope": "installed Windows dependencies; uninstalled optional platforms remain in lockfiles",
            "packages": rows,
        },
        ensure_ascii=False,
        indent=2,
    ),
    "utf-8",
)
(OUT / "README.md").write_text(
    "# Third-party notices\n\nInstalled Windows dependency versions and original license/notice files are listed in dependencies.json. uv.lock and frontend/package-lock.json include remaining platform-specific resolutions. An empty notices list means no matching installed top-level notice was found, not permission to omit attribution. No license for ForestCheck itself is granted here; the owner decides distribution terms. Dependencies are installed separately; binary runtimes and third-party bundles are not redistributed in the code-demo archive.\n",
    "utf-8",
)
print("DEPENDENCIES", len(rows), "MISSING_NOTICES", sum(not r["notices"] for r in rows))
