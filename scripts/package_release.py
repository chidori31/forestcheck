"""Deterministic code/demo allowlist. Never reads local data configuration."""

import argparse
import hashlib
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entries():
    names = {
        "README.md",
        "release_notes.md",
        ".gitignore",
        ".dockerignore",
        ".env.example",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
        "configs/example.toml",
        "configs/ingest.example.json",
        "configs/experiments.cpu.json",
        "configs/ml.example.json",
        "configs/backend.env.example",
        "configs/xlsx.example.json",
        "configs/xlsx.mapping.synthetic.json",
        "configs/expert_review.example.json",
        "docs/recovery.md",
        "docs/openapi.json",
        "docs/xlsx_import.md",
        "docs/expert_clarifications.md",
        "docs/interim_demo.md",
        "docs/engineering_status.md",
        "docs/growth_reference_candidates.md",
        "docs/expert_reference_message.md",
        "backend/src/forestcheck/ingest/README.md",
        "backend/src/forestcheck/growth/README.md",
        "backend/src/forestcheck/ml/README.md",
        "backend/src/forestcheck/repair/README.md",
        "backend/src/forestcheck/rules/README.md",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/tsconfig.json",
        "frontend/index.html",
        "frontend/.node-version",
        "frontend/public/demo/synthetic.csv",
        "frontend/public/demo/mapping.json",
    }
    for folder, suffixes in (
        ("backend/src", {".py", ".sql"}),
        ("scripts", {".py", ".mjs"}),
        ("tests", {".py"}),
        ("frontend/src", {".ts", ".tsx", ".css"}),
        ("frontend/e2e", {".ts"}),
    ):
        names.update(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in suffixes
        )
    names.update(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "frontend").iterdir()
        if p.suffix in {".ts", ".js"}
    )
    names.update(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "third_party").rglob("*") if p.is_file()
    )
    if (ROOT / "release_checks.json").is_file():
        names.add("release_checks.json")
    result = {}
    for name in sorted(names):
        p = ROOT / name
        if p.is_symlink() or ".." in Path(name).parts:
            raise ValueError("unsafe_release_entry")
        data = p.read_bytes()
        if (b"ForestCheck" + b"-private") in data or re.search(
            rb"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+[\\/]", data
        ):
            raise ValueError("personal_path_in_release:" + name)
        result[name] = data
    return result


def build(destination):
    content = entries()
    manifest = "".join(
        hashlib.sha256(data).hexdigest() + "  " + name + "\n" for name, data in content.items()
    )
    content["MANIFEST.sha256"] = manifest.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(content)
        assert all(archive.read(n) == b for n, b in content.items())
    print(
        "CODE_DEMO_VERIFIED",
        len(content),
        "FILES",
        hashlib.sha256(destination.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
