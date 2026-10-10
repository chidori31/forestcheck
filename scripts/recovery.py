"""Stopped-service backup. Private bundles; never a public distribution format."""

import argparse
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from forestcheck.api.settings import Settings, ordinary_path
from forestcheck.ingest.csv_import import sha256
from forestcheck.storage.application import Store


def new_destination(path, roots):
    path = ordinary_path(Path(path), directory=True, missing=True)
    if path.exists() or any(path.is_relative_to(r) or r.is_relative_to(path) for r in roots):
        raise ValueError("new_separate_destination_required")
    return path


def files(root):
    for p in root.iterdir():
        if p.is_dir():
            ordinary_path(p, directory=True)
            yield from files(p)
        else:
            ordinary_path(p)
            yield p


def copy_verified(source, target):
    ordinary_path(source)
    before = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256(source) != before or sha256(target) != before:
        raise ValueError("file_changed_during_backup")


def backup(settings, destination, *, stopped=False):
    if not stopped:
        raise ValueError("stop_backend_and_worker_first")
    store = Store(settings)
    ordinary_path(store.database)
    destination = new_destination(destination, [store.root, settings.workspace])
    dependencies = []
    with closing(sqlite3.connect(store.database, timeout=5)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        if lock.execute(
            "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
        ).fetchone()[0]:
            raise ValueError("pending_jobs_finish_or_cancel_first")
        # A separate reader uses SQLite's online backup API, including committed WAL.
        destination.mkdir()
        target = destination / "store"
        target.mkdir()
        with (
            closing(sqlite3.connect(store.database)) as reader,
            closing(sqlite3.connect(target / "application.sqlite3")) as output,
        ):
            reader.backup(output)
            if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("sqlite_integrity_failed")
        for source in files(store.root):
            rel = source.relative_to(store.root)
            if rel.as_posix() in {
                "application.sqlite3",
                "application.sqlite3-wal",
                "application.sqlite3-shm",
            }:
                continue
            copy_verified(source, target / rel)
        external = set()
        for row in lock.execute("SELECT content_json FROM profiles"):
            profile = json.loads(row[0])
            for key in ("split_manifest_path", "reference_parquet_path"):
                if profile.get(key):
                    external.add(ordinary_path(Path(profile[key])))
            for model in profile.get("models", []):
                registry = ordinary_path(Path(model["registry"]), directory=True)
                external.update((registry / "registry.json", registry / "registry.jsonl"))
                external.update(files(ordinary_path(registry / model["model_id"], directory=True)))
        for i, source in enumerate(sorted(external)):
            # Only explicit registry dependencies, no arbitrary sibling directories.
            if source.is_relative_to(destination):
                raise ValueError("dependency_overlap")
            rel = f"dependencies/{i:06d}.bin"
            copy_verified(source, destination / rel)
            dependencies.append(
                {"original_path": str(source), "stored_as": rel, "sha256": sha256(source)}
            )
        for row in lock.execute("SELECT relative_path,sha256 FROM artifacts"):
            path = ordinary_path(target / row[0])
            if not path.is_relative_to(target) or sha256(path) != row[1]:
                raise ValueError("artifact_integrity_failed")
        config = {
            "operator": settings.operator,
            "origins": list(settings.origins),
            "data_mode": settings.data_mode,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_json_bytes": settings.max_json_bytes,
            "max_analysis_records": settings.max_analysis_records,
            "token": "EXCLUDED_SET_ENVIRONMENT",
        }
        (destination / "settings.json").write_text(json.dumps(config, indent=2), "utf-8")
        manifest = {
            "schema_version": "1.0",
            "database_version": lock.execute("PRAGMA user_version").fetchone()[0],
            "dependencies": dependencies,
            "files": {p.relative_to(destination).as_posix(): sha256(p) for p in files(destination)},
            "policy": "stopped_service_sqlite_backup_no_secrets_no_in_place_restore",
            "external_path_policy": "profiles retain original bindings; dependency copies are not automatically installed",
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8"
        )
        lock.rollback()
    return manifest


def restore(bundle, destination, *, workspace):
    bundle = ordinary_path(Path(bundle), directory=True)
    destination = new_destination(destination, [bundle, Path(workspace).resolve()])
    manifest = json.loads(ordinary_path(bundle / "manifest.json").read_text("utf-8"))
    if manifest.get("schema_version") != "1.0" or manifest.get("database_version") != 1:
        raise ValueError("unsupported_backup_version")
    listed = manifest["files"]
    actual = {p.relative_to(bundle).as_posix() for p in files(bundle)} - {"manifest.json"}
    if actual != set(listed):
        raise ValueError("backup_file_set_mismatch")
    for name, digest in listed.items():
        path = ordinary_path(bundle / name)
        if not path.is_relative_to(bundle) or sha256(path) != digest:
            raise ValueError("backup_hash_mismatch")
    destination.mkdir()
    for source in files(bundle / "store"):
        copy_verified(source, destination / source.relative_to(bundle / "store"))
    # Keep the external dependency snapshot beside, never overwrite original private paths.
    for name in ("manifest.json", "settings.json"):
        copy_verified(bundle / name, destination / ("recovery-" + name))
    if (bundle / "dependencies").exists():
        for source in files(bundle / "dependencies"):
            copy_verified(source, destination / "recovery-dependencies" / source.name)
    with closing(sqlite3.connect(destination / "application.sqlite3")) as cx:
        if cx.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("restored_integrity_failed")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["backup", "restore"])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--stopped", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "backup":
            backup(Settings.from_env(), args.destination, stopped=args.stopped)
        else:
            restore(args.bundle, args.destination, workspace=Path(__file__).resolve().parents[1])
        print("RECOVERY_COMPLETE")
        return 0
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError):
        print("RECOVERY_FAILED_CHECK_STOPPED_STATE_PATHS_AND_MANIFEST")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
