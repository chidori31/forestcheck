"""Local-only batch import; config, mappings and outputs remain private."""
import argparse
import json
from pathlib import Path

from forestcheck.ingest.csv_import import normalize_dataset
from forestcheck.ingest.mapping import ImportMapping


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def run_batch(config_path: Path, *, workspace_root: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if set(config) != {"source_dir", "private_dir", "imports"}:
        raise ValueError("invalid_config")
    root, private = Path(config["source_dir"]), Path(config["private_dir"])
    if not isinstance(config["imports"], list) or not config["imports"]:
        raise ValueError("empty_import_list")
    for item in config["imports"]:
        if set(item) != {"source_file", "mapping_file"}:
            raise ValueError("invalid_import_entry")
        name = item["source_file"]
        if not isinstance(name, str) or not name or any(c in name for c in "/\\:") or name in (".", ".."):
            raise ValueError("source_basename_required")
        mapping_path = Path(item["mapping_file"])
        if not mapping_path.is_absolute() or not mapping_path.resolve().is_relative_to(private.resolve()):
            raise ValueError("mapping_must_be_private")
        mapping = ImportMapping.model_validate_json(mapping_path.read_text(encoding="utf-8"))
        normalize_dataset(root / name, mapping, private, workspace_root=workspace_root)


def main() -> int:
    parser = StatusParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    try:
        args = parser.parse_args()
        print("IMPORT_STARTED", flush=True)
        run_batch(args.config, workspace_root=Path(__file__).resolve().parents[1])
    except KeyboardInterrupt:
        print("IMPORT_INTERRUPTED")
        return 130
    except Exception:
        # No raw data, schema, config paths or exception payloads on the console.
        print("IMPORT_FAILED")
        return 3
    print("IMPORT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
