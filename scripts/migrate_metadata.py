"""Owner-operated metadata setup. Never run on real storage in an agent session."""
import argparse
import sqlite3
from pathlib import Path

from forestcheck.storage.metadata import migrate_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize external private SQLite metadata")
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()
    try:
        migrate_metadata(args.database, workspace_root=Path(__file__).resolve().parents[1])
    except (ValueError, OSError, sqlite3.Error):
        print("METADATA_SETUP_FAILED")
        return 2
    print("METADATA_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
