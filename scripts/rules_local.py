"""Offline development checks; paths and record values never go to the console."""

import argparse
from pathlib import Path

from forestcheck.rules.local import run_development


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def main():
    parser = StatusParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    try:
        args = parser.parse_args()
        print("RULES_STARTED", flush=True)
        run_development(args.config, workspace_root=Path(__file__).resolve().parents[1])
    except KeyboardInterrupt:
        print("RULES_INTERRUPTED")
        return 130
    except Exception:
        print("RULES_FAILED")
        return 3
    print("RULES_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
