"""Freeze local evaluation partitions; console emits status only."""

import argparse
from pathlib import Path

from forestcheck.evaluation.local import prepare_evaluation


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def main():
    parser = StatusParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    try:
        args = parser.parse_args()
        print("EVALUATION_STARTED", flush=True)
        prepare_evaluation(args.config, workspace_root=Path(__file__).resolve().parents[1])
    except KeyboardInterrupt:
        print("EVALUATION_INTERRUPTED")
        return 130
    except Exception:
        print("EVALUATION_FAILED")
        return 3
    print("EVALUATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
