"""Generate only invented artifacts in a new explicitly chosen directory."""

import argparse
from pathlib import Path

from forestcheck.evaluation.smoke import run_synthetic_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_synthetic_smoke(args.output_dir)
    except Exception:
        print("SYNTHETIC_SMOKE_FAILED")
        return 3
    print("SYNTHETIC_SMOKE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
