"""Explicit offline recommendations dev run; never applies a patch to source files."""

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID, uuid4


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def main():
    parser = StatusParser(add_help=False)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-id", type=UUID)
    try:
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        from forestcheck.ml.data import LocalData
        from forestcheck.ml.local import directory, write_json
        from forestcheck.repair.local import prepare, run

        data = LocalData(args.config, workspace_root=root)
        print("RECOMMENDATIONS_STARTED", flush=True)
        if args.command == "freeze":
            prepare(args.config, workspace_root=root)
        elif args.worker:
            if args.run_id is None:
                raise ValueError("worker_run_id_required")
            run(args.config, workspace_root=root, run_id=args.run_id)
        else:
            from forestcheck.ml.resources import run_bounded

            prepare(args.config, workspace_root=root)
            run_id = args.run_id or uuid4()
            stage = directory(data.private / "stage7")
            jobs = directory(stage / "jobs")
            job = jobs / str(run_id)
            job.mkdir()
            result = run_bounded(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "evaluate",
                    "--config",
                    str(args.config.resolve()),
                    "--run-id",
                    str(run_id),
                    "--worker",
                ],
                cwd=root,
                log_dir=job,
                wall_seconds=900,
                memory_mib=4096,
                cpu_threads=2,
            )
            write_json(job / "resources.json", result)
            if result["status"] != "success":
                print("RECOMMENDATIONS_FAILED")
                return 3
            write_json(
                job / "run.json",
                {"schema_version": "1.0", "run_id": str(run_id), "patches_applied": 0},
            )
        print("RECOMMENDATIONS_COMPLETE")
        return 0
    except Exception as exc:
        # Diagnostic failure itself must not disclose a traceback or replace older evidence.
        try:
            if "data" in locals() and "args" in locals() and args.worker and args.run_id:
                destination = data.private / "stage7/runs" / str(args.run_id)
                if destination.is_dir() and destination.resolve() == destination:
                    with (destination / "failure.json").open("x", encoding="utf-8") as stream:
                        json.dump(
                            {"type": type(exc).__name__, "message": str(exc)},
                            stream,
                            ensure_ascii=False,
                        )
        except OSError:
            pass
        print("RECOMMENDATIONS_FAILED")
        return 3
    except KeyboardInterrupt:
        print("RECOMMENDATIONS_INTERRUPTED")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
