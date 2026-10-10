"""Explicit local train/evaluate/predict; console contains fixed statuses only."""

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
    parser.add_argument("command", choices=("train", "evaluate", "predict", "promote", "rollback"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=UUID)
    parser.add_argument("--role", choices=("tune", "calibration_check"), default="tune")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--model-id", type=UUID)
    parser.add_argument("--expected-current", type=UUID)
    parser.add_argument("--reason")
    parser.add_argument("--confirm-manual-promotion", action="store_true")
    job = None
    try:
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        from forestcheck.evaluation.local import safe_directory
        from forestcheck.ingest.csv_import import checked_private, regular_output
        from forestcheck.ml.local import directory, write_json

        if not regular_output(args.config) or args.config.stat().st_size > 32 * 1024**2:
            raise ValueError("config_type_or_size")
        config = json.loads(args.config.read_text("utf-8-sig"))
        private = checked_private(
            Path(config["private_dir"]), Path(config["source_dir"]).resolve() / "boundary.csv", root
        )
        safe_directory(private / "stage6")
        print("ML_STARTED", flush=True)
        if args.command == "train":
            run_id = args.run_id or uuid4()
            if args.worker:
                from forestcheck.ml.local import train_run

                train_run(args.config, workspace_root=root, run_id=run_id)
            else:
                from forestcheck.ml.resources import run_bounded

                jobs = directory(private / "stage6/jobs")
                job = jobs / str(run_id)
                job.mkdir()
                result = run_bounded(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "train",
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
                    print("ML_FAILED")
                    return 3
                write_json(
                    job / "run.json",
                    {
                        "schema_version": "1.0",
                        "run_id": str(run_id),
                        "report": "stage6/runs/" + str(run_id) + "/report.json",
                        "promoted": False,
                    },
                )
        elif args.command in {"evaluate", "predict"}:
            if args.run_id is None:
                raise ValueError("run_id_required")
            from forestcheck.ml.local import evaluate_run, predict_run

            operation = evaluate_run if args.command == "evaluate" else predict_run
            operation(args.config, workspace_root=root, run_id=args.run_id, role=args.role)
        else:
            from forestcheck.ml.artifacts import promote, rollback
            from forestcheck.ml.data import LocalData
            from forestcheck.ml.local import load_model_for_plan

            data = LocalData(args.config, workspace_root=root)
            target = next((t for t in data.plan.targets if t.name == args.target), None)
            if (
                not target
                or args.model_id is None
                or not args.reason
                or not args.confirm_manual_promotion
            ):
                raise ValueError("explicit_manual_promotion_arguments_required")
            operation = promote if args.command == "promote" else rollback
            load_model_for_plan(
                private / "stage6/models" / target.name, args.model_id, data.plan, target
            )
            operation(
                private / "stage6/models" / target.name,
                args.model_id,
                expected_current=args.expected_current,
                manual_confirmation=True,
                reason=args.reason,
            )
        print("ML_COMPLETE")
        return 0
    except KeyboardInterrupt:
        print("ML_INTERRUPTED")
        return 130
    except Exception as exc:
        # Private diagnostic only if this is our already validated run directory.
        try:
            if (
                "args" in locals()
                and args.worker
                and args.run_id is not None
                and "private" in locals()
            ):
                destination = private / "stage6/runs" / str(args.run_id)
                if destination.is_dir() and destination.resolve() == destination:
                    with (destination / "failure.json").open("x", encoding="utf-8") as stream:
                        json.dump(
                            {"error_type": type(exc).__name__, "message": str(exc)},
                            stream,
                            ensure_ascii=False,
                        )
        except OSError:
            # Preserve previous diagnostics; a failed write must not expose a traceback.
            pass
        print("ML_FAILED")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
