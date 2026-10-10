"""Local operator entrypoint. No secrets in arguments or generic log output."""

import argparse
import json
import logging
import time
from pathlib import Path


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid_arguments")


def main():
    parser = StatusParser(add_help=False)
    parser.add_argument("command", choices=("init", "serve", "worker", "register-local", "openapi"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    try:
        args = parser.parse_args()
        from forestcheck.api.app import create_app
        from forestcheck.api.settings import Settings, ordinary_path
        from forestcheck.storage.application import Store

        logging.disable(logging.CRITICAL)
        if args.command == "openapi":
            if args.output is None:
                raise ValueError("output_required")
            path = ordinary_path(args.output.resolve(), missing=True)
            path.write_text(
                json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + "\n", "utf-8"
            )
        else:
            settings = Settings.from_env()
            store = Store(settings)
            if args.command == "init":
                store.initialize()
            elif args.command == "register-local":
                from forestcheck.api.admin import register_forestcheck

                if args.config is None:
                    raise ValueError("config_required")
                register_forestcheck(store, args.config, workspace_root=settings.workspace)
            elif args.command == "serve":
                import uvicorn

                if not store.database.is_file() or not 1024 <= args.port <= 65535:
                    raise ValueError("initialize_and_choose_unprivileged_port")
                uvicorn.run(
                    create_app(settings),
                    host="127.0.0.1",
                    port=args.port,
                    access_log=False,
                    log_config=None,
                )
            else:
                from forestcheck.api.execution import Worker

                worker = Worker(store)
                while True:
                    claimed = worker.run_once()
                    if args.once:
                        break
                    if not claimed:
                        time.sleep(1)
        print("BACKEND_COMPLETE")
        return 0
    except Exception:
        print("BACKEND_FAILED")
        return 3
    except KeyboardInterrupt:
        print("BACKEND_INTERRUPTED")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
