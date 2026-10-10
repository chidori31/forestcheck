"""Local static build only. No repository browsing, directory listings or request logs."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(404)

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        base = Path(self.directory).resolve()
        if not path.resolve().is_relative_to(base) or path.is_symlink():
            self.send_error(404)
            return
        super().do_GET()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=4173)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1] / "frontend/dist"
    if not (root / "index.html").is_file():
        raise SystemExit("BUILD_REQUIRED")
    ThreadingHTTPServer(
        ("127.0.0.1", a.port), partial(Handler, directory=str(root))
    ).serve_forever()
