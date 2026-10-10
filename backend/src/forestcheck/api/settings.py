"""Explicit local operator configuration. Never serialized into API responses."""

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def ordinary_path(path: Path, *, directory=False, missing=False) -> Path:
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
        raise ValueError("absolute_local_path_required")
    for part in (path, *path.parents):
        if not part.exists() and not part.is_symlink():
            if missing:
                continue
            raise ValueError("path_missing")
        value = part.lstat()
        if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
            raise ValueError("links_forbidden")
    if path.exists():
        if directory and not path.is_dir():
            raise ValueError("directory_required")
        if not directory and (not path.is_file() or path.stat().st_nlink != 1):
            raise ValueError("ordinary_file_required")
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    root: Path
    workspace: Path
    token: str = field(repr=False)
    operator: str = "local-operator"
    origins: tuple[str, ...] = ()
    max_upload_bytes: int = 64 * 1024**2
    max_json_bytes: int = 2 * 1024**2
    max_analysis_records: int = 50_000
    data_mode: str = "real"

    def __post_init__(self):
        root = ordinary_path(Path(self.root), directory=True, missing=True)
        workspace = Path(self.workspace).resolve()
        if root.is_relative_to(workspace) or workspace.is_relative_to(root):
            raise ValueError("private_root_must_be_separate")
        if len(self.token) < 32 or not self.token.isascii() or any(c.isspace() for c in self.token):
            raise ValueError("strong_operator_token_required")
        if not self.operator or len(self.operator) > 128:
            raise ValueError("operator_id_required")
        if min(self.max_upload_bytes, self.max_json_bytes, self.max_analysis_records) <= 0:
            raise ValueError("positive_limits_required")
        if self.data_mode not in {"real", "synthetic"}:
            raise ValueError("invalid_data_mode")
        for origin in self.origins:
            url = urlsplit(origin)
            if (
                url.scheme not in {"http", "https"}
                or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.username
                or url.password
                or url.path
                or url.query
                or url.fragment
            ):
                raise ValueError("exact_loopback_origins_only")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "workspace", workspace)

    @classmethod
    def from_env(cls):
        return cls(
            root=Path(os.environ["FORESTCHECK_BACKEND_ROOT"]),
            workspace=Path(__file__).resolve().parents[4],
            token=os.environ["FORESTCHECK_OPERATOR_TOKEN"],
            operator=os.environ.get("FORESTCHECK_OPERATOR_ID", "local-operator"),
            origins=tuple(
                filter(None, os.environ.get("FORESTCHECK_ALLOWED_ORIGINS", "").split(","))
            ),
            data_mode=os.environ.get("FORESTCHECK_DATA_MODE", "real"),
        )
