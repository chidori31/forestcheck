"""Apply metadata schema explicitly; never called on import or health checks."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path


def schema_signature(connection: sqlite3.Connection) -> set[tuple[str, str, str]]:
    return {
        (kind, name, " ".join(sql.split()))
        for kind, name, sql in connection.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }


def apply_statements(connection: sqlite3.Connection, sql: str) -> None:
    # Only the trusted shipped migration, not arbitrary uploaded SQL.
    for statement in sql.split(";"):
        if statement.strip():
            connection.execute(statement)


def migrate_metadata(database: Path, *, workspace_root: Path) -> None:
    database = Path(database)
    if not database.is_absolute() or str(database).startswith(("\\\\", "//")):
        raise ValueError("absolute_local_database_path_required")
    resolved = database.resolve()
    workspace = workspace_root.resolve()
    if resolved.is_relative_to(workspace) or database.suffix != ".sqlite3":
        raise ValueError("database_must_be_outside_workspace")
    if not resolved.parent.is_dir():
        raise ValueError("private_directory_must_exist")
    if resolved.exists() and resolved.stat().st_nlink != 1:
        raise ValueError("linked_database_forbidden")
    sql = files("forestcheck.storage").joinpath("migrations/001_metadata.sql").read_text("utf-8")
    with closing(sqlite3.connect(resolved)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError("unsupported_database_version")
        if version == 1:
            with closing(sqlite3.connect(":memory:")) as expected:
                apply_statements(expected, sql)
                if schema_signature(connection) != schema_signature(expected):
                    raise ValueError("incompatible_database_schema")
            history = connection.execute("SELECT version FROM schema_migrations").fetchall()
            if history != [(1,)]:
                raise ValueError("incompatible_migration_history")
            return
        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if existing:
            raise ValueError("unversioned_nonempty_database")
        apply_statements(connection, sql)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        connection.execute("PRAGMA user_version=1")
