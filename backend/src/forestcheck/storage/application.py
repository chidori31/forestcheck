"""Local persistent workflow. One transaction owns each publication; leases are fenced."""

import hashlib
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from importlib.resources import files
from uuid import uuid4

from forestcheck.api.settings import ordinary_path
from forestcheck.ingest.csv_import import canonical, sha256
from forestcheck.storage.metadata import schema_signature


class WorkflowError(Exception):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value):
    from uuid import UUID

    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise WorkflowError("invalid_id", 422) from None


def statements(connection, sql):
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            connection.execute(buffer)
            buffer = ""
    if buffer.strip():
        raise ValueError("incomplete_migration")


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.root
        self.database = self.root / "application.sqlite3"

    def initialize(self):
        ordinary_path(self.root, directory=True, missing=True)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        ordinary_path(self.database, missing=True)
        sql = (
            files("forestcheck.storage")
            .joinpath("migrations/001_application.sql")
            .read_text("utf-8")
        )
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise WorkflowError("unsupported_workflow_database")
            if version == 0:
                if connection.execute(
                    "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchone():
                    raise WorkflowError("foreign_workflow_database")
                statements(connection, sql)
                connection.execute("PRAGMA user_version=1")
            else:
                with closing(sqlite3.connect(":memory:")) as expected:
                    statements(expected, sql)
                    if schema_signature(connection) != schema_signature(expected):
                        raise WorkflowError("incompatible_workflow_database")
                if connection.execute("SELECT * FROM application_schema").fetchall() != [
                    (1, "forestcheck-workflow")
                ]:
                    raise WorkflowError("foreign_workflow_database")
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
        for name in ("uploads", "ingested", "versions", "exports", "runs", "staging"):
            path = self.root / name
            ordinary_path(path, directory=True, missing=True)
            path.mkdir(exist_ok=True, mode=0o700)

    @contextmanager
    def transaction(self):
        ordinary_path(self.database)
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise WorkflowError("unsupported_workflow_database", 503)
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def one(self, table, item_id, *, connection=None):
        if table not in {
            "datasets",
            "versions",
            "profiles",
            "jobs",
            "findings",
            "suggestions",
            "review_events",
            "artifacts",
        }:
            raise ValueError("internal_table")
        if connection is None:
            with self.transaction() as cx:
                return self.one(table, item_id, connection=cx)
        row = connection.execute(
            f"SELECT * FROM {table} WHERE id=?", (identifier(item_id),)
        ).fetchone()
        if row is None:
            raise WorkflowError("not_found", 404)
        return dict(row)

    def idempotent(self, cx, operator, scope, key, payload, create):
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 128
            or not key.isascii()
            or any(c.isspace() for c in key)
        ):
            raise WorkflowError("idempotency_key_required", 422)
        key_hash, request_hash = digest(key), digest(payload)
        old = cx.execute(
            "SELECT request_sha256,response_json FROM idempotency WHERE operator=? AND scope=? AND key_sha256=?",
            (operator, scope, key_hash),
        ).fetchone()
        if old:
            if old[0] != request_hash:
                raise WorkflowError("idempotency_conflict")
            return json.loads(old[1])
        result = create(cx)
        cx.execute(
            "INSERT INTO idempotency VALUES (?,?,?,?,?)",
            (operator, scope, key_hash, request_hash, canonical(result)),
        )
        return result

    def queue(self, cx, kind, payload, operator):
        job_id = str(uuid4())
        cx.execute(
            "INSERT INTO jobs(id,kind,status,operator,request_json,request_sha256,created_at) VALUES (?,?,'queued',?,?,?,?)",
            (job_id, kind, operator, canonical(payload), digest(payload), time.time()),
        )
        return {"schema_version": "1.0", "id": job_id, "kind": kind, "status": "queued"}

    def enqueue(self, kind, payload, operator, key):
        with self.transaction() as cx:
            return self.idempotent(
                cx, operator, kind, key, payload, lambda cx: self.queue(cx, kind, payload, operator)
            )

    def job(self, job_id):
        row = self.one("jobs", job_id)
        row["request"] = json.loads(row.pop("request_json"))
        row["result"] = json.loads(row.pop("result_json")) if row["result_json"] else None
        row.pop("result_json", None)
        return row

    def list_jobs(self):
        with self.transaction() as cx:
            return {"total": cx.execute("SELECT count(*) FROM jobs").fetchone()[0]}

    def claim(self, *, now=None, lease_seconds=30):
        now = time.time() if now is None else now
        with self.transaction() as cx:
            expired = cx.execute(
                "SELECT id,attempts,max_attempts,cancel_requested FROM jobs WHERE status='running' AND lease_until<=?",
                (now,),
            ).fetchall()
            for job in expired:
                status = "cancelled" if job[3] else "failed" if job[1] >= job[2] else "queued"
                cx.execute(
                    "UPDATE jobs SET status=?,owner=NULL,lease_until=NULL,error_code='worker_interrupted',finished_at=? WHERE id=?",
                    (status, now if status != "queued" else None, job[0]),
                )
            job = cx.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1"
            ).fetchone()
            if not job:
                return None
            owner = str(uuid4())
            cx.execute(
                "UPDATE jobs SET status='running',owner=?,lease_until=?,attempts=attempts+1,started_at=COALESCE(started_at,?),error_code=NULL WHERE id=?",
                (owner, now + lease_seconds, now, job["id"]),
            )
            result = dict(cx.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone())
            result["request"] = json.loads(result.pop("request_json"))
            return result

    def owned(self, cx, claim, now, *, allow_cancel=False):
        job = self.one("jobs", claim["id"], connection=cx)
        if (
            job["owner"] != claim["owner"]
            or job["status"] != "running"
            or job["lease_until"] <= now
        ):
            raise WorkflowError("worker_lease_lost")
        if job["cancel_requested"] and not allow_cancel:
            raise WorkflowError("cancel_requested")
        return job

    def heartbeat(self, claim, *, lease_seconds=30):
        now = time.time()
        with self.transaction() as cx:
            self.owned(cx, claim, now)
            cx.execute(
                "UPDATE jobs SET lease_until=? WHERE id=?", (now + lease_seconds, claim["id"])
            )

    def finish(self, claim, result, *, publish=None, now=None):
        now = time.time() if now is None else now
        with self.transaction() as cx:
            self.owned(cx, claim, now)
            if publish:
                publish(cx)
            cx.execute(
                "UPDATE jobs SET status='succeeded',result_json=?,finished_at=?,owner=NULL,lease_until=NULL WHERE id=?",
                (canonical(result), now, claim["id"]),
            )

    def fail(self, claim, code, *, retryable=False):
        now = time.time()
        with self.transaction() as cx:
            job = self.owned(cx, claim, now, allow_cancel=True)
            status = (
                "cancelled"
                if job["cancel_requested"] or code == "cancelled"
                else "queued"
                if retryable and job["attempts"] < job["max_attempts"]
                else "failed"
            )
            cx.execute(
                "UPDATE jobs SET status=?,error_code=?,finished_at=?,owner=NULL,lease_until=NULL WHERE id=?",
                (status, code, now if status != "queued" else None, claim["id"]),
            )

    def cancel(self, job_id):
        with self.transaction() as cx:
            job = self.one("jobs", job_id, connection=cx)
            if job["status"] == "queued":
                cx.execute(
                    "UPDATE jobs SET status='cancelled',cancel_requested=1,finished_at=? WHERE id=?",
                    (time.time(), job_id),
                )
            elif job["status"] == "running":
                cx.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
        return self.job(job_id)

    def artifact(self, path, media_type, *, connection=None):
        path = ordinary_path(path)
        if not path.is_relative_to(self.root):
            raise WorkflowError("artifact_outside_registry", 422)
        if connection is None:
            with self.transaction() as cx:
                return self.artifact(path, media_type, connection=cx)
        relative = path.relative_to(self.root).as_posix()
        found = connection.execute(
            "SELECT id,sha256 FROM artifacts WHERE relative_path=?", (relative,)
        ).fetchone()
        checksum = sha256(path)
        if found:
            if checksum != found[1]:
                raise WorkflowError("artifact_changed")
            return found[0]
        item_id = str(uuid4())
        connection.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?)",
            (item_id, relative, checksum, path.stat().st_size, media_type, time.time()),
        )
        return item_id

    def resolve(self, artifact_id, *, verify=True):
        item = self.one("artifacts", artifact_id)
        path = self.root / item["relative_path"]
        if path.resolve() != path or not path.is_relative_to(self.root):
            raise WorkflowError("invalid_artifact_registry", 409)
        try:
            ordinary_path(path)
        except ValueError:
            raise WorkflowError("invalid_artifact_registry", 409) from None
        if path.stat().st_size != item["size_bytes"] or (verify and sha256(path) != item["sha256"]):
            raise WorkflowError("artifact_changed")
        return path
