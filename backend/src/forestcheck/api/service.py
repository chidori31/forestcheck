"""Transactional application operations. Computation is performed by the worker."""

import csv
import json
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

from forestcheck.api.contracts import AnalysisProfile
from forestcheck.api.settings import ordinary_path
from forestcheck.domain.contracts import AnalysisRun, Review
from forestcheck.ingest.csv_import import canonical, inspect_source, sha256, validate_mapping
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.repair.models import DatasetBinding, RecommendationResult
from forestcheck.storage.application import WorkflowError, digest, identifier


def page(items, total, offset, limit):
    return {
        "schema_version": "1.0",
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit if offset + limit < total else None,
    }


def bounds(offset, limit):
    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 200:
        raise WorkflowError("invalid_pagination", 422)


class Service:
    def __init__(self, store):
        self.store = store

    def register_upload(self, path, name, actor, key):
        name = str(name)
        if (
            not name.lower().endswith(".csv")
            or any(c in name for c in "/\\:\x00\r\n")
            or name in {".csv", "..csv"}
            or len(name) > 180
        ):
            raise WorkflowError("csv_filename_required", 415)
        path = ordinary_path(path)
        if path.parent != self.store.root / "staging":
            raise WorkflowError("upload_outside_staging", 422)
        size = path.stat().st_size
        if not 0 < size <= self.store.settings.max_upload_bytes:
            raise WorkflowError("upload_size_limit", 413)
        with path.open("rb") as stream:
            prefix = stream.read(4096)
        if prefix.startswith((b"MZ", b"PK", b"\x7fELF", b"\x1f\x8b")) or b"\x00" in prefix:
            raise WorkflowError("unsupported_upload_content", 415)
        checksum = sha256(path)

        def create(cx):
            destination = self.store.root / "uploads" / (str(uuid4()) + ".csv")
            path.rename(destination)
            artifact_id = self.store.artifact(destination, "text/csv", connection=cx)
            dataset_id = str(uuid4())
            cx.execute(
                "INSERT INTO datasets(id,name,source_artifact_id,source_sha256,state,created_at) VALUES (?,?,?,?,'uploaded',?)",
                (dataset_id, name, artifact_id, checksum, time.time()),
            )
            return self._dataset(self.store.one("datasets", dataset_id, connection=cx))

        with self.store.transaction() as cx:
            result = self.store.idempotent(
                cx, actor, "upload", key, {"name": name, "sha256": checksum}, create
            )
        if path.exists():
            path.unlink()  # Only this request's bounded temporary file.
        return result

    @staticmethod
    def _dataset(row):
        return {
            "schema_version": "1.0",
            **{
                k: row[k]
                for k in ("id", "name", "state", "etag", "current_version_id", "created_at")
            },
        }

    def dataset(self, dataset_id):
        return self._dataset(self.store.one("datasets", dataset_id))

    def versions(self, dataset_id, *, offset=0, limit=50):
        bounds(offset, limit)
        dataset = self.store.one("datasets", dataset_id)
        with self.store.transaction() as cx:
            total = cx.execute(
                "SELECT count(*) FROM versions WHERE dataset_id=?", (dataset["id"],)
            ).fetchone()[0]
            rows = cx.execute(
                "SELECT id,number,parent_id,created_at FROM versions WHERE dataset_id=? ORDER BY number,id LIMIT ? OFFSET ?",
                (dataset["id"], limit, offset),
            ).fetchall()
        return page([dict(r) for r in rows], total, offset, limit)

    def datasets(self, *, offset=0, limit=50):
        bounds(offset, limit)
        with self.store.transaction() as cx:
            total = cx.execute("SELECT count(*) FROM datasets").fetchone()[0]
            rows = cx.execute(
                "SELECT * FROM datasets ORDER BY created_at,id LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return page([self._dataset(dict(r)) for r in rows], total, offset, limit)

    def schema(self, dataset_id, *, encoding=None, delimiter=None):
        dataset = self.store.one("datasets", dataset_id)
        mapping = (
            ImportMapping.model_validate_json(dataset["mapping_json"])
            if dataset["mapping_json"]
            else None
        )
        if not mapping and (encoding is None or delimiter is None):
            raise WorkflowError("explicit_csv_dialect_required", 422)
        encoding = mapping.encoding if mapping else encoding
        delimiter = mapping.delimiter if mapping else delimiter
        if encoding not in {"utf-8", "utf-8-sig", "cp1251"} or delimiter not in {
            ";",
            ",",
            "\t",
            "|",
        }:
            raise WorkflowError("unsupported_csv_dialect", 422)
        source = self.store.resolve(dataset["source_artifact_id"])
        csv.field_size_limit(1024**2)
        try:
            with source.open(encoding=encoding, newline="") as stream:
                headers = next(csv.reader(stream, delimiter=delimiter, strict=True), [])
        except (UnicodeError, csv.Error):
            raise WorkflowError("csv_dialect_invalid", 422) from None
        if not headers or len(headers) > 2048:
            raise WorkflowError("column_limit", 422)
        report = None
        if dataset["current_version_id"]:
            version = self.store.one("versions", dataset["current_version_id"])
            report = json.loads(version["import_report_json"])
        return {
            "schema_version": "1.0",
            "dataset_id": dataset["id"],
            "headers": headers,
            "encoding": encoding,
            "delimiter": delimiter,
            "mapping": mapping.model_dump(mode="json") if mapping else None,
            "import_report": report,
            "units_inferred": False,
            "mapping_sha256": dataset["mapping_sha256"],
        }

    def confirm_mapping(self, dataset_id, request, actor, key):
        dataset = self.store.one("datasets", dataset_id)
        source = self.store.resolve(dataset["source_artifact_id"])
        try:
            validate_mapping(
                inspect_source(
                    source, encoding=request.mapping.encoding, delimiter=request.mapping.delimiter
                ),
                request.mapping,
            )
        except (ValueError, UnicodeError, csv.Error):
            raise WorkflowError("mapping_header_mismatch", 422) from None
        payload = request.model_dump(mode="json")

        def create(cx):
            current = self.store.one("datasets", dataset_id, connection=cx)
            if current["etag"] != request.expected_version:
                raise WorkflowError("version_conflict")
            if current["state"] != "uploaded":
                raise WorkflowError("mapping_already_frozen")
            mapping_json = canonical(payload["mapping"])
            cx.execute(
                "UPDATE datasets SET mapping_json=?,mapping_sha256=?,confirmed_by=?,confirmed_at=?,state='import_queued',etag=etag+1 WHERE id=?",
                (mapping_json, digest(payload["mapping"]), actor, time.time(), dataset_id),
            )
            return self.store.queue(
                cx,
                "import",
                {
                    "dataset_id": dataset_id,
                    "source_sha256": current["source_sha256"],
                    "mapping_sha256": digest(payload["mapping"]),
                },
                actor,
            )

        with self.store.transaction() as cx:
            return self.store.idempotent(cx, actor, "mapping:" + dataset_id, key, payload, create)

    def binding(self, version):
        return DatasetBinding(
            schema_version="1.0",
            dataset_id=version["domain_dataset_id"],
            dataset_revision=version["domain_revision"],
            source_sha256=version["source_sha256"],
            mapping_sha256=version["mapping_sha256"],
            snapshot_sha256=version["snapshot_sha256"],
        )

    def register_profile(self, profile):
        from forestcheck.api.runtime import checked_models

        checked_models(profile)  # Admin-controlled CBM/JSON registry only.
        data = profile.model_dump(mode="json")
        with self.store.transaction() as cx:
            old = cx.execute(
                "SELECT sha256 FROM profiles WHERE id=?", (str(profile.id),)
            ).fetchone()
            if old:
                if old[0] != digest(data):
                    raise WorkflowError("profile_version_conflict")
                return str(profile.id)
            cx.execute(
                "INSERT INTO profiles VALUES (?,?,?,?,?)",
                (str(profile.id), profile.name, canonical(data), digest(data), time.time()),
            )
        return str(profile.id)

    def profile(self, profile_id):
        row = self.store.one("profiles", profile_id)
        data = json.loads(row["content_json"])
        if digest(data) != row["sha256"]:
            raise WorkflowError("profile_integrity_failure")
        return AnalysisProfile.model_validate(data), row["sha256"]

    def profiles(self):
        with self.store.transaction() as cx:
            ids = [r[0] for r in cx.execute("SELECT id FROM profiles ORDER BY name,id")]
        result = []
        for profile_id in ids:
            p, checksum = self.profile(profile_id)
            result.append(
                {
                    "id": str(p.id),
                    "name": p.name,
                    "version": p.version,
                    "sha256": checksum,
                    "mapping_sha256": p.mapping_sha256,
                    "rule_count": len(p.rules),
                    "source_kind": p.source_kind,
                    "cohort_size": len(p.allowed_record_ids) if p.allowed_record_ids else None,
                    "models": [
                        {
                            "id": str(m.model_id),
                            "target": m.target,
                            "metadata_sha256": m.metadata_sha256,
                            "status": "registered_candidate",
                        }
                        for m in p.models
                    ],
                    "limitations": p.limitations,
                }
            )
        return {"schema_version": "1.0", "items": result, "total": len(result)}

    def start_analysis(self, request, actor, key):
        version = self.store.one("versions", request.version_id)
        profile, checksum = self.profile(request.profile_id)
        if version["mapping_sha256"] != profile.mapping_sha256:
            raise WorkflowError("profile_mapping_incompatible", 422)
        if request.use_models and profile.models:
            plan = profile.ml_plan
            if (
                version["snapshot_sha256"] != plan.snapshot_sha256
                or version["source_sha256"] != plan.source_sha256
                or version["domain_dataset_id"] != str(plan.dataset_id)
            ):
                raise WorkflowError("model_snapshot_incompatible", 422)
        payload = {
            **request.model_dump(mode="json"),
            "binding": self.binding(version).model_dump(mode="json"),
            "profile_sha256": checksum,
            "rule_sha256": digest([r.model_dump(mode="json") for r in profile.rules]),
            "model_versions": [
                m.model_dump(mode="json", exclude={"registry"}) for m in profile.models
            ]
            if request.use_models
            else [],
        }
        return self.store.enqueue("analysis", payload, actor, key)

    def job(self, job_id):
        data = self.store.job(job_id)
        for key in ("owner", "lease_until", "request_json", "result_json"):
            data.pop(key, None)
        data["schema_version"] = "1.0"
        data["progress_percent"] = None
        if data["kind"] == "analysis":
            binding = data["request"]["binding"]
            lifecycle = {"queued": "pending", "succeeded": "completed"}.get(
                data["status"], data["status"]
            )

            def stamp(value):
                return datetime.fromtimestamp(value, timezone.utc) if value is not None else None

            # A queued retry has prior attempts, but the legacy v1 pending lifecycle has no start.
            native = AnalysisRun(
                schema_version="1.0",
                id=UUID(data["id"]),
                dataset_id=UUID(binding["dataset_id"]),
                dataset_revision=binding["dataset_revision"],
                status=lifecycle,
                created_at=stamp(data["created_at"]),
                started_at=None if lifecycle == "pending" else stamp(data["started_at"]),
                finished_at=None if lifecycle == "pending" else stamp(data["finished_at"]),
                config_digest=data["request_sha256"],
                model_version_ids=tuple(
                    UUID(m["model_id"]) for m in data["request"]["model_versions"]
                ),
            )
            data["analysis_run"] = native.model_dump(mode="json")
        return data

    def jobs(self, *, kind=None, status=None, offset=0, limit=50):
        bounds(offset, limit)
        if kind not in {None, "analysis", "import", "version", "export"} or status not in {
            None,
            "queued",
            "running",
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise WorkflowError("invalid_filter", 422)
        clauses, args = [], []
        for column, value in (("kind", kind), ("status", status)):
            if value is not None:
                clauses.append(column + "=?")
                args.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.store.transaction() as cx:
            total = cx.execute("SELECT count(*) FROM jobs" + where, args).fetchone()[0]
            ids = [
                r[0]
                for r in cx.execute(
                    "SELECT id FROM jobs" + where + " ORDER BY created_at,id LIMIT ? OFFSET ?",
                    (*args, limit, offset),
                )
            ]
        return page([self.job(i) for i in ids], total, offset, limit)

    def findings(
        self,
        analysis_id,
        *,
        severity=None,
        field_name=None,
        reviewed=None,
        offset=0,
        limit=50,
        search=None,
        sort="priority",
        descending=False,
    ):
        bounds(offset, limit)
        analysis_id = identifier(analysis_id)
        clauses, args = ["analysis_id=?"], [analysis_id]
        if severity is not None:
            if severity not in {"info", "warning", "error", "critical"}:
                raise WorkflowError("invalid_filter", 422)
            clauses.append("severity=?")
            args.append(severity)
        if field_name is not None:
            clauses.append("field_name=?")
            args.append(field_name)
        if reviewed is not None:
            clauses.append("review_version>0" if reviewed else "review_version=0")
        columns = {
            "priority": "priority",
            "field": "field_name",
            "record": "record_id",
            "signal": "signal_kind",
        }
        if sort not in columns or search is not None and len(search) > 200:
            raise WorkflowError("invalid_filter", 422)
        if search:
            clauses.append("(instr(record_id,?)>0 OR instr(field_name,?)>0)")
            args.extend((search, search))
        where = " AND ".join(clauses)
        with self.store.transaction() as cx:
            total = cx.execute("SELECT count(*) FROM findings WHERE " + where, args).fetchone()[0]
            rows = cx.execute(
                "SELECT * FROM findings WHERE "
                + where
                + " ORDER BY "
                + columns[sort]
                + (" DESC" if descending else " ASC")
                + ",id LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
        return page(
            [
                {
                    "finding": json.loads(r["content_json"]),
                    "review_version": r["review_version"],
                    "latest_review": self.latest_review(r["id"]),
                }
                for r in rows
            ],
            total,
            offset,
            limit,
        )

    def finding(self, finding_id):
        row = self.store.one("findings", finding_id)
        with self.store.transaction() as cx:
            suggestions = [
                json.loads(r[0])
                for r in cx.execute(
                    "SELECT content_json FROM suggestions WHERE finding_id=? ORDER BY id",
                    (row["id"],),
                )
            ]
        return {
            "schema_version": "1.0",
            "finding": json.loads(row["content_json"]),
            "review_version": row["review_version"],
            "suggestions": suggestions,
            "latest_review": self.latest_review(finding_id),
        }

    def latest_review(self, finding_id):
        with self.store.transaction() as cx:
            row = cx.execute(
                "SELECT content_json FROM review_events WHERE finding_id=? ORDER BY sequence DESC LIMIT 1",
                (identifier(finding_id),),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def record_query(self, version_id, ids):
        from forestcheck.api.execution import read_records

        version = self.store.one("versions", version_id)
        wanted = set(ids)
        try:
            records = list(read_records(self.store.resolve(version["artifact_id"]), wanted))
        except WorkflowError as error:
            if error.code == "profile_cohort_not_in_dataset":
                raise WorkflowError("not_found", 404) from None
            raise
        records.sort(key=lambda r: str(r.id))
        return page([r.model_dump(mode="json") for r in records], len(records), 0, len(ids))

    def review(self, request, actor, key):
        payload = request.model_dump(mode="json")

        def create(cx):
            finding = self.store.one("findings", request.finding_id, connection=cx)
            if finding["review_version"] != request.expected_version:
                raise WorkflowError("version_conflict")
            job = self.store.one("jobs", finding["analysis_id"], connection=cx)
            if job["status"] != "succeeded":
                raise WorkflowError("analysis_not_complete")
            previous = cx.execute(
                "SELECT id,content_json FROM review_events WHERE finding_id=? ORDER BY sequence DESC LIMIT 1",
                (finding["id"],),
            ).fetchone()
            if request.action == "revoke" and (
                not previous or previous[0] != str(request.revoke_event_id)
            ):
                raise WorkflowError("revocation_version_conflict")
            recommendation = None
            if request.suggestion_id:
                row = self.store.one("suggestions", request.suggestion_id, connection=cx)
                if row["finding_id"] != finding["id"]:
                    raise WorkflowError("suggestion_finding_mismatch", 422)
                recommendation = RecommendationResult.model_validate_json(row["content_json"])
            if request.suggestion_decision == "accepted" and (
                not recommendation or recommendation.patch is None
            ):
                raise WorkflowError("suggestion_has_no_patch", 422)
            native = Review(
                schema_version="1.0",
                id=uuid4(),
                finding_id=UUID(finding["id"]),
                finding_revision=json.loads(finding["content_json"])["revision"],
                reviewer_id=actor,
                reviewed_at=datetime.now(timezone.utc),
                error_assessment=request.error_assessment,
                suggestion_id=request.suggestion_id,
                suggestion_decision=request.suggestion_decision,
                note=request.comment if request.comment else None,
            )
            event = {
                "schema_version": "1.0",
                "id": str(native.id),
                "finding_id": finding["id"],
                "sequence": request.expected_version + 1,
                "operator": actor,
                "action": request.action,
                "revoke_event_id": str(request.revoke_event_id)
                if request.revoke_event_id
                else None,
                "review": native.model_dump(mode="json"),
                "before": recommendation.suggestion.before.model_dump(mode="json")
                if recommendation and recommendation.suggestion.before
                else None,
                "after": recommendation.patch.operation.after.model_dump(mode="json")
                if recommendation and request.suggestion_decision == "accepted"
                else None,
                "analysis_id": finding["analysis_id"],
                "version_id": json.loads(job["request_json"])["version_id"],
            }
            cx.execute(
                "INSERT INTO review_events VALUES (?,?,?,?,?,?)",
                (
                    event["id"],
                    finding["id"],
                    event["sequence"],
                    actor,
                    time.time(),
                    canonical(event),
                ),
            )
            cx.execute(
                "UPDATE findings SET review_version=? WHERE id=?",
                (event["sequence"], finding["id"]),
            )
            return event

        with self.store.transaction() as cx:
            return self.store.idempotent(
                cx, actor, "review:" + str(request.finding_id), key, payload, create
            )

    def reviews(self, finding_id, *, offset=0, limit=50):
        bounds(offset, limit)
        finding_id = identifier(finding_id)
        with self.store.transaction() as cx:
            total = cx.execute(
                "SELECT count(*) FROM review_events WHERE finding_id=?", (finding_id,)
            ).fetchone()[0]
            rows = cx.execute(
                "SELECT content_json FROM review_events WHERE finding_id=? ORDER BY sequence,id LIMIT ? OFFSET ?",
                (finding_id, limit, offset),
            ).fetchall()
        return page([json.loads(r[0]) for r in rows], total, offset, limit)

    def check_version_request(self, cx, dataset_id, payload):
        dataset = self.store.one("datasets", dataset_id, connection=cx)
        if (
            dataset["etag"] != payload["expected_version"]
            or dataset["current_version_id"] != payload["base_version_id"]
        ):
            raise WorkflowError("version_conflict")
        analysis = self.store.one("jobs", payload["analysis_id"], connection=cx)
        if (
            analysis["kind"] != "analysis"
            or analysis["status"] != "succeeded"
            or json.loads(analysis["request_json"])["version_id"] != payload["base_version_id"]
        ):
            raise WorkflowError("analysis_version_mismatch")
        patches = []
        for event_id in payload["review_event_ids"]:
            event_row = self.store.one("review_events", event_id, connection=cx)
            event = json.loads(event_row["content_json"])
            finding = self.store.one("findings", event["finding_id"], connection=cx)
            if (
                event_row["sequence"] != finding["review_version"]
                or event["analysis_id"] != analysis["id"]
            ):
                raise WorkflowError("review_no_longer_current")
            if (
                event["action"] != "decide"
                or event["review"]["error_assessment"] != "confirmed_error"
                or event["review"]["suggestion_decision"] != "accepted"
            ):
                raise WorkflowError("accepted_review_required")
            suggestion = self.store.one(
                "suggestions", event["review"]["suggestion_id"], connection=cx
            )
            result = RecommendationResult.model_validate_json(suggestion["content_json"])
            if not result.patch:
                raise WorkflowError("patch_required")
            patches.append(result.patch)
        fields = [(p.record_id, p.operation.field_name) for p in patches]
        if len(set(fields)) != len(fields):
            raise WorkflowError("overlapping_patch_operations")
        return patches

    def create_version(self, dataset_id, request, actor, key):
        dataset_id = identifier(dataset_id)
        payload = {**request.model_dump(mode="json"), "dataset_id": dataset_id}

        def create(cx):
            self.check_version_request(cx, dataset_id, payload)
            return self.store.queue(cx, "version", payload, actor)

        with self.store.transaction() as cx:
            return self.store.idempotent(cx, actor, "version:" + dataset_id, key, payload, create)

    def records(self, version_id, *, offset=0, limit=50):
        import pyarrow.parquet as pq

        from forestcheck.api.execution import read_records

        bounds(offset, limit)
        version = self.store.one("versions", version_id)
        path = self.store.resolve(version["artifact_id"])
        total = pq.ParquetFile(path).metadata.num_rows
        records = []
        for index, record in enumerate(read_records(path)):
            if index < offset:
                continue
            records.append(record.model_dump(mode="json"))
            if len(records) >= limit:
                break
        return page(records, total, offset, limit)

    def export(self, request, actor, key):
        version = self.store.one("versions", request.version_id)
        if request.kind == "corrected" and version["parent_id"] is None:
            raise WorkflowError("corrected_version_required", 422)
        if request.kind == "original" and version["parent_id"] is not None:
            raise WorkflowError("original_version_required", 422)
        if request.kind == "report":
            if not request.analysis_id:
                raise WorkflowError("analysis_required", 422)
            analysis = self.store.job(request.analysis_id)
            if (
                analysis["status"] != "succeeded"
                or analysis["kind"] != "analysis"
                or analysis["request"]["version_id"] != str(request.version_id)
            ):
                raise WorkflowError("analysis_version_mismatch")
        return self.store.enqueue("export", request.model_dump(mode="json"), actor, key)

    def download(self, export_id):
        job = self.store.job(export_id)
        if job["kind"] != "export" or job["status"] != "succeeded":
            raise WorkflowError("export_not_ready")
        artifact_id = job["result"]["artifact_id"]
        return (
            self.store.resolve(artifact_id),
            self.store.one("artifacts", artifact_id)["media_type"],
            job["request"]["format"],
        )
