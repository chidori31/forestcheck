"""Single persistent worker; no heavy in-process HTTP background tasks."""

import json
import sqlite3
import threading
import time
from collections import defaultdict
from uuid import uuid4

from forestcheck.api.runtime import analyse, context_for
from forestcheck.api.service import Service
from forestcheck.domain.contracts import Record
from forestcheck.ingest.csv_import import canonical, normalize_dataset, sha256
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.repair.engine import _aggregate, verify_patch
from forestcheck.rules.engine import RuleEngine
from forestcheck.rules.models import UniqueKeyCheck
from forestcheck.storage.application import WorkflowError, digest


def read_records(path, ids=None):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    allowed = pa.array([str(i) for i in ids], type=pa.string()) if ids is not None else None
    seen = set()
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["record_id", "record_json"], batch_size=512
    ):
        if allowed is not None:
            batch = batch.filter(pc.is_in(batch.column(0), value_set=allowed))
        for item in batch.to_pylist():
            record = Record.model_validate_json(item["record_json"])
            if str(record.id) != item["record_id"] or record.id in seen:
                raise WorkflowError("record_identity_invalid")
            seen.add(record.id)
            yield record
    if ids is not None and seen != set(ids):
        raise WorkflowError("profile_cohort_not_in_dataset")


def write_records(path, records):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("record_id", pa.string()), ("record_json", pa.string())])
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        batch = []
        for record in records:
            batch.append({"record_id": str(record.id), "record_json": record.model_dump_json()})
            if len(batch) == 512:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))


class Worker:
    def __init__(self, store):
        self.store = store
        self.service = Service(store)

    def run_once(self):
        claim = self.store.claim()
        if claim is None:
            return False
        stop, lost = threading.Event(), threading.Event()

        def beat():
            while not stop.wait(2):
                try:
                    self.store.heartbeat(claim)
                except Exception:
                    lost.set()
                    return

        heartbeat = threading.Thread(target=beat, daemon=True)
        heartbeat.start()

        def checkpoint():
            if lost.is_set():
                raise WorkflowError("cancelled_or_lease_lost")
            self.store.heartbeat(claim)

        try:
            result, publish = getattr(self, "_" + claim["kind"])(claim, checkpoint)
            checkpoint()
            self.store.finish(claim, result, publish=publish)
        except Exception as error:
            code = (
                error.code
                if isinstance(error, WorkflowError)
                else "invalid_workflow_input"
                if isinstance(error, (ValueError, KeyError))
                else "local_io_failure"
                if isinstance(error, (OSError, sqlite3.OperationalError))
                else "worker_failure"
            )
            try:
                self.store.fail(
                    claim, code, retryable=isinstance(error, (OSError, sqlite3.OperationalError))
                )
            except WorkflowError:
                pass  # The new lease owner controls publication and terminal status.
        finally:
            stop.set()
            heartbeat.join(timeout=3)
        return True

    def _import(self, claim, checkpoint):
        request = claim["request"]
        dataset = self.store.one("datasets", request["dataset_id"])
        mapping = ImportMapping.model_validate_json(dataset["mapping_json"])
        if (
            dataset["mapping_sha256"] != request["mapping_sha256"]
            or digest(mapping.model_dump(mode="json")) != request["mapping_sha256"]
        ):
            raise WorkflowError("mapping_integrity_failure")
        source = self.store.resolve(dataset["source_artifact_id"])
        if sha256(source) != request["source_sha256"]:
            raise WorkflowError("source_integrity_failure")
        folder = normalize_dataset(
            source,
            mapping,
            self.store.root / "ingested",
            workspace_root=self.store.settings.workspace,
        )
        manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
        report = json.loads((folder / "report.json").read_text("utf-8"))
        checkpoint()
        version_id = str(uuid4())
        artifact_id = self.store.artifact(
            folder / "normalized.parquet", "application/vnd.apache.parquet"
        )

        def publish(cx):
            current = self.store.one("datasets", dataset["id"], connection=cx)
            if (
                current["current_version_id"]
                or current["mapping_sha256"] != request["mapping_sha256"]
            ):
                raise WorkflowError("dataset_version_conflict")
            cx.execute(
                "INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    version_id,
                    dataset["id"],
                    1,
                    manifest["dataset_id"],
                    1,
                    artifact_id,
                    manifest["output_hashes"]["normalized.parquet"],
                    manifest["mapping_sha256"],
                    manifest["source_sha256"],
                    None,
                    canonical(report),
                    time.time(),
                ),
            )
            cx.execute(
                "UPDATE datasets SET current_version_id=?,state='ready',etag=etag+1 WHERE id=?",
                (version_id, dataset["id"]),
            )

        return {
            "version_id": version_id,
            "dataset_id": dataset["id"],
            "import_report": report,
        }, publish

    def _analysis(self, claim, checkpoint):
        request = claim["request"]
        version = self.store.one("versions", request["version_id"])
        profile, checksum = self.service.profile(request["profile_id"])
        if (
            checksum != request["profile_sha256"]
            or self.service.binding(version).model_dump(mode="json") != request["binding"]
        ):
            raise WorkflowError("analysis_binding_changed")
        artifact = self.store.resolve(version["artifact_id"])
        selected = (
            set(profile.allowed_record_ids) if profile.allowed_record_ids is not None else None
        )
        records = []
        for record in read_records(artifact, selected):
            records.append(record)
            if len(records) > self.store.settings.max_analysis_records:
                raise WorkflowError("analysis_record_limit")
        if not records:
            raise WorkflowError("no_analysable_records")
        findings, suggestions, report = analyse(
            records,
            profile,
            self.service.binding(version),
            claim["id"],
            use_models=request["use_models"],
            checkpoint=checkpoint,
        )
        report["scope"] = "registered_cohort" if selected is not None else "whole_imported_version"
        report["mapping_blockers"] = ImportMapping.model_validate_json(
            self.store.one("datasets", version["dataset_id"])["mapping_json"]
        ).analysis_blockers
        report["coverage"] = "partial_registered_checks"
        report["domain_analysis_complete"] = False
        self.store.resolve(version["artifact_id"])
        if self.service.profile(profile.id)[1] != checksum:
            raise WorkflowError("profile_changed_during_analysis")

        def publish(cx):
            for f in findings:
                cx.execute(
                    "INSERT INTO findings(id,analysis_id,record_id,field_name,severity,priority,signal_kind,content_json) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        str(f.id),
                        claim["id"],
                        str(f.record_id),
                        f.field_name,
                        f.severity,
                        f.priority,
                        f.signal_kind,
                        f.model_dump_json(),
                    ),
                )
            for result in suggestions:
                s = result.suggestion
                cx.execute(
                    "INSERT INTO suggestions VALUES (?,?,?,?)",
                    (str(s.id), str(s.finding_id), claim["id"], result.model_dump_json()),
                )

        return report, publish

    def _version(self, claim, checkpoint):
        payload = claim["request"]
        with self.store.transaction() as cx:
            patches = self.service.check_version_request(cx, payload["dataset_id"], payload)
        old = self.store.one("versions", payload["base_version_id"])
        analysis = self.store.job(payload["analysis_id"])
        profile, checksum = self.service.profile(analysis["request"]["profile_id"])
        if checksum != analysis["request"]["profile_sha256"]:
            raise WorkflowError("profile_changed")
        binding = self.service.binding(old)
        source = self.store.resolve(old["artifact_id"])
        operations = defaultdict(dict)
        by_record = defaultdict(list)
        for patch in patches:
            by_record[patch.record_id].append(patch)
            operations[patch.record_id][patch.operation.field_name] = patch.operation.after
        local_rules = [r for r in profile.rules if not isinstance(r.check, UniqueKeyCheck)]
        global_rules = [r for r in profile.rules if isinstance(r.check, UniqueKeyCheck)]
        engine = RuleEngine(local_rules, codebooks=list(profile.codebooks))
        context = context_for(profile, claim["id"])
        checks = {"before": 0, "after": 0}

        def validate(before, after):
            checks["before"] += len(before)
            checks["after"] += len(after)
            prior_map = {(o.record_id, o.rule_id): o for o in before}
            for outcome in after:
                prior = prior_map[outcome.record_id, outcome.rule_id]
                if outcome.status == "violation" and prior.status != "violation":
                    raise WorkflowError("patch_creates_rule_violation")
                if prior.status in {"passed", "violation"} and outcome.status not in {
                    "passed",
                    "violation",
                }:
                    raise WorkflowError("patch_degrades_rule_check")
                rule = next(r for r in profile.rules if r.id == outcome.rule_id)
                if (
                    outcome.status == "violation"
                    and set(rule.dependencies) & operations.get(outcome.record_id, {}).keys()
                ):
                    raise WorkflowError("patch_target_violation_unresolved")

        related_ids = {
            i for a in profile.aggregates for i in (*a.component_record_ids, a.total_record_id)
        }
        related = {}
        key_fields = {f for r in global_rules for f in r.dependencies}
        global_before, global_after = [], []
        seen_patches = set()

        def changed_records():
            for index, record in enumerate(read_records(source)):
                values = operations.get(record.id, {})
                for patch in by_record.get(record.id, ()):
                    verify_patch(patch, record, binding)
                    seen_patches.add(patch.id)
                if values:
                    updated = Record.model_validate(
                        {
                            **record.model_dump(),
                            "revision": record.revision + 1,
                            "dataset_revision": old["domain_revision"] + 1,
                            "fields": tuple(values.get(f.name, f) for f in record.fields),
                        }
                    )
                    validate(
                        engine.evaluate_many([record], context),
                        engine.evaluate_many([updated], context),
                    )
                else:
                    # The shipped non-key checks are record-local; unchanged fields cannot gain a new violation.
                    updated = record.model_copy(
                        update={"dataset_revision": old["domain_revision"] + 1}
                    )
                if record.id in related_ids:
                    related[record.id] = updated
                if global_rules:
                    if len(global_before) >= self.store.settings.max_analysis_records:
                        raise WorkflowError("global_key_validation_record_limit")
                    global_before.append(
                        record.model_copy(
                            update={
                                "fields": tuple(f for f in record.fields if f.name in key_fields)
                            }
                        )
                    )
                    global_after.append(
                        updated.model_copy(
                            update={
                                "fields": tuple(f for f in updated.fields if f.name in key_fields)
                            }
                        )
                    )
                if index % 512 == 0:
                    checkpoint()
                yield updated

        new_id = str(uuid4())
        destination = self.store.root / "versions" / (new_id + ".parquet")
        write_records(destination, changed_records())
        if seen_patches != {p.id for p in patches}:
            raise WorkflowError("patch_record_missing")
        if global_rules:
            keys = RuleEngine(global_rules, codebooks=list(profile.codebooks))
            validate(
                keys.evaluate_many(global_before, context),
                keys.evaluate_many(global_after, context),
            )
        contexts = {i: context for i in related}
        for relation in profile.aggregates:
            result = _aggregate(relation, related, contexts, context)
            if result.status != "passed":
                raise WorkflowError("patch_aggregate_not_validated")
        checkpoint()
        artifact_id = self.store.artifact(destination, "application/vnd.apache.parquet")
        new_hash = sha256(destination)
        self.store.resolve(old["artifact_id"])

        def publish(cx):
            self.service.check_version_request(cx, payload["dataset_id"], payload)
            report = {
                **json.loads(old["import_report_json"]),
                "confirmed_patch_count": len(patches),
                "review_event_ids": payload["review_event_ids"],
                "parent_version_id": old["id"],
                "raw_unchanged": True,
                "validation_rule_counts": checks,
                "validation_scope": "changed_rows_all_local_rules_global_keys_and_declared_aggregates",
                "aggregate_coverage": "checked_declared_relations"
                if profile.aggregates
                else "unavailable_reference",
            }
            cx.execute(
                "INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id,
                    old["dataset_id"],
                    old["number"] + 1,
                    old["domain_dataset_id"],
                    old["domain_revision"] + 1,
                    artifact_id,
                    new_hash,
                    old["mapping_sha256"],
                    old["source_sha256"],
                    old["id"],
                    canonical(report),
                    time.time(),
                ),
            )
            cx.execute(
                "UPDATE datasets SET current_version_id=?,etag=etag+1 WHERE id=?",
                (new_id, old["dataset_id"]),
            )

        return {
            "version_id": new_id,
            "parent_version_id": old["id"],
            "confirmed_patch_count": len(patches),
            "source_unchanged": True,
        }, publish

    def _export(self, claim, checkpoint):
        from forestcheck.api.exporting import create_export

        return create_export(self.store, claim, checkpoint), None
