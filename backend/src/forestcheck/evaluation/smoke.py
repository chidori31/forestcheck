"""Entirely invented, non-biological end-to-end infrastructure smoke test."""

import json
import time
import tracemalloc
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from forestcheck.domain.contracts import FieldValue, Record, SourceCoordinates
from forestcheck.evaluation.annotation import build_annotation_queue
from forestcheck.evaluation.corruption import CorruptionSpec, case_payloads, generate_case
from forestcheck.evaluation.scoring import (
    DetectionCase,
    ResourceMeasurement,
    ScoringProtocol,
    score_frozen_cohort,
)
from forestcheck.evaluation.split import SplitIdentity, freeze_split, hash_value


def run_synthetic_smoke(destination: Path) -> Path:
    if tracemalloc.is_tracing():
        raise ValueError("external_tracemalloc_session_active")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("new_synthetic_output_directory_required")
    started = time.perf_counter()
    tracemalloc.start()
    try:
        return _run(destination, started)
    finally:
        tracemalloc.stop()


def _run(destination, started):
    dataset = uuid5(NAMESPACE_URL, "forestcheck:invented-evaluation-smoke")
    identities = tuple(SplitIdentity(
        schema_version="1.0", record_id=uuid5(dataset, str(i)), dataset_id=dataset,
        group_id=hash_value(["synthetic-group", i]), parent_id=hash_value(["synthetic-parent", i]),
    ) for i in range(80))
    manifest = freeze_split(identities, seed="synthetic-smoke-v1",
                            grouping_basis="Invented independent object identifiers",
                            inputs={"invented_generator": hash_value("smoke-v1")})
    destination.mkdir(parents=True)

    def save(name, value):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    save("split-manifest.json", manifest.model_dump(mode="json"))
    train = [a for a in manifest.assignments if a.role == "train"]
    source = SourceCoordinates(artifact_id=dataset, table="invented", row=1, column="x")

    def record(assignment, x=20):
        return Record(schema_version="1.0", id=assignment.record_id, dataset_id=dataset,
                      fields=tuple(FieldValue(schema_version="1.0", name=name,
                          original_value=value, normalized_value=value, state="observed", unit=unit,
                          source=source.model_copy(update={"column": name}))
                          for name, value, unit in (("x", x, "m"), ("y", 2, "m"),
                                                     ("total", x * 2, "m2"), ("code", "A", None))))

    def field(name, unit="m", kind="number"):
        return {"name": name, "unit": unit, "data_type": kind}

    definitions = [
        ("missing", [field("x")], {}),
        ("decimal_shift", [field("x")], {"factor": "0.01"}),
        ("wrong_unit", [field("x")], {"replacement_unit": "cm"}),
        ("field_swap", [field("x"), field("y")], {}),
        ("invalid_code", [field("code", None, "code")],
         {"allowed_codes": ["A", "B"], "replacement_code": "INVALID", "codebook_version": "synthetic-1"}),
        ("multi_field", [field("x"), field("y"), field("total", "m2")],
         {"factor": "10", "relation": "product", "tolerance": "0",
          "relation_scope": "Invented rectangle area, one object", "relation_units_confirmed": True}),
    ]
    for index, (kind, fields, params) in enumerate(definitions):
        spec = CorruptionSpec(schema_version="1.0", id=f"synthetic.{kind}", version="1",
                              kind=kind, fields=fields, source="Invented fixture, no forestry norms", **params)
        case = generate_case(record(train[index]), spec, manifest=manifest, seed="independent-smoke-17",
                             mode="synthetic_stress", source_kind="synthetic")
        assert case.original.revision == 1 and case.modified.revision == 2
        for name, content in case_payloads(case).items():
            path = destination / "cases" / str(case.lineage.case_id) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
    # Rare admissible cases are defined by the synthetic algebra, not biological bounds.
    save("rare-valid-observations.json", [record(train[6], 0).model_dump(mode="json"),
                                         record(train[7], 1000).model_dump(mode="json")])
    queue = build_annotation_queue(tuple(a.record_id for a in train[:12]),
                                   flagged=tuple(a.record_id for a in train[:6]), manifest=manifest,
                                   seed="synthetic-review-v1", control_count=3, max_flagged=3)
    save("annotation-forms.json", [f.model_dump(mode="json") for f in queue.forms])
    save("annotation-sampling.json", [s.model_dump(mode="json") for s in queue.sampling])
    # Hand-authored arithmetic fixture, independent of corruption labels.
    # This is a scorer check, not a trained predictor or a forestry-quality result.
    cases = tuple(DetectionCase(schema_version="1.0", record_id=a.record_id, group_id=a.group_id,
                                 field_name="x", is_error=truth, predicted_error=prediction,
                                 queue_rank=i + 1)
                  for i, (a, truth, prediction) in enumerate(zip(train[:4],
                                         (True, False, True, False), (True, True, False, False))))
    elapsed, peak = time.perf_counter() - started, tracemalloc.get_traced_memory()[1]
    report = score_frozen_cohort(
        ScoringProtocol(schema_version="1.0", mode="synthetic_stress", label_basis="synthetic_truth",
                        evaluated_fields=("x",), evaluated_method_id="hand-authored-prediction-fixture",
                        complete_labels=True, label_producer_ids=("independent-arithmetic-fixture",)),
        manifest, tuple(a.record_id for a in train[:4]), cases, role="train", queue_k=2,
        resources=ResourceMeasurement(schema_version="1.0", wall_seconds=elapsed,
           peak_memory_bytes=peak, measurement_method="perf_counter; tracemalloc peak Python allocations, not process RSS"),
    )
    assert report.rows.precision.value == .5 and report.rows.recall.value == .5
    save("scorer-arithmetic-fixture.json", report.model_dump(mode="json"))
    save("smoke-report.json", {"schema_version": "1.0", "source_kind": "synthetic", "status": "PASSED",
         "corruption_cases": 6, "rare_valid_cases": 2, "real_metrics": "NOT_RUN", "model_training": "NOT_RUN",
         "wall_seconds": elapsed, "peak_traced_python_bytes": peak, "process_peak_rss": "NOT_RUN",
         "interpretation": "Infrastructure and hand-calculated arithmetic only; no model quality claim"})
    return destination
