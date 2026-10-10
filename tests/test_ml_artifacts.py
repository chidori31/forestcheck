"""Only synthetic bytes and metadata; never loads arbitrary serialized Python."""

import json
import os
from uuid import uuid4

import pytest
from forestcheck.ml.artifacts import (
    ArtifactCompatibility,
    ArtifactMetadata,
    current_version,
    initialize_registry,
    load_candidate,
    promote,
    register_candidate,
    rollback,
)
from pydantic import ValidationError


def metadata(**changes):
    values = dict(
        schema_version="1.0",
        method_id="synthetic-native-cbm",
        config={"depth": 2},
        library_versions={"synthetic-library": "1.0"},
        seed=1701,
        record_schema_version="1.0",
        mapping_version="synthetic-1",
        mapping_sha256="1" * 64,
        split_sha256="2" * 64,
        train_sha256="3" * 64,
        features=("synthetic_age", "synthetic_species"),
        feature_lineage_sha256="4" * 64,
        target="synthetic_height",
        target_unit="m",
        dev_cohort_sha256="5" * 64,
        dev_report={"status": "RUN", "mae": 1.25, "reference": "observed_not_truth"},
        source_kind="synthetic",
        feedback_sha256=None,
    )
    values.update(changes)
    return ArtifactMetadata(**values)


def compatibility(meta):
    return ArtifactCompatibility(
        schema_version="1.0",
        record_schema_version=meta.record_schema_version,
        mapping_version=meta.mapping_version,
        mapping_sha256=meta.mapping_sha256,
        features=meta.features,
        target=meta.target,
        target_unit=meta.target_unit,
    )


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "synthetic-private-registry"
    path.mkdir()
    initialize_registry(path)
    return path


def candidate(registry, meta=None):
    return register_candidate(
        registry,
        meta or metadata(),
        {"point.cbm": b"synthetic-native-format", "baseline.json": b'{"median":2}'},
    )


def test_roundtrip_pins_metadata_bytes_and_starts_unpromoted(registry):
    meta = metadata()
    version = candidate(registry, meta)
    artifact = load_candidate(registry, version, compatibility(meta))
    assert artifact.model_id == version
    assert artifact.metadata == meta
    assert artifact.model_files["point.cbm"] == b"synthetic-native-format"
    assert current_version(registry) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "2.0"},
        {"surprise": True},
        {"features": ()},
        {"features": ("synthetic_height",)},
        {"features": ("a", "a")},
        {"mapping_sha256": "not-a-hash"},
        {"seed": True},
        {"record_schema_version": "2.0"},
        {"config": {"bad": float("nan")}},
    ],
)
def test_metadata_contract_rejects_unknown_and_invalid(changes):
    with pytest.raises((ValueError, ValidationError)):
        metadata(**changes)


@pytest.mark.parametrize(
    "change",
    [
        {"mapping_version": "other"},
        {"mapping_sha256": "a" * 64},
        {"target": "synthetic_diameter"},
        {"target_unit": "cm"},
        {"features": ("synthetic_species", "synthetic_age")},
    ],
)
def test_application_refuses_incompatible_schema_mapping_and_features(registry, change):
    meta = metadata()
    version = candidate(registry, meta)
    expected = compatibility(meta).model_dump()
    expected.update(change)
    with pytest.raises(ValueError, match="incompatible"):
        load_candidate(registry, version, ArtifactCompatibility(**expected))


def test_unregistered_version_cannot_load(registry):
    with pytest.raises(ValueError, match="unregistered"):
        load_candidate(registry, uuid4(), compatibility(metadata()))


def test_required_library_versions_are_checked(registry):
    version = candidate(registry)
    expected = compatibility(metadata()).model_dump()
    expected["library_versions"] = {"synthetic-library": "2.0"}
    with pytest.raises(ValueError, match="incompatible_library"):
        load_candidate(registry, version, ArtifactCompatibility(**expected))


def test_rechecked_native_path_and_size_limits(registry, monkeypatch):
    from forestcheck.ml import artifacts

    version = candidate(registry)
    artifact = load_candidate(registry, version, compatibility(metadata()))
    assert artifact.path("point.cbm").read_bytes() == b"synthetic-native-format"
    artifact.path("point.cbm").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        artifact.path("point.cbm")
    with pytest.raises(ValueError, match="unknown_model_file"):
        artifact.path("../point.cbm")
    monkeypatch.setattr(artifacts, "MAX_MODEL_BYTES", 1)
    with pytest.raises(ValueError, match="limit"):
        candidate(registry)


@pytest.mark.parametrize(
    "name", ["../bad.json", "/bad.json", "C:\\bad.json", "bad.pkl", "bad.joblib"]
)
def test_only_native_or_declarative_files_with_safe_names(registry, name):
    with pytest.raises(ValueError):
        register_candidate(registry, metadata(), {name: b"synthetic"})
    assert current_version(registry) is None


@pytest.mark.parametrize("payload", [b"not-json", b'{"x":NaN}', b'{"x":1,"x":2}'])
def test_declarative_json_is_parsed_strictly_before_registration(registry, payload):
    with pytest.raises(ValueError):
        register_candidate(registry, metadata(), {"baseline.json": payload})


def test_registry_requires_explicit_empty_initialization(tmp_path):
    path = tmp_path / "uncontrolled"
    path.mkdir()
    (path / "foreign.json").write_text("{}")
    with pytest.raises(ValueError):
        initialize_registry(path)
    with pytest.raises(ValueError):
        candidate(path)


@pytest.mark.parametrize("target", ["point.cbm", "manifest.json"])
def test_integrity_check_refuses_modified_artifact(registry, target):
    version = candidate(registry)
    file = registry / str(version) / target
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        load_candidate(registry, version, compatibility(metadata()))


def test_unlisted_file_refused(registry):
    version = candidate(registry)
    (registry / str(version) / "injected.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        load_candidate(registry, version, compatibility(metadata()))


def test_model_file_hardlinks_refused(registry, tmp_path):
    version = candidate(registry)
    file = registry / str(version) / "point.cbm"
    os.link(file, tmp_path / "second-link")
    with pytest.raises(ValueError, match="ordinary"):
        load_candidate(registry, version, compatibility(metadata()))


def test_hash_chained_receipt_tampering_refused(registry):
    candidate(registry)
    ledger = registry / "registry.jsonl"
    event = json.loads(ledger.read_text())
    event["manifest_sha256"] = "a" * 64
    ledger.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="integrity"):
        current_version(registry)


def test_manual_promotion_and_rollback_append_history(registry):
    first, second = candidate(registry), candidate(registry)
    before = (registry / "registry.jsonl").read_bytes()
    with pytest.raises(ValueError, match="manual"):
        promote(registry, first, expected_current=None, manual_confirmation=False, reason="check")
    promote(registry, first, expected_current=None, manual_confirmation=True, reason="reviewed")
    first_history = (registry / "registry.jsonl").read_bytes()
    assert first_history.startswith(before)
    promote(
        registry, second, expected_current=first, manual_confirmation=True, reason="dev reviewed"
    )
    assert current_version(registry) == second
    rollback(registry, first, expected_current=second, manual_confirmation=True, reason="revert")
    assert current_version(registry) == first
    assert (registry / "registry.jsonl").read_bytes().startswith(first_history)


@pytest.mark.parametrize(
    "change",
    [
        {"split_sha256": "a" * 64},
        {"dev_cohort_sha256": "b" * 64},
        {"target": "synthetic_diameter"},
        {"target_unit": "cm"},
    ],
)
def test_promotion_refuses_incomparable_dev_experiments(registry, change):
    first = candidate(registry)
    second = candidate(registry, metadata(**change))
    promote(registry, first, expected_current=None, manual_confirmation=True, reason="reviewed")
    with pytest.raises(ValueError, match="incomparable"):
        promote(registry, second, expected_current=first, manual_confirmation=True, reason="check")
    assert current_version(registry) == first


def test_stale_promotion_and_nonhistorical_rollback_refused(registry):
    first, second = candidate(registry), candidate(registry)
    promote(registry, first, expected_current=None, manual_confirmation=True, reason="reviewed")
    with pytest.raises(ValueError, match="stale"):
        promote(registry, second, expected_current=None, manual_confirmation=True, reason="check")
    with pytest.raises(ValueError, match="previously_promoted"):
        rollback(registry, second, expected_current=first, manual_confirmation=True, reason="check")
