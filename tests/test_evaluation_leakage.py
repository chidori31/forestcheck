import pytest
from forestcheck.evaluation.leakage import FeatureColumn, fit_transformer, validate_features
from pydantic import ValidationError
from test_evaluation_split import identities, split


class Spy:
    called = False

    def fit(self, rows):
        self.called = True
        self.rows = rows


def columns():
    return (FeatureColumn(schema_version="1.0", name="height", source_kind="observed_field"),)


def test_labels_provenance_and_identifiers_never_features():
    for kind in ("label", "provenance", "identifier"):
        with pytest.raises(ValueError, match="non_predictor_feature"):
            validate_features((FeatureColumn(schema_version="1.0", name="height", source_kind=kind),))
    for name in ("is_error", "expected_value", "corruption_kind", "original_value", "split"):
        with pytest.raises(ValueError, match="reserved_feature"):
            validate_features((FeatureColumn(schema_version="1.0", name=name, source_kind="observed_field"),))
    with pytest.raises(ValidationError):
        FeatureColumn(schema_version="7", name="height", source_kind="observed_field")


def test_fit_on_test_tune_or_calibration_never_calls_transformer():
    manifest = split()
    for role in ("holdout", "tune", "calibration"):
        spy = Spy()
        a = next(a for a in manifest.assignments if a.role == role)
        with pytest.raises(ValueError, match="role_not_allowed"):
            fit_transformer(spy, {a.record_id: {"height": 12}}, manifest=manifest, columns=columns())
        assert not spy.called


def test_mixed_training_and_holdout_batch_rejected_before_fit():
    manifest = split()
    spy = Spy()
    rows = {next(a.record_id for a in manifest.assignments if a.role == role): {"height": 2}
            for role in ("train", "holdout")}
    with pytest.raises(ValueError, match="role_not_allowed"):
        fit_transformer(spy, rows, manifest=manifest, columns=columns())
    assert not spy.called


def test_training_fit_receipt_bound_to_split_and_allowlist():
    manifest = split()
    a = next(a for a in manifest.assignments if a.role == "train")
    spy = Spy()
    receipt = fit_transformer(
        spy, {a.record_id: {"height": 0}}, manifest=manifest, columns=columns()
    )
    assert spy.called and spy.rows == [{"height": 0}]
    assert receipt.split_sha256 == manifest.digest
    assert receipt.record_ids == (a.record_id,)
    with pytest.raises(ValueError, match="feature_keys_mismatch"):
        fit_transformer(Spy(), {a.record_id: {"height": 12, "label": True}}, manifest=manifest, columns=columns())


def test_cv_fit_excludes_validation_fold_and_requires_it():
    manifest = split(identities(9))
    held = manifest.assignments[0]
    train = next(a for a in manifest.assignments if a.fold != held.fold)
    with pytest.raises(ValueError, match="cv_validation_fold_required"):
        fit_transformer(Spy(), {train.record_id: {"height": 12}}, manifest=manifest, columns=columns())
    with pytest.raises(ValueError, match="cv_validation_leakage"):
        fit_transformer(Spy(), {held.record_id: {"height": 12}}, manifest=manifest, columns=columns(), validation_fold=held.fold)
    receipt = fit_transformer(Spy(), {train.record_id: {"height": 12}}, manifest=manifest, columns=columns(), validation_fold=held.fold)
    assert receipt.validation_fold == held.fold
