"""Synthetic-only reference selection, split boundaries and descriptive statistics."""

from uuid import UUID

import pytest
from forestcheck.evaluation.split import Assignment, SplitManifest, hash_value
from forestcheck.repair.references import ReferenceObservation, ReferencePool, ReferenceQuery
from pydantic import ValidationError


def identity(number, group, role="train", **changes):
    return (
        dict(
            schema_version="1.0",
            record_id=UUID(int=number),
            dataset_id=UUID(int=1000),
            group_id=hash_value(["group", group]),
            parent_id=hash_value(["parent", number]),
            duplicate_id=None,
            role=role,
        )
        | changes
    )


def observation(number, group, value=10, **changes):
    values = identity(number, group) | dict(
        record_level="forest_element",
        region="synthetic-region",
        species="synthetic-species",
        age=40,
        age_unit="year",
        target_unit="m",
        target_field="height",
        target_value=value,
        target_state="observed",
        raw_record_sha256=hash_value(["full-raw-with-business-keys", number]),
        source_snapshot_sha256="a" * 64,
    )
    return ReferenceObservation(**(values | changes))


def query(number=999, group=99, role="tune", **changes):
    values = identity(number, group, role) | dict(
        record_level="forest_element",
        region="synthetic-region",
        species="synthetic-species",
        age=40,
        age_unit="year",
        target_unit="m",
        target_field="height",
        raw_record_sha256=hash_value(["full-raw-with-business-keys", number]),
    )
    return ReferenceQuery(**(values | changes))


def manifest(rows, queries):
    all_rows = {r.record_id: r for r in (*rows, *queries)}
    return SplitManifest(
        schema_version="1.0",
        seed="synthetic",
        strategy="official",
        grouping_basis="synthetic",
        inputs={"fake:parquet": "a" * 64, "fake:mapping": "b" * 64},
        official_source="Synthetic frozen fixture, not organizer split",
        assignments=tuple(
            Assignment(
                **{
                    k: getattr(r, k)
                    for k in (
                        "schema_version",
                        "record_id",
                        "dataset_id",
                        "group_id",
                        "parent_id",
                        "duplicate_id",
                        "role",
                    )
                }
            )
            for r in all_rows.values()
        ),
    )


def pool(rows, queries, **changes):
    options = dict(
        manifest=manifest(rows, queries),
        dataset_id=UUID(int=1000),
        target_field="height",
        target_unit="m",
        age_unit="year",
        snapshot_input_key="fake:parquet",
        snapshot_sha256="a" * 64,
        mapping_input_key="fake:mapping",
        mapping_version="synthetic-map-1",
        mapping_sha256="b" * 64,
        minimum_rows=20,
        minimum_groups=5,
    )
    return ReferencePool(rows, **(options | changes))


def training():
    return [observation(1 + 4 * g + r, g, 10 + g) for g in range(5) for r in range(4)]


def test_available_is_group_balanced_and_target_not_query_condition():
    rows, q = training(), query()
    result = pool(rows, [q]).query(q)
    assert result.status == "available"
    assert result.point == 12 and result.lower == pytest.approx(10.4)
    assert result.upper == pytest.approx(13.6)
    assert result.n_rows == 20 and result.n_groups == 5
    assert result.unit == "m"
    assert result.interval_kind == "descriptive_group_medians_uncalibrated"
    assert result.reference_ids_sha256 == hash_value(sorted(str(r.record_id) for r in rows))
    assert result.split_sha256 == manifest(rows, [q]).digest
    assert result.mapping_version == "synthetic-map-1"
    assert result.dependency_family == "source_snapshot:" + "a" * 64
    assert "not_expert_verified" in result.limitations
    assert "target_value" not in ReferenceQuery.model_fields


def test_big_group_does_not_dominate_median():
    rows = training() + [observation(100 + i, 0, 0) for i in range(100)]
    q = query()
    result = pool(rows, [q]).query(q)
    assert result.point == 12 and result.n_rows == 120
    assert result.n_groups == 5


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"region": "new-region"}, "no_exact_comparable_context"),
        ({"species": "unknown-code"}, "no_exact_comparable_context"),
        ({"age": 41}, "no_exact_comparable_context"),
        ({"record_level": "regeneration"}, "no_exact_comparable_context"),
        ({"target_unit": "cm"}, "target_unit_mismatch"),
        ({"age_unit": "month"}, "age_unit_mismatch"),
    ],
)
def test_no_implicit_context_distance_or_unit_conversion(changes, reason):
    q = query(**changes)
    result = pool(training(), [q]).query(q)
    assert result.status == "unavailable_reference" and result.reason == reason
    assert result.point is result.lower is result.upper is None


def test_missing_context_and_too_few_groups_are_not_success():
    q = query(region=None)
    result = pool(training(), [q]).query(q)
    assert result.status == "insufficient_data" and result.reason == "missing_context"
    q = query()
    rows = [observation(1 + i, 0, i) for i in range(100)]
    result = pool(rows, [q]).query(q)
    assert result.status == "insufficient_data"
    assert result.reason == "insufficient_reference_groups_or_rows"
    assert result.n_rows == 100 and result.n_groups == 1 and result.point is None


def test_exact_full_raw_copies_excluded_without_deduplicating_equal_numbers():
    rows, q = training(), query()
    rows += [observation(100, 0, 999, raw_record_sha256=rows[0].raw_record_sha256)]
    result = pool(rows, [q]).query(q)
    assert result.n_rows == 20 and result.point == 12
    assert result.excluded_counts["repeated_raw_copy"] == 1
    assert UUID(int=100) not in result.reference_ids


def test_current_object_related_group_parent_duplicate_and_whole_oof_fold_excluded():
    rows = training()
    rows += [observation(100, 0, 999, parent_id=rows[0].parent_id)]
    family = hash_value(["known-copy-family"])
    rows[0] = observation(1, 0, 10, duplicate_id=family)
    rows += [observation(101, 0, 999, duplicate_id=family)]
    q = query(1, 0, "train", duplicate_id=family)
    folds = {hash_value(["group", g]): g % 3 for g in range(5)}
    p = pool(rows, [q], train_folds=folds, minimum_rows=8, minimum_groups=2)
    with pytest.raises(ValueError, match="complete_validation_fold_exclusion_required"):
        p.query(q, excluded_groups={q.group_id})
    excluded = {g for g, fold in folds.items() if fold == 0}
    result = p.query(q, excluded_groups=excluded)
    assert result.status == "available" and result.n_rows == 12 and result.n_groups == 3
    assert set(result.reference_ids).isdisjoint({UUID(int=1), UUID(int=100), UUID(int=101)})
    assert result.excluded_counts["excluded_fold_group"] == 10


def test_train_query_requires_frozen_fold_map():
    rows = training()
    q = query(1, 0, "train")
    with pytest.raises(ValueError, match="train_query_requires_frozen_folds"):
        pool(rows, [q]).query(q)


@pytest.mark.parametrize("role", ["tune", "calibration", "holdout"])
def test_reference_pool_rejects_nontraining_reference(role):
    rows = [observation(1, 0, role=role)]
    with pytest.raises(ValueError, match="reference_role_not_train"):
        pool(rows, [query()])


def test_holdout_query_and_manifest_identity_spoofing_rejected():
    q = query(role="holdout")
    p = pool(training(), [q])
    with pytest.raises(ValueError, match="query_role_not_allowed"):
        p.query(q)
    q = query()
    p = pool(training(), [q])
    with pytest.raises(ValueError, match="split_identity_mismatch"):
        p.query(q.model_copy(update={"parent_id": "f" * 64}))


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"snapshot_sha256": "c" * 64}, "source_snapshot_not_bound_to_split"),
        ({"mapping_sha256": "c" * 64}, "mapping_not_bound_to_split"),
        ({"dataset_id": UUID(int=666)}, "reference_dataset_mismatch"),
    ],
)
def test_source_mapping_and_dataset_binding(changes, error):
    with pytest.raises(ValueError, match=error):
        pool(training(), [query()], **changes)


def test_reference_units_rejected_and_nonobserved_values_not_used():
    rows = training()
    with pytest.raises(ValueError, match="reference_unit_mismatch"):
        pool([rows[0].model_copy(update={"target_unit": "cm"})], [query()])
    rows[0] = observation(1, 0, None, target_state="parse_error")
    q = query()
    result = pool(rows, [q]).query(q)
    assert result.status == "insufficient_data" and result.n_rows == 19
    assert result.excluded_counts["unknown_target"] == 1
    assert result.point is None


def test_reference_and_query_fields_must_match_even_when_units_equal():
    rows, q = training(), query()
    with pytest.raises(ValueError, match="reference_target_field_mismatch"):
        pool([rows[0].model_copy(update={"target_field": "unrelated_height"})], [q])
    p = pool(rows, [q])
    with pytest.raises(ValueError, match="query_target_field_mismatch"):
        p.query(q.model_copy(update={"target_field": "unrelated_height"}))


def test_zero_valid_unknown_versions_and_fields_rejected():
    zero = observation(1, 0, 0)
    assert zero.target_value == 0
    for data in (
        zero.model_dump() | {"schema_version": "2.0"},
        zero.model_dump() | {"confidence_percent": 99},
        zero.model_dump() | {"target_value": float("nan")},
        zero.model_dump() | {"target_value": None},
    ):
        with pytest.raises(ValidationError):
            ReferenceObservation.model_validate(data)
    with pytest.raises(ValidationError):
        ReferenceQuery.model_validate(query().model_dump() | {"target_value": 12})


def test_empty_reference_and_invalid_support_configuration():
    q = query()
    assert pool([], [q]).query(q).status == "unavailable_reference"
    for options in ({"minimum_rows": 0}, {"minimum_groups": 1}, {"minimum_rows": True}):
        with pytest.raises(ValueError, match="minimum_reference_support_invalid"):
            pool(training(), [q], **options)


def test_reference_snapshot_and_row_identity_cannot_be_relabelled():
    q, rows = query(), training()
    with pytest.raises(ValueError, match="reference_snapshot_mismatch"):
        pool([rows[0].model_copy(update={"source_snapshot_sha256": "f" * 64})], [q])
    frozen = manifest(rows, [q])
    rows[0] = rows[0].model_copy(update={"parent_id": "f" * 64})
    with pytest.raises(ValueError, match="split_identity_mismatch"):
        pool(rows, [q], manifest=frozen)


def test_query_copy_exclusion_does_not_admit_another_member_of_that_family():
    # One row proven to be a query copy also links its known duplicate family.
    # The other family's representation must not leak through a different fingerprint.
    q, rows = query(), training()
    family = hash_value(["different-copy-representations"])
    rows += [observation(100, 0, 999, duplicate_id=family, raw_record_sha256=q.raw_record_sha256)]
    rows += [observation(101, 0, 999, duplicate_id=family)]
    result = pool(rows, [q]).query(q)
    assert result.n_rows == 16 and result.n_groups == 4
    assert result.status == "insufficient_data"
    assert UUID(int=101) not in result.reference_ids


def test_full_raw_copy_excludes_related_groups_transitively_before_context_matching():
    q, rows = query(), training()
    # Query copy in group0; another copy links group0 to group1, although the
    # linking rows do not match the query age. All these related records are excluded.
    rows += [observation(100, 0, 999, raw_record_sha256=q.raw_record_sha256)]
    copy = hash_value(["full-copy-age-90"])
    rows += [
        observation(101, 0, 999, age=90, raw_record_sha256=copy),
        observation(102, 1, 999, age=90, raw_record_sha256=copy),
    ]
    result = pool(rows, [q]).query(q)
    assert result.n_rows == 12 and result.n_groups == 3
    assert result.excluded_counts["query_copy"] == 1
    assert result.excluded_counts["copy_related_group"] == 8


def test_copy_linked_groups_not_counted_as_independent_reference_groups():
    q, rows = query(), training()
    copy = hash_value(["full-copy-with-business-keys"])
    rows += [
        observation(100, 0, 10, raw_record_sha256=copy),
        observation(101, 1, 10, raw_record_sha256=copy),
    ]
    result = pool(rows, [q]).query(q)
    assert result.n_groups == 4 and result.status == "insufficient_data"


def test_oof_excluded_fold_also_removes_copy_linked_training_groups():
    rows, q = training(), query(1, 0, "train")
    copy = hash_value(["full-copy-between-frozen-folds"])
    rows += [
        observation(100, 0, 10, age=90, raw_record_sha256=copy),
        observation(101, 1, 10, age=90, raw_record_sha256=copy),
    ]
    folds = {hash_value(["group", g]): g % 3 for g in range(5)}
    p = pool(rows, [q], train_folds=folds, minimum_rows=8, minimum_groups=2)
    result = p.query(q, excluded_groups={g for g, f in folds.items() if f == 0})
    assert result.n_rows == 8 and result.n_groups == 2 and result.point == 13
    assert result.excluded_counts["copy_related_group"] == 4


def test_final_reference_requires_matching_release_and_keeps_dev_guard():
    q = query(role="holdout")
    rows = training()
    p = pool(rows, [q])
    release = {
        "schema_version": "1.0",
        "split": manifest(rows, [q]).digest,
        "cohort": [str(q.record_id)],
        "groups": [q.group_id],
    }
    assert p.query_final(q, release=release).status == "available"
    for key, value in (("schema_version", "2.0"), ("split", "x"), ("cohort", []), ("groups", [])):
        with pytest.raises(ValueError, match="final_reference_release_mismatch"):
            p.query_final(q, release={**release, key: value})
    with pytest.raises(ValueError, match="query_role_not_allowed"):
        p.query(q)
