"""Synthetic identities only; no forestry source rows."""

import hashlib
from uuid import NAMESPACE_URL, uuid5

import pytest
from forestcheck.evaluation.split import SplitIdentity, SplitManifest, freeze_split
from pydantic import ValidationError


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def identities(n=80):
    return tuple(
        SplitIdentity(
            schema_version="1.0",
            record_id=uuid5(NAMESPACE_URL, f"synthetic:{g}:{layer}"),
            dataset_id=uuid5(NAMESPACE_URL, f"table:{layer}"),
            group_id=digest(f"quarter:{g}"),
            parent_id=digest(f"stand:{g}"),
        )
        for g in range(n)
        for layer in (1, 2)
    )


def split(rows=None, **kwargs):
    return freeze_split(
        identities() if rows is None else rows,
        seed="synthetic-split-v1",
        grouping_basis="Synthetic quarters; both layers of each stand together",
        inputs={"synthetic_input": digest("input")},
        **kwargs,
    )


def test_deterministic_order_independent_whole_groups_and_hash():
    result = split()
    assert result == split(tuple(reversed(identities())))
    assert result.digest == split().digest
    assert result.strategy == "four_way"
    assert set(a.role for a in result.assignments) == {"train", "tune", "calibration", "holdout"}
    for g in {a.group_id for a in result.assignments}:
        assert len({a.role for a in result.assignments if a.group_id == g}) == 1
    assert SplitManifest.model_validate_json(result.model_dump_json()) == result


def test_prior_exposed_groups_never_calibration_or_holdout():
    exposed = tuple(digest(f"quarter:{g}") for g in range(20))
    result = split(prior_exposed_groups=exposed)
    assert all(a.role in {"train", "tune"} for a in result.assignments if a.group_id in exposed)


def test_small_group_cv_has_no_mini_holdout():
    result = split(identities(9))
    assert result.strategy == "group_cv"
    assert result.folds == 3
    assert set(a.role for a in result.assignments) == {"development_cv"}
    assert set(a.fold for a in result.assignments) == {0, 1, 2}
    assert result.limitations
    with pytest.raises(ValueError, match="insufficient_independent_groups"):
        split(identities(2))


def test_exposure_can_force_cv_and_unknown_exposure_is_rejected():
    result = split(prior_exposed_groups=tuple(digest(f"quarter:{g}") for g in range(76)))
    assert result.strategy == "group_cv"
    with pytest.raises(ValueError, match="unknown_exposed_group"):
        split(prior_exposed_groups=(digest("absent"),))


def test_group_overlap_rejected_on_loading_manifest():
    data = split().model_dump(mode="json")
    data["assignments"][1]["role"] = (
        "train" if data["assignments"][0]["role"] != "train" else "tune"
    )
    # Find two related rows independent of record UUID sort order.
    group = data["assignments"][0]["group_id"]
    related = [a for a in data["assignments"] if a["group_id"] == group]
    related[0]["role"], related[1]["role"] = "train", "holdout"
    with pytest.raises(ValidationError, match="group_overlap"):
        SplitManifest.model_validate(data)


def test_parent_and_duplicates_cannot_span_groups():
    rows = list(identities(20))
    rows[2] = rows[2].model_copy(update={"parent_id": rows[0].parent_id})
    with pytest.raises(ValueError, match="unresolved_parent_linkage"):
        split(rows)
    rows = list(identities(20))
    rows[0] = rows[0].model_copy(update={"duplicate_id": digest("same")})
    rows[2] = rows[2].model_copy(update={"duplicate_id": digest("same")})
    with pytest.raises(ValueError, match="unresolved_duplicate_linkage"):
        split(rows)


def test_unknown_version_fields_duplicate_records_and_holdout_guard():
    result = split()
    for update in ({"schema_version": "2.0"}, {"future": True}):
        with pytest.raises(ValidationError):
            SplitManifest.model_validate(result.model_dump(mode="json") | update)
    with pytest.raises(ValueError, match="duplicate_record"):
        split(identities() + identities()[:1])
    holdout = next(a for a in result.assignments if a.role == "holdout")
    with pytest.raises(ValueError, match="role_not_allowed"):
        result.assert_allowed(holdout.record_id, roles={"train", "tune"})
    with pytest.raises(ValueError, match="record_not_in_split"):
        result.assert_allowed(uuid5(NAMESPACE_URL, "absent"), roles={"train"})


def test_official_split_preserved_or_conflict_is_blocker():
    base = split()
    roles = {a.group_id: a.role for a in base.assignments}
    result = split(official_roles=roles, official_source="Synthetic organizer protocol §2")
    assert result.strategy == "official"
    assert {a.group_id: a.role for a in result.assignments} == roles
    exposed = next(g for g, role in roles.items() if role == "holdout")
    with pytest.raises(ValueError, match="official_split_exposure_conflict"):
        split(official_roles=roles, official_source="Synthetic §2", prior_exposed_groups=(exposed,))
