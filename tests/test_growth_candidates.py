"""Synthetic candidate metadata never grants numerical reference admission."""

import pytest
from forestcheck.growth import GrowthTableRegistry
from forestcheck.growth.candidates import GrowthCandidateRegistry, GrowthReferenceCandidate
from pydantic import ValidationError


def candidate(**changes):
    payload = {
        "schema_version": "1.0",
        "candidate_id": "synthetic-candidate",
        "candidate_version": "1",
        "source": {
            "source_id": "synthetic-reference",
            "title": "Independently constructed test source",
            "page": "1",
        },
        "public_url": "https://example.org/synthetic-table.pdf",
        "source_access": "tables_inspected",
        "inspected_locations": ["Table A, page 1"],
        "species": ["synthetic-pine"],
        "regions": ["synthetic-region"],
        "stand_form": "normal",
        "purity": "pure",
        "fullness_value": 1.0,
        "fullness_definition": "Synthetic relative stocking convention",
        "unresolved": ["Real applicability is not established"],
    }
    return GrowthReferenceCandidate.model_validate(payload | changes)


def checked(**changes):
    return candidate(
        **({
            "status": "applicability_checked",
            "unresolved": [],
            "checked_scope_sha256": "a" * 64,
            "applicability_evidence": {
                "source_id": "synthetic-scope-check", "title": "Synthetic check", "section": "A"
            },
        } | changes)
    )


def test_found_candidate_is_not_an_admitted_growth_table():
    item = candidate()
    assert item.status == "found"
    with pytest.raises(TypeError, match="validated_growth_tables"):
        GrowthTableRegistry((item,))
    assert GrowthTableRegistry().tables == ()


@pytest.mark.parametrize("changes", [
    {"schema_version": "2.0"}, {"auto_admit": True},
    {"public_url": "private://real-input"}, {"fullness_value": True},
    {"source_access": "tables_inspected", "inspected_locations": []},
    {"status": "applicability_checked"},
    {"status": "expert_confirmed"},
])
def test_unknown_or_unsupported_claims_are_rejected(changes):
    with pytest.raises(ValidationError):
        candidate(**changes)


def test_unavailable_text_stays_found_and_has_no_inspection_claim():
    item = candidate(source_access="unavailable_text", inspected_locations=[])
    assert item.source_access == "unavailable_text" and item.status == "found"
    with pytest.raises(ValidationError):
        checked(source_access="unavailable_text", inspected_locations=[])


def test_checked_and_expert_confirmed_are_distinct_and_neither_admits():
    item = checked()
    assert item.status == "applicability_checked" and item.expert_confirmation is None
    with pytest.raises(ValidationError):
        checked(status="expert_confirmed")
    confirmed = checked(status="expert_confirmed", expert_confirmation={
        "source_id": "synthetic-expert", "title": "Synthetic confirmation", "section": "1",
    })
    with pytest.raises(TypeError, match="validated_growth_tables"):
        GrowthTableRegistry((confirmed,))


def test_unresolved_dimensions_cannot_be_claimed_checked():
    with pytest.raises(ValidationError):
        checked(unresolved=["Mixed stand conversion is unknown"])
    with pytest.raises(ValidationError):
        checked(checked_scope_sha256=None)
    with pytest.raises(ValidationError):
        checked(applicability_evidence=None)


def test_normal_pure_metadata_preserves_fullness_without_rescaling():
    item = candidate()
    payload = item.model_dump(mode="json")
    assert payload["purity"] == "pure" and payload["fullness_value"] == 1.0
    assert "points" not in payload and "proposed_value" not in payload
    with pytest.raises(ValidationError):
        candidate(fullness_definition=None)
    with pytest.raises(ValidationError):
        item.status = "expert_confirmed"


def test_candidate_registry_preserves_versions_and_rejects_duplicate_identity():
    first, second = candidate(), candidate(candidate_version="2")
    registry = GrowthCandidateRegistry(schema_version="1.0", candidates=(first, second))
    assert tuple(c.candidate_version for c in registry.candidates) == ("1", "2")
    with pytest.raises(ValidationError):
        GrowthCandidateRegistry(schema_version="1.0", candidates=(first, first))
