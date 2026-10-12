"""Invented observations: bundle equality is not row deduplication."""
from uuid import uuid4

import pytest
from forestcheck.domain.contracts import FieldValue, Record, SourceCoordinates


def record(value=1):
    return Record(schema_version='1.0', id=uuid4(), dataset_id=uuid4(), fields=(
        FieldValue(schema_version='1.0', name='value', original_value=value,
                   normalized_value=value, unit='m', state='observed',
                   source=SourceCoordinates(artifact_id=uuid4(), table='invented', row=2, column='A')),))


def api():
    from forestcheck.ingest import bundle_duplicates
    return bundle_duplicates


def bundle(*values, object_id='object-a', period='2020', complete=True, parent=1):
    m = api()
    root = m.BundleNode(record=record(parent), level='stand', children=tuple(
        m.BundleNode(record=record(v), level='layer') for v in values))
    return m.ObservationBundle(schema_version='1.0', root=root, object_identity=object_id,
        observation_identity=period, identity_basis='Invented confirmed object and year',
        identity_namespace='invented-register', identity_version='1',
        comparison_basis='All source fields and complete descendants', complete=complete)


def test_order_independent_complete_equality_and_inactive_plan():
    a, b = bundle(1, 2, 2), bundle(2, 1, 2)
    before = a.model_dump_json()
    result = api().compare_bundles((a, b))
    assert len(result.duplicate_groups) == 1
    assert result.duplicate_groups[0].status == 'confirmed_full_duplicate'
    assert result.exclusion_plan[0].active is False
    assert len(result.exclusion_plan[0].excluded_source_record_ids) == 4
    assert a.model_dump_json() == before


@pytest.mark.parametrize('other', [dict(values=(1, 3)), dict(values=(1, 2, 2)),
                                  dict(values=(1, 2), parent=9)])
def test_same_parent_different_children_or_multiplicity_not_duplicate(other):
    values = other.pop('values')
    result = api().compare_bundles((bundle(1, 2), bundle(*values, **other)))
    assert not result.exclusion_plan
    assert result.duplicate_groups[0].status == 'different_bundle'


@pytest.mark.parametrize('context', [dict(object_id='other'), dict(period='2021')])
def test_other_object_or_year_never_excluded(context):
    result = api().compare_bundles((bundle(1), bundle(1, **context)))
    assert not result.exclusion_plan and not result.duplicate_groups


@pytest.mark.parametrize('context', [dict(period=None), dict(complete=False)])
def test_missing_identity_or_ambiguous_join_cannot_prove_duplicate(context):
    result = api().compare_bundles((bundle(1, **context), bundle(1, **context)))
    assert not result.exclusion_plan
    assert result.duplicate_groups[0].status == 'ambiguous_duplicate'


def test_hash_collision_is_not_equality(monkeypatch):
    monkeypatch.setattr(api(), 'content_digest', lambda _: '0' * 64)
    result = api().compare_bundles((bundle(1), bundle(2)))
    assert not result.exclusion_plan


def test_regeneration_difference_preserved():
    a, b = bundle(1), bundle(1)
    b = b.model_copy(update={'root': b.root.model_copy(update={'children': (
        *b.root.children, api().BundleNode(record=record(3), level='regeneration'))})})
    assert not api().compare_bundles((a, b)).exclusion_plan


def test_reused_source_child_prevents_ambiguous_join_proof():
    a, b = bundle(1), bundle(1)
    b = b.model_copy(update={'root': b.root.model_copy(update={'children': a.root.children})})
    result = api().compare_bundles((a, b))
    assert not result.exclusion_plan
    assert result.duplicate_groups[0].status == 'ambiguous_duplicate'


def test_unknown_version_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        api().ObservationBundle.model_validate({**bundle(1).model_dump(), 'schema_version': '2.0'})


def test_other_identity_namespace_is_not_same_object():
    a, b = bundle(1), bundle(1)
    b = b.model_copy(update={'identity_namespace': 'historical-local-number'})
    assert not api().compare_bundles((a, b)).exclusion_plan


def test_malformed_tree_cannot_prove_full_duplicate():
    from pydantic import ValidationError
    m = api()
    with pytest.raises(ValidationError, match='invalid_bundle_topology'):
        m.BundleNode(record=record(), level='stand', children=(m.BundleNode(record=record(), level='stand'),))


def test_virtual_layers_are_not_excluded_source_ids():
    m = api()
    a, b = bundle(1), bundle(1)
    virtual_ids = []
    converted = []
    for item in (a, b):
        child = item.root.children[0]
        virtual_ids.append(child.record.id)
        child = m.BundleNode(record=child.record, level='layer', physical_source_row=False,
                             children=(m.BundleNode(record=record(), level='element'),))
        converted.append(item.model_copy(update={'root': item.root.model_copy(update={'children': (child,)})}))
    result = m.compare_bundles(tuple(converted))
    assert result.exclusion_plan
    assert not set(virtual_ids) & set(result.exclusion_plan[0].excluded_source_record_ids)
