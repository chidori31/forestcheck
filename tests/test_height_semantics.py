"""Height values are invented, and positions require a separate source."""
from uuid import uuid4

import pytest
from forestcheck.domain.contracts import FieldValue, Record, SourceCoordinates

ARTIFACT = uuid4()


def api():
    from forestcheck.ingest import height_semantics
    return height_semantics


def row(a, b):
    return Record(schema_version='1.0', id=uuid4(), dataset_id=uuid4(), fields=tuple(
        FieldValue(schema_version='1.0', name=name, original_value=v,
            normalized_value=v, unit='m', state='missing' if v is None else 'observed',
            source=SourceCoordinates(artifact_id=ARTIFACT, table='invented', row=2, column=col))
        for name, col, v in [('h_a', 'column_0007', a), ('h_b', 'column_0008', b)]))


def mapping(confirmed=False, swapped=False):
    m = api()
    return m.HeightMapping(schema_version='1.0', mapping_version='test-1', columns=(
        m.HeightColumn(field='h_a', source_column='column_0007', artifact_id=ARTIFACT, source_table='invented', semantic=(
            'shrub' if swapped else 'woody') if confirmed else 'unresolved',
            basis='Synthetic explicit assignment' if confirmed else None),
        m.HeightColumn(field='h_b', source_column='column_0008', artifact_id=ARTIFACT, source_table='invented', semantic=(
            'woody' if swapped else 'shrub') if confirmed else 'unresolved',
            basis='Synthetic explicit assignment' if confirmed else None)))


def test_order_and_fraction_do_not_assign_semantics():
    out = api().resolve_heights(row(12, 1.2), mapping())
    assert out.status == 'unresolved' and out.woody is None and out.shrub is None


def test_explicit_swapped_columns_and_shrub_one_metre():
    r = row(1.0, None)
    before = r.model_dump_json()
    out = api().resolve_heights(r, mapping(True, swapped=True), species_kind='shrub', species_basis='Confirmed dictionary')
    assert out.shrub.normalized_value == 1.0 and out.shrub.unit == 'm'
    assert out.candidate.normalized_value == 1.0 and out.status == 'available'
    assert out.presentation_step == '0.1' and r.model_dump_json() == before


@pytest.mark.parametrize('a,b,status', [(12, 1.0, 'conflict'), (None, None, 'insufficient_data'),
                                       (12, 12, 'available')])
def test_two_columns_never_averaged_or_required_together(a, b, status):
    out = api().resolve_heights(row(a, b), mapping(True), species_kind='woody', species_basis='Confirmed dictionary')
    assert out.status == status
    if status != 'available':
        assert out.candidate is None


def test_unknown_species_kind_cannot_select_numeric_input():
    out = api().resolve_heights(row(12, None), mapping(True))
    assert out.status == 'insufficient_data' and out.candidate is None


def test_wrong_position_rejected():
    m = mapping(True)
    m = m.model_copy(update={'columns': (m.columns[0].model_copy(update={'source_column': 'Z'}), m.columns[1])})
    with pytest.raises(ValueError, match='height_source_position_mismatch'):
        api().resolve_heights(row(12, None), m)


def test_units_not_rescaled():
    r = row(12, None)
    r = r.model_copy(update={'fields': (r.fields[0].model_copy(update={'unit': 'dm'}), r.fields[1])})
    out = api().resolve_heights(r, mapping(True), species_kind='woody', species_basis='Dictionary')
    assert out.status == 'incompatible_unit' and out.candidate is None


def test_semantic_assignment_requires_evidence():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        api().HeightColumn(field='h', source_column='A', semantic='woody')


@pytest.mark.parametrize('change', [{'artifact_id': uuid4()}, {'table': 'other'}])
def test_same_position_in_other_source_cannot_reuse_mapping(change):
    r = row(12, None)
    r = r.model_copy(update={'fields': tuple(f.model_copy(update={'source': f.source.model_copy(update=change)}) for f in r.fields)})
    with pytest.raises(ValueError, match='height_source_binding_mismatch'):
        api().resolve_heights(r, mapping(True), species_kind='woody', species_basis='Confirmed dictionary')
