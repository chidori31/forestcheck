"""Explicit semantic sidecar; preserves both source fields and never infers their order."""
from typing import Literal, Self
from uuid import UUID

from pydantic import model_validator

from forestcheck.domain.contracts import Contract, FieldValue, Record, Text, ValueObject

Kind = Literal['woody', 'shrub', 'unresolved']


class HeightColumn(ValueObject):
    field: Text
    source_column: Text
    artifact_id: UUID | None = None
    source_table: Text | None = None
    semantic: Kind = 'unresolved'
    basis: Text | None = None

    @model_validator(mode='after')
    def assignment_evidence(self) -> Self:
        if self.semantic != 'unresolved' and (not self.basis or not self.artifact_id or not self.source_table):
            raise ValueError('height_semantic_assignment_requires_evidence')
        return self


class HeightMapping(Contract):
    mapping_version: Text
    columns: tuple[HeightColumn, HeightColumn]

    @model_validator(mode='after')
    def independent_columns(self) -> Self:
        if len({c.field for c in self.columns}) != 2 or len({c.source_column for c in self.columns}) != 2:
            raise ValueError('independent_height_positions_required')
        confirmed = [c.semantic for c in self.columns if c.semantic != 'unresolved']
        if len(confirmed) != len(set(confirmed)):
            raise ValueError('duplicate_height_semantic')
        return self


class HeightResolution(Contract):
    mapping_version: Text
    preprocessing_version: Literal['expert-preprocessing-1.0.0'] = 'expert-preprocessing-1.0.0'
    status: Literal['unresolved', 'available', 'conflict', 'insufficient_data', 'incompatible_unit']
    source_fields: tuple[FieldValue, ...]
    woody: FieldValue | None = None
    shrub: FieldValue | None = None
    candidate: FieldValue | None = None
    presentation_step: Literal['1', '0.1'] | None = None
    reason: Text


def resolve_heights(record: Record, mapping: HeightMapping, *, species_kind: Kind = 'unresolved',
                    species_basis: str | None = None) -> HeightResolution:
    fields = {f.name: f for f in record.fields}
    selected, semantic = [], {}
    for c in mapping.columns:
        field = fields.get(c.field)
        if field is None:
            continue
        if field.source.column != c.source_column:
            raise ValueError('height_source_position_mismatch')
        if (c.artifact_id is not None and c.artifact_id != field.source.artifact_id
                or c.source_table is not None and c.source_table != field.source.table):
            raise ValueError('height_source_binding_mismatch')
        selected.append(field)
        if c.semantic != 'unresolved':
            semantic[c.semantic] = field

    def result(status, reason, candidate=None, step=None):
        return HeightResolution(schema_version='1.0', mapping_version=mapping.mapping_version,
            status=status, source_fields=tuple(selected), woody=semantic.get('woody'), shrub=semantic.get('shrub'),
            candidate=candidate, presentation_step=step, reason=reason)

    if any(c.semantic == 'unresolved' for c in mapping.columns):
        return result('unresolved', 'Column semantics unconfirmed; no assignment by order or decimals')
    if len(selected) != 2:
        return result('insufficient_data', 'Mapped height field absent from export')
    observed = [f for f in selected if f.state == 'observed']
    if any(f.unit != 'm' for f in observed):
        return result('incompatible_unit', 'Metres required; no automatic scale conversion')
    if any(type(f.normalized_value) not in (float, int) for f in observed):
        return result('insufficient_data', 'Numeric height unavailable')
    if len(observed) == 2 and observed[0].normalized_value != observed[1].normalized_value:
        return result('conflict', 'Both heights populated differently; applicability requires expert review')
    if species_kind == 'unresolved' or not species_basis:
        return result('insufficient_data', 'Species kind requires independent confirmed evidence')
    candidate = semantic.get(species_kind)
    if candidate is None or candidate.state != 'observed':
        return result('insufficient_data', 'Applicable height is unavailable; missingness is not automatically an error')
    if any(f.state in ('parse_error', 'unknown') for f in selected):
        return result('insufficient_data', 'Unresolved other representation may conflict')
    return result('available', 'Explicit positional mapping and independent species kind; raw unchanged',
                  candidate, '1' if species_kind == 'woody' else '0.1')
