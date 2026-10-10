from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, model_validator

from forestcheck.domain.contracts import Contract, PositiveInt, Text, ValueObject


class ColumnMapping(ValueObject):
    position: PositiveInt
    source_header: str
    field: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    unit: Text | None
    transform: Literal["text", "integer", "decimal_comma", "decimal_point", "unknown"]
    basis: Text
    key_role: Literal["territory", "compartment", "stand", "layer", "element"] | None = None
    missing_tokens: tuple[str, ...] = ("",)
    not_applicable_tokens: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unambiguous_tokens(self) -> Self:
        if set(self.missing_tokens) & set(self.not_applicable_tokens):
            raise ValueError("overlapping_missing_states")
        if self.key_role and self.transform != "text":
            raise ValueError("identifiers_must_remain_text")
        return self


class ServiceRow(ValueObject):
    position: PositiveInt
    values: Annotated[tuple[str, ...], Field(min_length=1)]
    reason: Text


class ImportMapping(Contract):
    mapping_version: Text
    source_format: Literal["csv"]
    encoding: Literal["utf-8-sig", "utf-8", "cp1251"]
    delimiter: Literal[";", ",", "\t", "|"]
    table: Text
    record_level: Text
    grain_basis: Text
    columns: Annotated[tuple[ColumnMapping, ...], Field(min_length=1, max_length=2048)]
    analysis_blockers: tuple[Text, ...] = ()
    service_rows: tuple[ServiceRow, ...] = ()
    exclude_empty_records: StrictBool = True

    @model_validator(mode="after")
    def complete_positions(self) -> Self:
        if [c.position for c in self.columns] != list(range(1, len(self.columns) + 1)):
            raise ValueError("positions_must_be_complete_and_ordered")
        if len({c.field for c in self.columns}) != len(self.columns):
            raise ValueError("duplicate_internal_field")
        roles = [c.key_role for c in self.columns if c.key_role]
        if len(set(roles)) != len(roles):
            raise ValueError("duplicate_key_role")
        if any(s.position > len(self.columns) for s in self.service_rows):
            raise ValueError("service_column_out_of_range")
        return self
