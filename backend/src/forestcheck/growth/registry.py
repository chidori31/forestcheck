"""In-memory, exact-applicability growth references; no bundled forestry norms."""

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from forestcheck.domain.contracts import Contract, Scalar, ValueObject

ReferenceText = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
Number = StrictInt | StrictFloat
Age = Annotated[Number, Field(ge=0)]


class SourceReference(ValueObject):
    """A bibliographic or opaque private reference, with a precise source locator."""

    source_id: ReferenceText
    title: ReferenceText
    section: ReferenceText | None = None
    page: ReferenceText | None = None

    @model_validator(mode="after")
    def located_source(self) -> Self:
        if self.section is None and self.page is None:
            raise ValueError("source_section_or_page_required")
        return self


class AgeRange(ValueObject):
    minimum: Age
    maximum: Age
    unit: ReferenceText

    @model_validator(mode="after")
    def ordered_range(self) -> Self:
        if self.maximum < self.minimum:
            raise ValueError("invalid_age_range")
        return self


class InterpolationPolicy(ValueObject):
    """Only source-authorized linear interpolation exists; extrapolation is absent."""

    method: Literal["none", "linear"]
    authorization: SourceReference | None = None

    @model_validator(mode="after")
    def authorized_method(self) -> Self:
        if self.method == "linear" and self.authorization is None:
            raise ValueError("linear_interpolation_authority_required")
        return self


class GrowthTableMetadata(Contract):
    table_id: ReferenceText
    table_version: ReferenceText
    source: SourceReference
    region: ReferenceText
    species: ReferenceText
    origin: ReferenceText
    stand_type: ReferenceText
    bonitet: ReferenceText | None
    conditions: ReferenceText | None
    indicator: ReferenceText
    indicator_meaning: ReferenceText
    unit: ReferenceText
    age_range: AgeRange
    interpolation: InterpolationPolicy
    synthetic: StrictBool

    @model_validator(mode="after")
    def stated_conditions(self) -> Self:
        if self.bonitet is None and self.conditions is None:
            raise ValueError("bonitet_or_conditions_required")
        return self


class GrowthPoint(ValueObject):
    age: Age
    value: Number


class GrowthTable(Contract):
    metadata: GrowthTableMetadata
    points: Annotated[tuple[GrowthPoint, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def confirmed_range(self) -> Self:
        ages = tuple(point.age for point in self.points)
        if any(left >= right for left, right in zip(ages, ages[1:])):
            raise ValueError("table_ages_must_be_strictly_increasing")
        if (ages[0], ages[-1]) != (
            self.metadata.age_range.minimum,
            self.metadata.age_range.maximum,
        ):
            raise ValueError("points_must_cover_exact_confirmed_age_range")
        return self


class GrowthQuery(Contract):
    """None is missing information, never a wildcard for a reference dimension."""

    region: ReferenceText | None = None
    species: ReferenceText | None = None
    origin: ReferenceText | None = None
    stand_type: ReferenceText | None = None
    bonitet: ReferenceText | None = None
    conditions: ReferenceText | None = None
    indicator: ReferenceText | None = None
    unit: ReferenceText | None = None
    age_unit: ReferenceText | None = None
    age: Age | None = None
    allow_interpolation: StrictBool = False


class GrowthMismatch(ValueObject):
    table_id: ReferenceText
    table_version: ReferenceText
    field: ReferenceText
    reason: ReferenceText
    expected: Scalar
    actual: Scalar


class GrowthLookupResult(Contract):
    status: Literal["available", "unavailable_reference", "insufficient_data"]
    reason: ReferenceText
    message: ReferenceText
    value: Number | None = None
    unit: ReferenceText | None = None
    method: Literal["exact", "linear"] | None = None
    metadata: GrowthTableMetadata | None = None
    supporting_points: tuple[GrowthPoint, ...] = ()
    mismatches: tuple[GrowthMismatch, ...] = ()
    missing_fields: tuple[ReferenceText, ...] = ()
    candidate_ids: tuple[ReferenceText, ...] = ()

    @model_validator(mode="after")
    def result_consistency(self) -> Self:
        if self.status == "available":
            if self.value is None or self.unit is None or self.metadata is None:
                raise ValueError("available_reference_requires_value_and_provenance")
            expected_points = {"exact": 1, "linear": 2}.get(self.method)
            if expected_points != len(self.supporting_points):
                raise ValueError("available_reference_requires_supporting_points")
            if self.unit != self.metadata.unit:
                raise ValueError("result_unit_must_match_reference")
        elif any(item is not None for item in (self.value, self.unit, self.method, self.metadata)):
            raise ValueError("unavailable_reference_cannot_have_value")
        elif self.supporting_points:
            raise ValueError("unavailable_reference_cannot_have_supporting_points")
        return self


_DIMENSIONS = (
    "region",
    "species",
    "origin",
    "stand_type",
    "bonitet",
    "conditions",
    "indicator",
    "unit",
)
_REQUIRED_QUERY = (
    "region",
    "species",
    "origin",
    "stand_type",
    "indicator",
    "unit",
    "age_unit",
    "age",
)


@dataclass(frozen=True, init=False)
class GrowthTableRegistry:
    """Validated reference snapshots. Loading files or verifying sources is external."""

    tables: tuple[GrowthTable, ...]

    def __init__(self, tables: tuple[GrowthTable, ...] = ()) -> None:
        snapshots = tuple(tables)
        if any(not isinstance(table, GrowthTable) for table in snapshots):
            raise TypeError("registry_requires_validated_growth_tables")
        identities = [
            (table.metadata.table_id, table.metadata.table_version) for table in snapshots
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate_table_version")
        object.__setattr__(
            self,
            "tables",
            tuple(
                sorted(
                    snapshots,
                    key=lambda item: (item.metadata.table_id, item.metadata.table_version),
                )
            ),
        )

    def lookup(self, query: GrowthQuery, *, allow_synthetic: bool = False) -> GrowthLookupResult:
        if type(allow_synthetic) is not bool:
            raise TypeError("allow_synthetic_must_be_bool")
        if not isinstance(query, GrowthQuery):
            raise TypeError("lookup_requires_validated_growth_query")
        missing = [field for field in _REQUIRED_QUERY if getattr(query, field) is None]
        if query.bonitet is None and query.conditions is None:
            missing.extend(("bonitet", "conditions"))
        if not self.tables:
            return GrowthLookupResult(
                schema_version="1.0",
                status="unavailable_reference",
                reason="no_registered_reference",
                message="No growth references are registered.",
                missing_fields=tuple(missing),
            )
        if missing:
            return GrowthLookupResult(
                schema_version="1.0",
                status="insufficient_data",
                reason="missing_applicability",
                message="Required applicability or age information is missing.",
                missing_fields=tuple(missing),
            )

        rejected: list[GrowthMismatch] = []
        incomplete: set[str] = set()
        candidates: list[GrowthTable] = []
        for table in self.tables:
            metadata = table.metadata
            mismatches: list[GrowthMismatch] = []

            def reject(field: str, reason: str, expected: Scalar, actual: Scalar) -> None:
                mismatches.append(
                    GrowthMismatch(
                        table_id=metadata.table_id,
                        table_version=metadata.table_version,
                        field=field,
                        reason=reason,
                        expected=expected,
                        actual=actual,
                    )
                )

            if metadata.synthetic and not allow_synthetic:
                reject("synthetic", "synthetic_reference_excluded", False, True)
            for field in (*_DIMENSIONS, "age_unit"):
                expected = (
                    metadata.age_range.unit if field == "age_unit" else getattr(metadata, field)
                )
                actual = getattr(query, field)
                if expected != actual:
                    reason = "missing_applicability" if actual is None else "applicability_mismatch"
                    reject(field, reason, expected, actual)
            if mismatches:
                if all(item.reason == "missing_applicability" for item in mismatches):
                    incomplete.update(item.field for item in mismatches)
                rejected.extend(mismatches)
                continue
            if query.age < metadata.age_range.minimum or query.age > metadata.age_range.maximum:
                reject(
                    "age",
                    "age_outside_confirmed_range",
                    f"{metadata.age_range.minimum}..{metadata.age_range.maximum}",
                    query.age,
                )
            elif not any(point.age == query.age for point in table.points):
                if metadata.interpolation.method != "linear" or not query.allow_interpolation:
                    reject(
                        "interpolation",
                        "interpolation_not_authorized",
                        "authorized_linear",
                        metadata.interpolation.method
                        if query.allow_interpolation
                        else "not_requested",
                    )
            if not mismatches:
                candidates.append(table)
            rejected.extend(mismatches)

        if not candidates:
            return GrowthLookupResult(
                schema_version="1.0",
                status="insufficient_data" if incomplete else "unavailable_reference",
                reason="missing_applicability" if incomplete else "no_applicable_reference",
                message="No reference satisfies all applicability, age and interpolation requirements.",
                mismatches=tuple(rejected),
                missing_fields=tuple(sorted(incomplete)),
            )
        if len(candidates) > 1:
            return GrowthLookupResult(
                schema_version="1.0",
                status="unavailable_reference",
                reason="ambiguous_reference",
                message="Multiple applicable references remain; an explicit reference choice is required.",
                candidate_ids=tuple(table.metadata.table_id for table in candidates),
                mismatches=tuple(rejected),
            )

        table = candidates[0]
        exact = next((point for point in table.points if point.age == query.age), None)
        if exact is not None:
            value, method, points = exact.value, "exact", (exact,)
        else:
            lower, upper = next(
                (left, right)
                for left, right in zip(table.points, table.points[1:])
                if left.age < query.age < right.age
            )
            fraction = (query.age - lower.age) / (upper.age - lower.age)
            value = lower.value * (1 - fraction) + upper.value * fraction
            method, points = "linear", (lower, upper)
        return GrowthLookupResult(
            schema_version="1.0",
            status="available",
            reason="applicable_reference",
            message="An applicable reference supports the value using the stated method.",
            value=value,
            unit=table.metadata.unit,
            method=method,
            metadata=table.metadata,
            supporting_points=points,
            mismatches=tuple(rejected),
        )
