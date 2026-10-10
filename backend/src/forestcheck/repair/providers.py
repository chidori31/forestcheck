"""Evidence adapters. Unavailable components never supply invented replacement values."""

import math
from decimal import Decimal, localcontext

from forestcheck.repair.models import Candidate, SourceSupport
from forestcheck.rules.models import StockBalanceCheck


def model_source(prediction, *, model_id, plan, target, method):
    if method not in {"conditional_model", "baseline"}:
        raise ValueError("unsupported_observation_method")
    valid = prediction.status == "ok" and prediction.value is not None
    valid = valid and prediction.unit == target.unit
    basis = (
        f"{method}; frozen plan {plan.digest}; interval={prediction.interval_status}; "
        "reference=unverified source observations"
    )
    if prediction.unit != target.unit:
        basis += "; unit_mismatch_no_conversion"
    if prediction.warnings:
        basis += "; " + "; ".join(prediction.warnings)
    source = SourceSupport(
        source_id=f"model:{model_id}:{method}",
        version=str(model_id),
        method=method,
        status="available" if valid else "insufficient_data",
        basis=basis,
        dependency_roots=(f"observations:{plan.snapshot_sha256}",),
        input_fields=tuple(f.source_field for f in plan.features),
        limitations=(
            "unverified_measurement_procedure",
            "same_observations_as_baseline_and_comparables",
            "source_model_calibration_does_not_calibrate_final_recommendation",
        ),
    )
    candidate = (
        None
        if not valid
        else Candidate(
            support=source,
            value=prediction.value,
            lower=prediction.lower,
            upper=prediction.upper,
            unit=prediction.unit,
            uncertainty_basis="Conditional estimate of unverified observations; not error probability",
        )
    )
    return source, candidate


def growth_source(registry, query, *, field_bindings=None, allow_synthetic=False):
    result = registry.lookup(query, allow_synthetic=allow_synthetic)
    metadata = result.metadata
    dimensions = {"region", "species", "origin", "stand_type", "bonitet", "conditions", "age"}
    bindings = field_bindings or {}
    if set(bindings) - dimensions or any(not value for value in bindings.values()):
        raise ValueError("invalid_growth_field_bindings")
    required = {name for name in dimensions if getattr(query, name) is not None}
    status = result.status
    missing_bindings = status == "available" and not required.issubset(bindings)
    if missing_bindings:
        status = "insufficient_data"
    source = SourceSupport(
        source_id=f"growth:{metadata.table_id}" if metadata else "growth_registry",
        version=metadata.table_version if metadata else "registry-contract-1.0",
        method="growth",
        status=status,
        basis=(
            f"{result.reason}; {result.message}"
            + ("; query_field_dependencies_unbound" if missing_bindings else "")
            + (
                f"; {metadata.source.source_id}; page={metadata.source.page}; section={metadata.source.section}; "
                f"region={metadata.region}; species={metadata.species}; origin={metadata.origin}; "
                f"stand_type={metadata.stand_type}; bonitet={metadata.bonitet}; conditions={metadata.conditions}; "
                f"indicator={metadata.indicator_meaning}; method={result.method}"
                if metadata
                else ""
            )
        ),
        dependency_roots=(f"reference:{metadata.source.source_id}",) if metadata else (),
        input_fields=tuple(sorted(set(bindings.values()))),
        limitations=("reference_value_not_individual_truth",),
    )
    candidate = (
        None
        if status != "available"
        else Candidate(
            support=source,
            value=float(result.value),
            unit=result.unit,
            uncertainty_basis="Applicable table value; individual variation not calibrated",
        )
    )
    return source, candidate


def stock_sources(record, field_name, rule_engine, context, *, data_root):
    fields = {f.name: f for f in record.fields}
    sources, candidates = [], []
    for rule in rule_engine.rules:
        check = rule.check
        if not isinstance(check, StockBalanceCheck) or field_name not in rule.dependencies:
            continue
        other = tuple(n for n in (check.per_ha, check.area, check.total) if n != field_name)
        status, reason = "available", "Compatible documented stock relation; observational inputs"
        if context.record_level not in rule.record_levels:
            status, reason = "not_applicable", "relation_level_or_category_not_applicable"
        elif rule.categories and context.category is None:
            status, reason = "insufficient_data", "object_category_unavailable"
        elif rule.categories and context.category not in rule.categories:
            status, reason = "not_applicable", "relation_level_or_category_not_applicable"
        elif rule.source.kind == "heuristic" or check.tolerance_kind == "heuristic":
            status, reason = "unavailable_reference", "heuristic_relation_not_replacement_reference"
        elif rule.source.kind == "synthetic" and not context.allow_synthetic:
            status, reason = "unavailable_reference", "synthetic_rule_disabled"
        elif (
            not check.relationship_confirmed
            or len(set(check.quantity_levels)) != 1
            or check.quantity_levels[0] != context.record_level
            or len(set(check.stock_kinds)) != 1
            or check.tolerance_kind == "unavailable"
        ):
            status, reason = "unavailable_reference", "stock_relationship_or_tolerance_unconfirmed"
        elif any(n not in fields for n in rule.dependencies):
            status, reason = "insufficient_data", "relation_field_unavailable"
        elif [fields[n].unit for n in (check.per_ha, check.area, check.total)] != [
            "m3/ha",
            "ha",
            "m3",
        ]:
            status, reason = "insufficient_data", "unit_hypothesis_requires_review_no_conversion"
        elif any(
            fields[n].state != "observed" or type(fields[n].normalized_value) not in (int, float)
            for n in other
        ):
            status, reason = "insufficient_data", "relation_input_not_observed"
        number = None
        if status == "available":
            with localcontext() as decimal_context:
                decimal_context.prec = 2048
                values = {n: Decimal(str(fields[n].normalized_value)) for n in other}
                divisor = (
                    values.get(check.area)
                    if field_name == check.per_ha
                    else values.get(check.per_ha)
                )
                if field_name != check.total and divisor == 0:
                    status, reason = "insufficient_data", "zero_divisor"
                else:
                    number = float(
                        values[check.per_ha] * values[check.area]
                        if field_name == check.total
                        else values[check.total] / divisor
                    )
                    if not math.isfinite(number):
                        status, reason = "insufficient_data", "nonfinite_arithmetic_result"
        source = SourceSupport(
            source_id=rule.id,
            version=rule.version,
            method="arithmetic",
            status=status,
            basis=f"{reason}; {rule.source.reference_id}; {rule.source.locator}; tolerance={check.tolerance_kind}",
            dependency_roots=(data_root,),
            input_fields=other,
            limitations=(
                "relation_consistency_not_proof_of_true_input",
                "no_independent_measurement",
                "arithmetic_point_has_no_calibrated_uncertainty",
            ),
        )
        sources.append(source)
        if status == "available":
            candidates.append(
                Candidate(
                    support=source,
                    value=number,
                    unit=fields[field_name].unit,
                    uncertainty_basis="Arithmetic relation only; source tolerance is not confidence",
                )
            )
    if not sources:
        sources.append(
            SourceSupport(
                source_id="stock_relationships",
                version="rules-contract-1.0",
                method="arithmetic",
                status="unavailable_reference",
                basis="No applicable stock relation for requested field",
                dependency_roots=(),
                input_fields=(),
                limitations=("no_invented_relation",),
            )
        )
    return tuple(sources), tuple(candidates)
