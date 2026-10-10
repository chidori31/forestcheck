"""Freeze object groups before features. A manifest is an auditable declaration,
not proof that unobserved spatial/temporal relationships do not exist.
"""

import hashlib
import json
from collections import defaultdict
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from forestcheck.domain.contracts import Contract, Digest, Text

Role = Literal["train", "tune", "calibration", "holdout", "development_cv"]
Fold = Annotated[int, Field(strict=True, ge=0)]


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_value(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class SplitIdentity(Contract):
    record_id: UUID
    dataset_id: UUID
    group_id: Digest
    parent_id: Digest
    duplicate_id: Digest | None = None


class Assignment(SplitIdentity):
    role: Role
    fold: Fold | None = None


class SplitManifest(Contract):
    plan_version: Literal["split-1.0.0"] = "split-1.0.0"
    seed: Text
    strategy: Literal["four_way", "group_cv", "official"]
    grouping_basis: Text
    inputs: dict[Text, Digest]
    assignments: Annotated[tuple[Assignment, ...], Field(min_length=1)]
    prior_exposed_groups: tuple[Digest, ...] = ()
    folds: Annotated[int, Field(strict=True, ge=3)] | None = None
    limitations: tuple[Text, ...] = ()
    official_source: Text | None = None

    @model_validator(mode="after")
    def independent_assignments(self):
        if not self.inputs:
            raise ValueError("input_hashes_required")
        validate_identities(self.assignments)
        groups = defaultdict(set)
        for a in self.assignments:
            groups[a.group_id].add((a.role, a.fold))
        if any(len(roles) != 1 for roles in groups.values()):
            raise ValueError("group_overlap")
        exposed = set(self.prior_exposed_groups)
        if not exposed <= groups.keys():
            raise ValueError("unknown_exposed_group")
        if any(a.group_id in exposed and a.role in {"holdout", "calibration"}
               for a in self.assignments):
            raise ValueError("exposed_group_in_independent_partition")
        if self.strategy == "group_cv":
            if self.folds is None or any(a.role != "development_cv" or a.fold is None
                                         or a.fold >= self.folds for a in self.assignments):
                raise ValueError("invalid_cv_assignments")
            if {a.fold for a in self.assignments} != set(range(self.folds)):
                raise ValueError("empty_cv_fold")
        elif self.folds is not None or any(a.fold is not None or a.role == "development_cv"
                                           for a in self.assignments):
            raise ValueError("unexpected_cv_assignment")
        if self.strategy == "four_way" and {a.role for a in self.assignments} != {
            "train", "tune", "calibration", "holdout"
        }:
            raise ValueError("empty_partition")
        if (self.strategy == "official") != (self.official_source is not None):
            raise ValueError("official_source_mismatch")
        return self

    @property
    def digest(self) -> str:
        return hash_value(self.model_dump(mode="json"))

    @cached_property
    def _assignment_index(self):
        return MappingProxyType({a.record_id: a for a in self.assignments})

    def assert_allowed(self, record_id: UUID, *, roles: set[str]) -> Assignment:
        assignment = self._assignment_index.get(record_id)
        if assignment is None:
            raise ValueError("record_not_in_split")
        if assignment.role not in roles:
            raise ValueError("role_not_allowed")
        return assignment


def validate_identities(rows):
    seen, parents, duplicates = set(), {}, {}
    for row in rows:
        if row.record_id in seen:
            raise ValueError("duplicate_record")
        seen.add(row.record_id)
        for kind, key, mapping in (("parent", row.parent_id, parents),
                                   ("duplicate", row.duplicate_id, duplicates)):
            if key is None:
                continue
            if key in mapping and mapping[key] != row.group_id:
                raise ValueError(f"unresolved_{kind}_linkage")
            mapping[key] = row.group_id


def freeze_split(
    identities, *, seed: str, grouping_basis: str, inputs: dict[str, str],
    prior_exposed_groups=(), minimum_groups_per_role: int = 5,
    official_roles: dict[str, str] | None = None, official_source: str | None = None,
    limitations=(),
) -> SplitManifest:
    rows = tuple(SplitIdentity.model_validate(r.model_dump()) for r in identities)
    validate_identities(rows)
    if type(minimum_groups_per_role) is not int or minimum_groups_per_role < 2:
        raise ValueError("minimum_groups_per_role")
    if not seed or not grouping_basis:
        raise ValueError("seed_and_grouping_basis_required")
    groups = {r.group_id for r in rows}
    exposed = set(prior_exposed_groups)
    if not exposed <= groups:
        raise ValueError("unknown_exposed_group")
    ordered = sorted(groups, key=lambda g: (hash_value([seed, g]), g))
    eligible = [g for g in ordered if g not in exposed]
    roles, fold_map, folds = {}, {}, None
    notes = list(limitations)
    if official_roles is not None:
        if not official_source or set(official_roles) != groups or any(
            r not in {"train", "tune", "calibration", "holdout"} for r in official_roles.values()
        ):
            raise ValueError("official_split_incomplete_or_invalid")
        if any(official_roles[g] in {"calibration", "holdout"} for g in exposed):
            raise ValueError("official_split_exposure_conflict")
        strategy, roles = "official", dict(official_roles)
    elif official_source is not None:
        raise ValueError("official_roles_required")
    elif len(groups) >= 4 * minimum_groups_per_role and len(eligible) >= 2 * minimum_groups_per_role:
        strategy = "four_way"
        independent_count = min(
            max(minimum_groups_per_role, int(len(groups) * .15)),
            len(eligible) // 2, (len(groups) - 2 * minimum_groups_per_role) // 2,
        )
        for g in eligible[:independent_count]:
            roles[g] = "holdout"
        for g in eligible[independent_count:2 * independent_count]:
            roles[g] = "calibration"
        development = [g for g in ordered if g not in roles]
        tune_count = max(minimum_groups_per_role, int(len(groups) * .15))
        tune_count = min(tune_count, len(development) - minimum_groups_per_role)
        roles.update({g: "tune" if i < tune_count else "train"
                      for i, g in enumerate(development)})
    else:
        if len(groups) < 3:
            raise ValueError("insufficient_independent_groups")
        strategy, folds = "group_cv", min(5, max(3, len(groups) // 3))
        roles = dict.fromkeys(ordered, "development_cv")
        fold_map = {g: i % folds for i, g in enumerate(ordered)}
        notes.append("Development group CV only; no independent calibration or final holdout. "
                     "Prior exposed groups are not untouched validation. Nested fitting and "
                     "new independent data are required for a final quality claim.")
    return SplitManifest(
        schema_version="1.0", seed=seed, strategy=strategy, grouping_basis=grouping_basis,
        inputs=inputs, prior_exposed_groups=tuple(sorted(exposed)), folds=folds,
        limitations=tuple(notes), official_source=official_source,
        assignments=tuple(Assignment(**r.model_dump(), role=roles[r.group_id],
                                     fold=fold_map.get(r.group_id))
                          for r in sorted(rows, key=lambda r: str(r.record_id))),
    )
