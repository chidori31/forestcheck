"""Conservative comparison of complete observation trees. No deletion is performed."""
import hashlib
import json
from collections import defaultdict
from typing import Literal, Self
from uuid import UUID

from pydantic import StrictBool, model_validator

from forestcheck.domain.contracts import Contract, Digest, Record, Text, ValueObject

PREPROCESSING_VERSION = 'expert-preprocessing-1.0.0'


class BundleNode(ValueObject):
    record: Record
    level: Literal['stand', 'layer', 'element', 'regeneration', 'other']
    children: tuple['BundleNode', ...] = ()
    physical_source_row: StrictBool = True

    @model_validator(mode='after')
    def topology(self) -> Self:
        allowed = {'stand': {'layer', 'regeneration', 'other'}, 'layer': {'element', 'other'},
                   'element': set(), 'regeneration': set(), 'other': set()}
        if (any(c.level not in allowed[self.level] for c in self.children)
                or not self.physical_source_row and self.level != 'layer'):
            raise ValueError('invalid_bundle_topology')
        return self


def source_ids(node):
    own = (node.record.id,) if node.physical_source_row else ()
    return (*own, *(i for child in node.children for i in source_ids(child)))


class ObservationBundle(Contract):
    root: BundleNode
    object_identity: Text | None = None
    observation_identity: Text | None = None
    identity_basis: Text | None = None
    identity_namespace: Text | None = None
    identity_version: Text | None = None
    comparison_basis: Text
    complete: StrictBool
    limitations: tuple[Text, ...] = ()

    @model_validator(mode='after')
    def valid_tree(self) -> Self:
        ids = source_ids(self.root)
        if self.root.level != 'stand' or len(ids) != len(set(ids)):
            raise ValueError('stand_root_and_distinct_source_records_required')
        return self


class DuplicateGroup(ValueObject):
    root_ids: tuple[UUID, ...]
    status: Literal['confirmed_full_duplicate', 'different_bundle', 'ambiguous_duplicate']
    reason: Text
    content_sha256: Digest | None = None


class ExclusionProposal(ValueObject):
    representative: UUID
    excluded_root_ids: tuple[UUID, ...]
    excluded_source_record_ids: tuple[UUID, ...]
    content_sha256: Digest
    evidence: Text
    active: Literal[False] = False


class DuplicateReport(Contract):
    preprocessing_version: Literal['expert-preprocessing-1.0.0'] = PREPROCESSING_VERSION
    duplicate_groups: tuple[DuplicateGroup, ...]
    exclusion_plan: tuple[ExclusionProposal, ...]
    row_level_exclusion_admissible: Literal[False] = False


def canonical_content(node):
    """Exclude coordinates/technical IDs; keep typed raw, state, units and multiplicity."""
    fields = [f.model_dump(mode='json', exclude={'source', 'schema_version'}) for f in node.record.fields]
    return json.dumps({'level': node.level, 'fields': sorted(fields, key=lambda f: f['name']),
        'children': sorted(canonical_content(c) for c in node.children)},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_digest(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def compare_bundles(bundles: tuple[ObservationBundle, ...]) -> DuplicateReport:
    """Hash narrows candidates; full canonical equality proves representation equality.

    Completeness and event identity must be justified upstream. Any shared source ID,
    unresolved relation or incompatible comparison contract prevents an exclusion plan.
    """
    contexts, occurrences = defaultdict(list), defaultdict(int)
    roots = [b.root.record.id for b in bundles]
    if len(set(roots)) != len(roots):
        raise ValueError('duplicate_bundle_root')
    for b in bundles:
        contexts[(b.identity_namespace, b.identity_version, b.object_identity, b.observation_identity)].append(b)
        for sid in source_ids(b.root):
            occurrences[sid] += 1
    groups, plans = [], []
    for (namespace, version, object_id, period), candidates in contexts.items():
        if len(candidates) < 2:
            continue
        root_ids = tuple(sorted((b.root.record.id for b in candidates), key=str))
        uncertain = (not namespace or not version or not object_id or not period or any(not b.complete or not b.identity_basis or b.limitations
            or any(occurrences[sid] > 1 for sid in source_ids(b.root)) for b in candidates)
            or len({(b.comparison_basis, b.identity_basis) for b in candidates}) != 1)
        if uncertain:
            groups.append(DuplicateGroup(root_ids=root_ids, status='ambiguous_duplicate',
                reason='Event identity, full coverage, source ownership or comparison contract is unresolved'))
            continue
        buckets = defaultdict(list)
        # A digest collision must still fail the second, full-content comparison.
        for b in candidates:
            content = canonical_content(b.root)
            buckets[(content_digest(content), content)].append(b)
        if len(buckets) > 1:
            groups.append(DuplicateGroup(root_ids=root_ids, status='different_bundle',
                reason='Parent fields, descendants, relations or multiplicity differ'))
        for (digest, _), equal in buckets.items():
            if len(equal) < 2:
                continue
            equal.sort(key=lambda b: str(b.root.record.id))
            ids = tuple(b.root.record.id for b in equal)
            groups.append(DuplicateGroup(root_ids=ids, status='confirmed_full_duplicate', content_sha256=digest,
                reason='Same confirmed object/event; complete trees compared including multiplicity'))
            plans.append(ExclusionProposal(representative=ids[0], excluded_root_ids=ids[1:],
                excluded_source_record_ids=tuple(sid for b in equal[1:] for sid in source_ids(b.root)),
                content_sha256=digest, evidence='Full canonical equality after hash; ' + equal[0].identity_basis))
    return DuplicateReport(schema_version='1.0', duplicate_groups=tuple(groups), exclusion_plan=tuple(plans))
