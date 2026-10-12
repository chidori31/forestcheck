"""Offline structural review of an immutable workbook, scoped before field analysis.

Local configuration and all detailed outputs are private. This is deliberately
not an API upload route and performs no training, correction or data exclusion.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

from forestcheck.domain.contracts import Record
from forestcheck.evaluation.split import SplitManifest, hash_value
from forestcheck.ingest.bundle_duplicates import BundleNode, ObservationBundle, compare_bundles
from forestcheck.ingest.csv_import import canonical, regular_output, sha256
from forestcheck.ingest.height_semantics import HeightMapping, resolve_heights
from forestcheck.ingest.xlsx_import import WorkbookMapping
from forestcheck.rules.hierarchy import HierarchyPolicy, evaluate_hierarchy


def validate_scope(scope, split_path):
    if sha256(split_path) != scope.get('split_sha256'):
        raise ValueError('split_hash_mismatch')
    if (set(scope) != {'schema_version', 'seed', 'split_sha256', 'groups', 'roles', 'purpose'}
            or scope['schema_version'] != '1.0' or not scope['seed'] or not scope['groups']
            or not set(scope['roles']) <= {'train', 'tune', 'development_cv'}):
        raise ValueError('invalid_scope')
    manifest = SplitManifest.model_validate_json(split_path.read_text('utf-8'))
    roles = defaultdict(set)
    for a in manifest.assignments:
        roles[a.group_id].add(a.role)
    if any(not roles[g] or not roles[g] <= set(scope['roles']) for g in scope['groups']):
        raise ValueError('scope_not_development')
    return set(scope['groups'])


def private_path(private, value, *, directory=False):
    p = Path(value)
    if not p.is_absolute() or not p.resolve().is_relative_to(private):
        raise ValueError('private_input_required')
    for ancestor in (p, *p.parents):
        if ancestor.is_symlink() or getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('private_link_forbidden')
    if not (p.is_dir() if directory else regular_output(p)):
        raise ValueError('private_input_type')
    return p


def snapshot_inputs(folder):
    manifest_path = private_path(folder, str(folder / 'manifest.json'))
    manifest = json.loads(manifest_path.read_text('utf-8'))
    if manifest.get('schema_version') != '1.0' or manifest.get('pipeline_version') != 'xlsx-import-1.0.1':
        raise ValueError('workbook_snapshot_version')
    mapping_path = private_path(folder, str(folder / 'mapping.json'))
    mapping = WorkbookMapping.model_validate_json(mapping_path.read_text('utf-8'))
    expected = {'mapping.json', 'report.json', 'cells.jsonl', 'quarantine.jsonl', 'relations.jsonl', 'diagnostics.jsonl', 'metadata.json'}
    expected.update(f'sheet_{i:04d}.parquet' for i in range(1, len(mapping.sheets) + 1))
    if set(manifest['output_hashes']) != expected or sha256(folder / 'mapping.json') != manifest['mapping_sha256']:
        raise ValueError('workbook_manifest_artifacts')
    hashes = {}
    for name, digest in manifest['output_hashes'].items():
        path = private_path(folder, str(folder / name))
        if sha256(path) != digest:
            raise ValueError('workbook_snapshot_integrity')
        hashes[name] = digest
    if manifest.get('source_sha256') != manifest.get('source_sha256_after'):
        raise ValueError('source_integrity_not_established')
    return mapping, hashes


def validate_cohort(cohort, paths, folder, groups):
    if (set(cohort) != {'version', 'snapshot_manifest_sha256', 'split_sha256', 'scope_sha256', 'records'}
            or cohort['version'] != 'expert-cohort-1.0.0'
            or cohort['snapshot_manifest_sha256'] != sha256(folder / 'manifest.json')
            or cohort['split_sha256'] != sha256(paths['split_file'])
            or cohort['scope_sha256'] != sha256(paths['scope_file'])):
        raise ValueError('cohort_snapshot_binding')
    manifest = SplitManifest.model_validate_json(paths['split_file'].read_text('utf-8'))
    assignments = {str(a.record_id): a for a in manifest.assignments}
    approved = {}
    if not cohort['records'] or len(cohort['records']) > 10000:
        raise ValueError('bounded_cohort_required')
    for item in cohort['records']:
        if set(item) != {'record_id', 'group_id', 'split_record_id'} or item['record_id'] in approved:
            raise ValueError('invalid_cohort_record')
        a = assignments.get(item['split_record_id'])
        if (a is None or a.group_id != item['group_id'] or a.group_id not in groups
                or a.role not in {'train', 'tune', 'development_cv'}):
            raise ValueError('cohort_not_development')
        approved[item['record_id']] = item['group_id']
    return approved


def review(config_path, *, workspace_root):
    import pyarrow.parquet as pq
    from forestcheck.ingest.expert_workbook import WorkbookHierarchyBinding, build_hierarchy

    started = time.perf_counter()
    config = json.loads(config_path.read_text('utf-8-sig'))
    required = {'private_dir', 'snapshot_dir', 'scope_file', 'split_file', 'hierarchy_binding_file',
                'policy_file', 'height_mapping_file', 'cohort_file'}
    if set(config) != required:
        raise ValueError('invalid_config')
    private = Path(config['private_dir'])
    if not private.is_absolute() or str(private).startswith(('\\\\', '//')):
        raise ValueError('absolute_local_private_directory_required')
    private = private_path(private.resolve(), str(private), directory=True)
    if private.resolve().is_relative_to(workspace_root.resolve()):
        raise ValueError('private_directory_must_be_outside_workspace')
    folder = private_path(private, config['snapshot_dir'], directory=True)
    private_path(folder, str(folder / 'manifest.json'))
    private_path(folder, str(folder / 'mapping.json'))
    paths = {k: private_path(private, config[k]) for k in required - {'private_dir', 'snapshot_dir'}}
    before = {k: sha256(v) for k, v in paths.items()}
    before['manifest'] = sha256(folder / 'manifest.json')
    scope = json.loads(paths['scope_file'].read_text('utf-8'))
    groups = validate_scope(scope, paths['split_file'])
    approved = validate_cohort(json.loads(paths['cohort_file'].read_text('utf-8')), paths, folder, groups)
    mapping, snapshot_hashes = snapshot_inputs(folder)
    binding = WorkbookHierarchyBinding.model_validate_json(paths['hierarchy_binding_file'].read_text('utf-8'))
    policy = HierarchyPolicy.model_validate_json(paths['policy_file'].read_text('utf-8'))
    heights = HeightMapping.model_validate_json(paths['height_mapping_file'].read_text('utf-8'))
    selected, ids, scanned = {}, set(), Counter()
    for i, sheet in enumerate(mapping.sheets, 1):
        selected[sheet.name] = []
        for batch in pq.ParquetFile(folder / f'sheet_{i:04d}.parquet').iter_batches(columns=['record_json'], batch_size=1024):
            for item in batch.to_pylist():
                raw = json.loads(item['record_json'])
                key = raw.get('business_key') or {}
                # Mechanical key bridge only; no statistics/features/rules on excluded groups.
                group = hash_value(['territory_compartment', key.get('territory'), key.get('compartment')])
                scanned[sheet.name] += 1
                if raw.get('id') not in approved:
                    continue
                if approved[raw['id']] != group:
                    raise ValueError('cohort_record_group_mismatch')
                record = Record.model_validate(raw)
                if len(ids) >= 10000:
                    raise ValueError('bounded_review_scope_exceeded')
                if record.id in ids:
                    raise ValueError('duplicate_source_record_id')
                selected[sheet.name].append(record)
                ids.add(record.id)
    if not ids:
        raise ValueError('empty_development_scope')
    if {str(i) for i in ids} != set(approved):
        raise ValueError('cohort_source_record_missing')
    relations = []
    with (folder / 'relations.jsonl').open(encoding='utf-8') as stream:
        selected_ids = {str(i) for i in ids}
        for line in stream:
            r = json.loads(line)
            if r['child_record_id'] in selected_ids or r.get('parent_record_id') in selected_ids:
                relations.append(r)
    snapshot = build_hierarchy({s: tuple(rows) for s, rows in selected.items()}, tuple(relations), binding)
    outcomes = evaluate_hierarchy(snapshot, policy, uuid4())
    height_results = [resolve_heights(r, heights) for r in selected.get(binding.element_sheet, ())]
    nodes = {n.record.id: n for n in snapshot.nodes}
    children = defaultdict(list)
    for n in snapshot.nodes:
        if n.parent_record_id in nodes and n.link_status == 'matched':
            children[n.parent_record_id].append(n)

    def tree(n):
        return BundleNode(record=n.record, level=n.level, physical_source_row=n.level != 'layer',
                          children=tuple(tree(c) for c in children[n.record.id]))

    bundles = []
    for n in snapshot.nodes:
        if n.level != 'stand':
            continue
        key = n.record.business_key
        object_id = hash_value(key.model_dump()) if key else None
        bundles.append(ObservationBundle(schema_version='1.0', root=tree(n), object_identity=object_id,
            identity_namespace='source-composite-key', identity_version=binding.version,
            observation_identity=None, identity_basis=None, comparison_basis=binding.version,
            complete=False, limitations=('Observation identity and exporter completeness unconfirmed',)))
    duplicates = compare_bundles(tuple(bundles))
    after = {k: sha256(v) for k, v in paths.items()}
    after['manifest'] = sha256(folder / 'manifest.json')
    if before != after or snapshot_inputs(folder)[1] != snapshot_hashes:
        raise ValueError('input_changed_during_review')
    parent = private / 'expert_reviews'
    parent.mkdir(exist_ok=True)
    private_path(private, str(parent), directory=True)
    destination = parent / ('run-' + uuid4().hex)
    destination.mkdir()
    report = dict(schema_version='1.0', preprocessing_version='expert-preprocessing-1.0.0',
        status='PARTIAL', scope=scope, inputs=before, source_rows={k: len(v) for k, v in selected.items()},
        mechanical_key_rows=dict(scanned), node_levels=dict(Counter(n.level for n in snapshot.nodes)),
        outcomes=dict(Counter(o.status for o in outcomes)), reasons=dict(Counter(o.reason for o in outcomes)),
        heights=dict(Counter(h.status for h in height_results)), excluded_records=0,
        model_training='NOT_RUN', expert_label_metrics='NOT_RUN', growth_reference='unavailable_reference',
        stock_checks='unavailable_reference: units and definitions unconfirmed',
        old_models='Unchanged; new semantic preprocessing is not admitted to existing model registry',
        seconds=time.perf_counter() - started, limitations=['Exporter completeness and observation identity unconfirmed',
        'Hierarchy results are local sidecars, not yet enabled by CSV API profile'])
    outputs = {'report.json': report, 'duplicate_groups.json': duplicates.model_dump(mode='json'),
               'hierarchy.json': snapshot.model_dump(mode='json')}
    for name, value in outputs.items():
        (destination / name).write_text(canonical(value), 'utf-8')
    for name, values in [('rule_outcomes.jsonl', outcomes), ('heights.jsonl', height_results)]:
        with (destination / name).open('x', encoding='utf-8') as stream:
            for value in values:
                stream.write(value.model_dump_json() + '\n')
    hashes = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
    (destination / 'manifest.json').write_text(canonical({'schema_version': '1.0', 'output_hashes': hashes}), 'utf-8')
    return destination


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid_arguments')


def main():
    try:
        parser = StatusParser(add_help=False)
        parser.add_argument('--config', required=True, type=Path)
        args = parser.parse_args()
        print('EXPERT_REVIEW_STARTED', flush=True)
        review(args.config, workspace_root=Path(__file__).resolve().parents[1])
    except KeyboardInterrupt:
        print('EXPERT_REVIEW_INTERRUPTED')
        return 130
    except Exception:
        print('EXPERT_REVIEW_FAILED')
        return 3
    print('EXPERT_REVIEW_COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
