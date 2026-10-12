"""Local expert iteration runs only explicit frozen development groups."""
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest
from forestcheck.evaluation.split import SplitIdentity, freeze_split
from forestcheck.ingest.csv_import import sha256


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/review_expert_local.py'
    spec = importlib.util.spec_from_file_location('expert_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    return freeze_split(tuple(SplitIdentity(schema_version='1.0', record_id=uuid4(), dataset_id=uuid4(),
        group_id=f'{i:064x}', parent_id=f'{i+100:064x}') for i in range(30)),
        seed='synthetic', grouping_basis='Invented independent groups', inputs={'synthetic': 'a' * 64})


def test_scope_rejects_holdout_before_loading_fields(tmp_path):
    m = manifest()
    p = tmp_path / 'split.json'
    p.write_text(m.model_dump_json(), 'utf-8')
    scope = dict(schema_version='1.0', seed='synthetic', split_sha256=sha256(p),
        groups=[next(a.group_id for a in m.assignments if a.role == 'holdout')], roles=['train', 'tune'], purpose='development')
    with pytest.raises(ValueError, match='scope_not_development'):
        runner().validate_scope(scope, p)


def test_scope_hash_checked(tmp_path):
    p = tmp_path / 'split.json'
    p.write_text('{}', 'utf-8')
    with pytest.raises(ValueError, match='split_hash_mismatch'):
        runner().validate_scope(dict(split_sha256='a' * 64), p)


def test_cli_failure_stdout_contains_no_path_or_value(tmp_path, capsys, monkeypatch):
    p = tmp_path / 'private-secret-config.json'
    p.write_text(json.dumps({'private_dir': str(tmp_path)}), 'utf-8')
    monkeypatch.setattr('sys.argv', ['review_expert_local.py', '--config', str(p)])
    assert runner().main() == 3
    output = capsys.readouterr().out
    assert 'FAILED' in output and str(tmp_path) not in output and 'Traceback' not in output


def test_complete_synthetic_workbook_review_writes_private_sidecars(tmp_path):
    from forestcheck.evaluation.split import hash_value
    from forestcheck.ingest.xlsx_import import import_workbook
    from test_xlsx_import import make_book, mapping

    private, source_dir, workspace = (tmp_path / n for n in ('private', 'source', 'workspace'))
    for p in (private, source_dir, workspace):
        p.mkdir()
    source = source_dir / 'synthetic.xlsx'
    make_book(source)
    snapshot = import_workbook(source, mapping(), private, workspace_root=workspace)
    group = hash_value(['territory_compartment', '01', '02'])
    identities = tuple(SplitIdentity(schema_version='1.0', record_id=uuid4(), dataset_id=uuid4(),
        group_id=group if i == 0 else f'{i:064x}', parent_id=f'{i+100:064x}') for i in range(30))
    m = freeze_split(identities, seed='synthetic', grouping_basis='Invented groups', inputs={'synthetic': 'a' * 64},
                     prior_exposed_groups=(group,))
    split = private / 'split.json'
    split.write_text(m.model_dump_json(), 'utf-8')
    scope = dict(schema_version='1.0', seed='synthetic', split_sha256=sha256(split), groups=[group],
                 roles=['train', 'tune'], purpose='synthetic test')
    evidence = dict(reference_id='invented', locator='Synthetic fixture', kind='synthetic')
    binding = dict(schema_version='1.0', version='synthetic-1', root_sheet='plots', element_sheet='layers',
        regeneration_sheet='regeneration', layer_id_field='column_0003', layer_fields=['column_0004'],
        element_identity_field='column_0003', source=evidence,
        sheet_completeness={'plots': True, 'layers': True, 'regeneration': True})
    policy = dict(schema_version='1.0', id='synthetic', version='1', source=evidence,
                  fields=dict(area='column_0004', inventory_number='column_0003', land_kind='column_0005'))
    heights = dict(schema_version='1.0', mapping_version='synthetic-1', columns=[
        dict(field='column_0004', source_column='column_0004'), dict(field='column_0005', source_column='column_0005')])
    config = dict(private_dir=str(private), snapshot_dir=str(snapshot), split_file=str(split))
    for key, value in [('scope_file', scope), ('hierarchy_binding_file', binding), ('policy_file', policy), ('height_mapping_file', heights)]:
        p = private / (key + '.json')
        p.write_text(json.dumps(value), 'utf-8')
        config[key] = str(p)
    import pyarrow.parquet as pq
    bridge_id = next(str(a.record_id) for a in m.assignments if a.group_id == group)
    records = [dict(record_id=item['record_id'], group_id=group, split_record_id=bridge_id)
        for i in range(1, 4) for item in pq.read_table(snapshot / f'sheet_{i:04d}.parquet', columns=['record_id']).to_pylist()]
    cohort = dict(version='expert-cohort-1.0.0', snapshot_manifest_sha256=sha256(snapshot / 'manifest.json'),
                  split_sha256=sha256(split), scope_sha256=sha256(Path(config['scope_file'])), records=records)
    cohort_path = private / 'cohort.json'
    cohort_path.write_text(json.dumps(cohort), 'utf-8')
    config['cohort_file'] = str(cohort_path)
    path = workspace / 'config.json'
    path.write_text(json.dumps(config), 'utf-8')
    before = source.read_bytes()
    result = runner().review(path, workspace_root=workspace)
    report = json.loads((result / 'report.json').read_text('utf-8'))
    assert report['source_rows'] == {'plots': 1, 'layers': 2, 'regeneration': 1}
    assert report['node_levels'] == {'stand': 1, 'layer': 1, 'element': 2, 'regeneration': 1}
    assert report['excluded_records'] == 0 and report['model_training'] == 'NOT_RUN'
    assert report['heights'] == {'unresolved': 2} and source.read_bytes() == before
    assert result.is_relative_to(private)
    # Omitting a child from a frozen subset cannot assert a complete population.
    omitted = pq.read_table(snapshot / 'sheet_0002.parquet', columns=['record_id']).to_pylist()[0]['record_id']
    cohort['records'] = [item for item in records if item['record_id'] != omitted]
    cohort_path.write_text(json.dumps(cohort), 'utf-8')
    partial = runner().review(path, workspace_root=workspace)
    coverage = json.loads((partial / 'hierarchy.json').read_text('utf-8'))['coverage']
    assert any(c['status'] == 'incomplete' for c in coverage)
    # A valid workbook cannot be substituted without freezing a new cohort.
    cohort['snapshot_manifest_sha256'] = 'f' * 64
    cohort_path.write_text(json.dumps(cohort), 'utf-8')
    with pytest.raises(ValueError, match='cohort_snapshot_binding'):
        runner().review(path, workspace_root=workspace)
