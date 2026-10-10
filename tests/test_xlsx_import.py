"""All workbook contents below are invented; no competition fixtures."""
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pyarrow.parquet as pq
import pytest
from forestcheck.ingest.xlsx_import import WorkbookMapping, import_workbook
from pydantic import ValidationError

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def make_book(path, *, duplicate_parent=False, orphan=False, formula=False):
    from xml.sax.saxutils import escape
    headers = ['region', 'block', 'plot', 'height', 'height']
    rows = {
        'plots': [headers, ['01', '02', '003', '4', '5']],
        'layers': [headers, ['01', '02', '003', '6', '7'], ['01', '02', '003', '6', '7']],
        'regeneration': [headers, ['01', '02', '999' if orphan else '003', '0', '']],
    }
    if duplicate_parent:
        rows['plots'].append(['01', '02', '003', '8', '9'])
    with ZipFile(path, 'w', ZIP_DEFLATED) as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets>' + ''.join(
            f'<sheet name="{name}" sheetId="{i}" r:id="r{i}"/>' for i, name in enumerate(rows, 1)) + '</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(
            f'<Relationship Id="r{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 4)) + '</Relationships>')
        for i, data in enumerate(rows.values(), 1):
            xml = f'<worksheet xmlns="{NS}"><sheetData>'
            for rownum, row in enumerate(data, 1):
                xml += f'<row r="{rownum}">' + ''.join(f'<c r="{chr(65+c)}{rownum}" t="inlineStr"><is><t>{escape(v)}</t></is></c>' for c, v in enumerate(row)) + '</row>'
            if formula and i == 1:
                xml += '<row r="1048576"><c r="D1048576"><f>1+1</f><v>2</v></c></row>'
            z.writestr(f'xl/worksheets/sheet{i}.xml', xml + '</sheetData></worksheet>')


def mapping():
    return WorkbookMapping.model_validate({
        'schema_version': '1.0', 'mapping_version': 'invented-workbook-v1', 'source_format': 'xlsx',
        'sheets': [{'name': name, 'record_level': name, 'parent_sheet': None if name == 'plots' else 'plots',
                    'link_positions': [1, 2, 3], 'grain_basis': 'Synthetic test definition',
                    'columns': [{'position': i, 'source_header': header, 'field': f'column_{i:04d}',
                                 'unit': None if i < 4 else 'm', 'transform': 'text' if i < 4 else 'decimal_point',
                                 'key_role': {1: 'territory', 2: 'compartment', 3: 'stand'}.get(i),
                                 'basis': 'Synthetic test definition'}
                                for i, header in enumerate(['region', 'block', 'plot', 'height', 'height'], 1)]}
                   for name in ['plots', 'layers', 'regeneration']]})


def setup(tmp_path, **kwargs):
    source = tmp_path / 'source' / 'synthetic.xlsx'
    source.parent.mkdir()
    make_book(source, **kwargs)
    private = tmp_path / 'private'
    private.mkdir()
    return source, private


def run(source, private, m=None):
    return import_workbook(source, m or mapping(), private, workspace_root=private.parent / 'repo')


def test_levels_links_duplicates_positions_and_repeat(tmp_path):
    source, private = setup(tmp_path)
    before = source.read_bytes()
    result = run(source, private)
    report = json.loads((result / 'report.json').read_text())
    assert report['rows'] == {'plots': 1, 'layers': 2, 'regeneration': 1}
    assert report['links'] == {'matched': 3}
    assert report['excluded'] == 0
    assert report['diagnostics']['exact_rows'] == 1
    records = pq.read_table(result / 'sheet_0002.parquet').to_pylist()
    assert len(records) == 2 and records[0]['record_id'] != records[1]['record_id']
    record = json.loads(records[0]['record_json'])
    assert record['business_key']['stand'] == '003'
    assert record['fields'][3]['normalized_value'] == 6
    assert record['fields'][4]['normalized_value'] == 7
    assert record['fields'][3]['source']['column'] != record['fields'][4]['source']['column']
    assert run(source, private) == result
    assert source.read_bytes() == before


def test_ambiguous_parents_never_multiply_children(tmp_path):
    source, private = setup(tmp_path, duplicate_parent=True)
    result = run(source, private)
    report = json.loads((result / 'report.json').read_text())
    assert report['links'] == {'ambiguous_parent': 3}
    assert report['diagnostics']['conflicting_records'] == 1
    assert pq.read_table(result / 'sheet_0002.parquet').num_rows == 2
    links = [json.loads(line) for line in (result / 'relations.jsonl').read_text().splitlines()]
    assert all(link['parent_record_id'] is None for link in links)


def test_orphan_and_sparse_formula_preserved_without_execution(tmp_path):
    source, private = setup(tmp_path, orphan=True, formula=True)
    result = run(source, private)
    report = json.loads((result / 'report.json').read_text())
    assert report['quarantined'] == 1
    assert report['links'] == {'matched': 2, 'missing_parent': 1}
    q = json.loads((result / 'quarantine.jsonl').read_text())
    assert q['row'] == 1048576 and q['cells']['4']['formula'] == '1+1'
    record = json.loads(pq.read_table(result / 'sheet_0003.parquet')['record_json'][0].as_py())
    assert record['fields'][3]['normalized_value'] == 0
    assert record['fields'][4]['state'] == 'missing'


@pytest.mark.parametrize('update', [{'schema_version': '2.0'}, {'guess': True}])
def test_versions_and_unknown_fields_refused(update):
    with pytest.raises(ValidationError):
        WorkbookMapping.model_validate(mapping().model_dump() | update)


def test_different_headers_and_corrupt_snapshot_refused(tmp_path):
    source, private = setup(tmp_path)
    payload = mapping().model_dump()
    payload['sheets'][0]['columns'][0]['source_header'] = 'another schema'
    with pytest.raises(ValueError, match='header_mismatch'):
        run(source, private, WorkbookMapping.model_validate(payload))
    result = run(source, private)
    (result / 'relations.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        run(source, private)


@pytest.mark.parametrize('name,content', [('xl/vbaProject.bin', b'macro'), ('../outside', b'bad'), ('xl/externalLinks/externalLink1.xml', b'external')])
def test_active_or_unsafe_package_refused(tmp_path, name, content):
    source, private = setup(tmp_path)
    with ZipFile(source, 'a') as z:
        z.writestr(name, content)
    with pytest.raises(ValueError):
        run(source, private)


def test_separate_local_workbook_mapping_never_invents_region(tmp_path):
    source, private = setup(tmp_path)
    payload = mapping().model_dump()
    payload['mapping_version'] = 'invented-small-schema-v1'
    for sheet in payload['sheets']:
        sheet['link_scope'] = 'workbook_local'
        sheet['link_positions'] = (1,)
        for column in sheet['columns']:
            column['key_role'] = 'stand' if column['position'] == 1 else None
    result = run(source, private, WorkbookMapping.model_validate(payload))
    record = json.loads(pq.read_table(result / 'sheet_0001.parquet')['record_json'][0].as_py())
    assert record['business_key']['territory'] is None
    assert record['business_key']['compartment'] is None
    assert 'workbook_local_identity_not_global' in pq.read_table(result / 'sheet_0001.parquet')['blockers'][0].as_py()


def rewrite_member(path, member, transform):
    with ZipFile(path) as z:
        content = {n: z.read(n) for n in z.namelist()}
    content[member] = transform(content[member])
    with ZipFile(path, 'w', ZIP_DEFLATED) as z:
        for n, data in content.items():
            z.writestr(n, data)


def test_formula_in_mapped_field_is_not_cached_observation(tmp_path):
    source, private = setup(tmp_path)
    rewrite_member(source, 'xl/worksheets/sheet1.xml', lambda b: b.replace(
        b'<c r="D2" t="inlineStr"><is><t>4</t></is></c>', b'<c r="D2"><f>2+2</f><v>4</v></c>'))
    result = run(source, private)
    field = json.loads(pq.read_table(result / 'sheet_0001.parquet')['record_json'][0].as_py())['fields'][3]
    assert field['state'] == 'unknown' and field['normalized_value'] is None
    assert field['original_value'] == '=2+2'


@pytest.mark.parametrize('mutation', [
    lambda b: b.replace(b'<worksheet ', b'<!DOCTYPE worksheet [<!ENTITY x "oops">]><worksheet '),
    lambda b: b.replace(b'r="D2"', b'r="D3"'),
    lambda b: b.replace(b'r="E2"', b'r="D2"'),
])
def test_unsafe_xml_or_ambiguous_coordinates_refused(tmp_path, mutation):
    source, private = setup(tmp_path)
    rewrite_member(source, 'xl/worksheets/sheet1.xml', mutation)
    with pytest.raises(ValueError):
        run(source, private)


@pytest.mark.parametrize('update', [{'parent_sheet': 'layers'}, {'link_positions': (3,)}, {'link_positions': (True, 2, 3)}])
def test_invalid_hierarchy_and_keys_refused(update):
    payload = mapping().model_dump()
    payload['sheets'][0].update(update)
    with pytest.raises(ValidationError):
        WorkbookMapping.model_validate(payload)


def test_private_boundary_enforced(tmp_path):
    source, private = setup(tmp_path)
    with pytest.raises(ValueError, match='outside_workspace'):
        import_workbook(source, mapping(), private, workspace_root=tmp_path)


def test_zip_cli_selects_only_explicit_member_and_preserves_provenance(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('import_xlsx_local', Path('scripts/import_xlsx_local.py'))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source, private = setup(tmp_path)
    archive_path = source.with_suffix('.zip')
    with ZipFile(archive_path, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('sample.xlsx', source.read_bytes())
        archive.writestr('ignored.txt', 'not read as instructions')
    m = private / 'mapping.json'
    m.write_text(mapping().model_dump_json(), 'utf-8')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'source_dir': str(source.parent), 'private_dir': str(private),
        'imports': [{'source_file': archive_path.name, 'mapping_file': str(m), 'archive_member': 'sample.xlsx'}]}), 'utf-8')
    before = archive_path.read_bytes()
    paths = cli.run_batch(config, workspace_root=tmp_path / 'repo')
    assert len(paths) == 1 and (paths[0] / 'manifest.json').exists()
    assert len(list((private / 'xlsx_sources').glob('*.json'))) == 1
    assert cli.run_batch(config, workspace_root=tmp_path / 'repo') == paths
    assert archive_path.read_bytes() == before
