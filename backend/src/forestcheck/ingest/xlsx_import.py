"""Hierarchical workbook snapshot; no deduplication or mutation of CSV/model snapshots."""
import hashlib
import json
import sqlite3
import stat
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import Field, model_validator

from forestcheck.domain.contracts import (
    BusinessKey,
    Contract,
    FieldValue,
    PositiveInt,
    Record,
    SourceCoordinates,
    Text,
    ValueObject,
)
from forestcheck.ingest.csv_import import (
    canonical,
    checked_private,
    convert,
    parquet_schema,
    regular_output,
    sha256,
)
from forestcheck.ingest.mapping import ColumnMapping
from forestcheck.ingest.xlsx_reader import open_book, rows

VERSION = 'xlsx-import-1.0.1'


class SheetMapping(ValueObject):
    name: Text
    record_level: Text
    grain_basis: Text
    parent_sheet: Text | None = None
    link_scope: Literal['composite', 'workbook_local'] = 'composite'
    link_positions: Annotated[tuple[PositiveInt, ...], Field(min_length=1, max_length=3)]
    columns: Annotated[tuple[ColumnMapping, ...], Field(min_length=3, max_length=2048)]
    analysis_blockers: tuple[Text, ...] = ()

    @model_validator(mode='after')
    def positions(self) -> Self:
        if [c.position for c in self.columns] != list(range(1, len(self.columns) + 1)):
            raise ValueError('positions_must_be_complete_and_ordered')
        if len({c.field for c in self.columns}) != len(self.columns):
            raise ValueError('duplicate_internal_field')
        roles = [c.key_role for c in self.columns if c.key_role]
        if len(roles) != len(set(roles)):
            raise ValueError('duplicate_key_role')
        expected = ('territory', 'compartment', 'stand') if self.link_scope == 'composite' else ('stand',)
        if len(set(self.link_positions)) != len(expected) or any(type(p) is not int or p < 1 or p > len(self.columns) for p in self.link_positions):
            raise ValueError('invalid_link_positions')
        if tuple(self.columns[p - 1].key_role for p in self.link_positions) != expected:
            raise ValueError('composite_parent_key_required')
        return self


class WorkbookMapping(Contract):
    mapping_version: Text
    source_format: Literal['xlsx']
    sheets: Annotated[tuple[SheetMapping, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode='after')
    def hierarchy(self) -> Self:
        names = {s.name: s for s in self.sheets}
        if len(names) != len(self.sheets):
            raise ValueError('duplicate_sheet')
        for sheet in self.sheets:
            if sheet.parent_sheet is not None:
                parent = names.get(sheet.parent_sheet)
                if parent is None or parent.parent_sheet is not None or parent.name == sheet.name:
                    raise ValueError('invalid_parent_sheet')
                if parent.link_scope != sheet.link_scope:
                    raise ValueError('incompatible_link_scope')
        return self


def checked_workbook(source):
    source = Path(source)
    if not source.is_absolute() or str(source).startswith(('\\\\', '//')) or source.suffix.lower() != '.xlsx':
        raise ValueError('absolute_local_xlsx_required')
    for item in (source, *source.parents):
        s = item.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s, 'st_file_attributes', 0) & 0x400:
            raise ValueError('source_link_forbidden')
    if not regular_output(source) or source.stat().st_size > 512 * 1024**2:
        raise ValueError('source_type_or_size_limit')
    return source.resolve(strict=True)


def emit(stream, item):
    stream.write(canonical(item) + '\n')


def build_relations_and_diagnostics(db, mapping, staging):
    counts, statuses = Counter(), Counter()
    db.execute('CREATE INDEX parent_lookup ON rows(sheet,parent_key)')
    db.execute('CREATE INDEX raw_lookup ON rows(sheet,fingerprint)')
    db.execute('CREATE INDEX candidate_lookup ON rows(sheet,candidate_key)')
    with (staging / 'relations.jsonl').open('w', encoding='utf-8') as links, (staging / 'diagnostics.jsonl').open('w', encoding='utf-8') as diagnostic:
        def report(category, **values):
            counts[category] += 1
            emit(diagnostic, {'category': category, 'action': 'review_only_no_exclusion', **values})
        for sheet in mapping.sheets:
            for fingerprint, number in db.execute('SELECT fingerprint, COUNT(*) FROM rows WHERE sheet=? GROUP BY fingerprint HAVING COUNT(*)>1', (sheet.name,)):
                ids = [r[0] for r in db.execute('SELECT id FROM rows WHERE sheet=? AND fingerprint=? ORDER BY ordinal', (sheet.name, fingerprint))]
                report('exact_rows', sheet=sheet.name, record_ids=ids, count=number, meaning='Equal source cells; same event is unconfirmed')
            for key, number, variants in db.execute('SELECT candidate_key, COUNT(*), COUNT(DISTINCT fingerprint) FROM rows WHERE sheet=? GROUP BY candidate_key HAVING COUNT(*)>1', (sheet.name,)):
                ids = [r[0] for r in db.execute('SELECT id FROM rows WHERE sheet=? AND candidate_key=? ORDER BY ordinal', (sheet.name, key))]
                report('candidate_identifier_collisions', sheet=sheet.name, record_ids=ids, count=number)
                report('ambiguous_repetitions', sheet=sheet.name, record_ids=ids, meaning='Grain/time/element identity not established')
                if variants > 1 and sheet.parent_sheet is None:
                    report('conflicting_records', sheet=sheet.name, record_ids=ids, meaning='Candidate parent has different representations; no record declared wrong')
            if sheet.parent_sheet:
                for record_id, key in db.execute('SELECT id,parent_key FROM rows WHERE sheet=? ORDER BY ordinal', (sheet.name,)):
                    matches = db.execute('SELECT id FROM rows WHERE sheet=? AND parent_key=? ORDER BY ordinal LIMIT 2', (sheet.parent_sheet, key)).fetchall()
                    status = 'matched' if len(matches) == 1 else 'missing_parent' if not matches else 'ambiguous_parent'
                    statuses[status] += 1
                    emit(links, {'child_record_id': record_id, 'child_sheet': sheet.name, 'parent_sheet': sheet.parent_sheet,
                                 'parent_record_id': matches[0][0] if len(matches) == 1 else None, 'status': status})
                for key, number in db.execute('SELECT parent_key,COUNT(*) FROM rows WHERE sheet=? GROUP BY parent_key', (sheet.name,)):
                    if db.execute('SELECT COUNT(*) FROM rows WHERE sheet=? AND parent_key=?', (sheet.parent_sheet, key)).fetchone()[0] == 1:
                        report('allowed_cross_level_repetitions', sheet=sheet.name, parent_sheet=sheet.parent_sheet,
                               parent_key_hash=hashlib.sha256(key.encode()).hexdigest(), child_rows=number,
                               meaning='Expected parent-child multiplicity, not duplicate removal or biological validation')
    return dict(counts), dict(statuses)


def import_workbook(source: Path, mapping: WorkbookMapping, private: Path, *, workspace_root: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = checked_workbook(source)
    private = checked_private(private, source, workspace_root)
    source_hash = sha256(source)
    mapping_text = canonical(mapping.model_dump(mode='json'))
    mapping_hash = hashlib.sha256(mapping_text.encode()).hexdigest()
    dataset_id = uuid5(NAMESPACE_URL, f'{VERSION}:{source_hash}:{mapping_hash}')
    artifact_id = uuid5(NAMESPACE_URL, 'sha256:' + source_hash)
    parent = private / 'workbook_imports'
    parent.mkdir(exist_ok=True)
    if parent.is_symlink() or getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('output_link_forbidden')
    destination = parent / str(dataset_id)
    files = {'mapping.json', 'report.json', 'cells.jsonl', 'quarantine.jsonl', 'relations.jsonl', 'diagnostics.jsonl', 'metadata.json'}
    files.update(f'sheet_{i:04d}.parquet' for i in range(1, len(mapping.sheets) + 1))
    if destination.exists():
        if destination.is_symlink() or getattr(destination.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('existing_output_integrity')
        manifest_path = destination / 'manifest.json'
        if not regular_output(manifest_path) or manifest_path.stat().st_size > 1024**2:
            raise ValueError('existing_output_integrity')
        m = json.loads(manifest_path.read_text('utf-8'))
        if (m.get('source_sha256') != source_hash or m.get('mapping_sha256') != mapping_hash
                or m.get('pipeline_version') != VERSION or m.get('schema_version') != '1.0'
                or m.get('dataset_id') != str(dataset_id) or m.get('source_sha256_after') != source_hash
                or set(m.get('output_hashes', {})) != files):
            raise ValueError('existing_output_integrity')
        for name, digest in m['output_hashes'].items():
            if not regular_output(destination / name) or sha256(destination / name) != digest:
                raise ValueError('existing_output_integrity')
        return destination
    staging = parent / ('.incomplete-' + uuid4().hex)
    staging.mkdir(mode=0o700)
    (staging / 'mapping.json').write_text(mapping_text, encoding='utf-8')
    totals, physical, quarantined = {}, 0, 0
    schema = parquet_schema()
    with (open_book(source) as (archive, sheets, strings, metadata),
          sqlite3.connect(staging / 'index.sqlite') as db,
          (staging / 'cells.jsonl').open('w', encoding='utf-8') as cells_out,
          (staging / 'quarantine.jsonl').open('w', encoding='utf-8') as quarantine):
        if set(sheets) != {s.name for s in mapping.sheets}:
            raise ValueError('sheet_mapping_mismatch')
        (staging / 'metadata.json').write_text(canonical(metadata), encoding='utf-8')
        db.execute('CREATE TABLE rows(id TEXT PRIMARY KEY,sheet TEXT,ordinal INTEGER,parent_key TEXT,candidate_key TEXT,fingerprint TEXT)')
        for index, sheet in enumerate(mapping.sheets, 1):
            iterator = rows(archive, sheets[sheet.name], strings)
            header_row, headers = next(iterator, (None, {}))
            if (header_row != 1 or set(headers) != set(range(1, len(sheet.columns) + 1))
                    or any(headers[c.position]['value'] != c.source_header or headers[c.position]['formula'] is not None for c in sheet.columns)):
                raise ValueError('header_mismatch')
            accepted, batch = 0, []
            with pq.ParquetWriter(staging / f'sheet_{index:04d}.parquet', schema, compression='zstd') as writer:
                for ordinal, (number, cells) in enumerate(iterator, 1):
                    physical += 1
                    record_id = uuid5(dataset_id, f'{sheet.name}:{number}')
                    provenance = {'record_id': str(record_id), 'sheet': sheet.name, 'row': number, 'cells': cells}
                    if any(p > len(sheet.columns) for p in cells):
                        raise ValueError('unmapped_column')
                    fields, analytics, keys = [], [], {}
                    blockers = list(sheet.analysis_blockers)
                    if sheet.link_scope == 'workbook_local':
                        blockers.append('workbook_local_identity_not_global')
                    for c in sheet.columns:
                        cell = cells.get(c.position, {'value': '', 'formula': None, 'type': 'n'})
                        raw = cell['value']
                        value, state, history = convert(raw, c)
                        if cell['formula'] is not None or cell['type'] == 'e':
                            value, state, history = None, 'unknown', ['formula_or_excel_error_not_evaluated']
                            blockers.append('formula_or_excel_error:' + c.field)
                        if c.transform == 'unknown' or state == 'parse_error':
                            blockers.append('unresolved:' + c.field)
                        if c.key_role and state == 'observed' and value != '':
                            keys[c.key_role] = value
                        original = '=' + cell['formula'] if cell['formula'] is not None else raw
                        fields.append(FieldValue(schema_version='1.0', name=c.field, original_value=original,
                            normalized_value=value, unit=c.unit, state=state,
                            source=SourceCoordinates(artifact_id=artifact_id, table=sheet.name, row=number, column=f'column_{c.position:04d}')))
                        analytics.append({'name': c.field, 'original_value': original, 'normalized_string': value if isinstance(value, str) else None,
                            'normalized_integer': value if type(value) is int else None, 'normalized_number': value if type(value) is float else None,
                            'state': state, 'unit': c.unit, 'column_position': c.position, 'transformation_history': history})
                    record = Record(schema_version='1.0', id=record_id, dataset_id=dataset_id,
                                    business_key=BusinessKey(**keys) if 'stand' in keys else None, fields=tuple(fields))
                    link_roles = tuple(sheet.columns[p - 1].key_role for p in sheet.link_positions)
                    if not all(role in keys for role in link_roles):
                        quarantined += 1
                        emit(quarantine, {**provenance, 'reason': 'incomplete_parent_identity', 'record': record.model_dump(mode='json')})
                        continue
                    emit(cells_out, provenance)
                    key = canonical([keys[r] for r in link_roles])
                    # Exclude style/shared-string index from value equality, retain types and formulas.
                    signature = [{k: cells.get(c.position, {}).get(k, '' if k == 'value' else None)
                                  for k in ('value', 'type', 'formula')} for c in sheet.columns]
                    fingerprint = hashlib.sha256(canonical(signature).encode()).hexdigest()
                    db.execute('INSERT INTO rows VALUES(?,?,?,?,?,?)', (str(record_id), sheet.name, number, key, canonical(keys), fingerprint))
                    accepted += 1
                    batch.append({'record_id': str(record_id), 'dataset_id': str(dataset_id), 'record_level': sheet.record_level,
                        'source_record_ordinal': ordinal, 'source_line_start': number, 'source_line_end': number,
                        'fields': analytics, 'record_json': record.model_dump_json(), 'analysis_blocked': bool(blockers), 'blockers': sorted(set(blockers))})
                    if len(batch) >= 500:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            totals[sheet.name] = accepted
        diagnostics, links = build_relations_and_diagnostics(db, mapping, staging)
    db.close()
    (staging / 'index.sqlite').unlink()
    after = sha256(checked_workbook(source))
    if after != source_hash:
        raise ValueError('source_changed_during_import')
    if physical != sum(totals.values()) + quarantined:
        raise ValueError('row_balance')
    report = {'schema_version': '1.0', 'rows': totals, 'physical_data_rows': physical, 'quarantined': quarantined,
              'excluded': 0, 'links': links, 'diagnostics': diagnostics, 'source_unchanged': True,
              'row_expansion': 'none', 'analysis_admission': 'requires_explicit_profile_not_automatic'}
    (staging / 'report.json').write_text(canonical(report), encoding='utf-8')
    manifest = {'schema_version': '1.0', 'pipeline_version': VERSION, 'dataset_id': str(dataset_id),
        'source_sha256': source_hash, 'source_sha256_after': after, 'mapping_sha256': mapping_hash,
        'output_hashes': {name: sha256(staging / name) for name in sorted(files)}}
    (staging / 'manifest.json').write_text(canonical(manifest), encoding='utf-8')
    staging.rename(destination)
    return destination
