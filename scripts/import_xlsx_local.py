"""Explicit local workbook import. stdout contains status only, never paths/rows."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from forestcheck.ingest.csv_import import canonical, checked_private, regular_output, sha256
from forestcheck.ingest.xlsx_import import WorkbookMapping, import_workbook
from forestcheck.ingest.xlsx_reader import check_archive


def run_batch(config_path: Path, *, workspace_root: Path):
    config = json.loads(config_path.read_text('utf-8-sig'))
    if set(config) != {'source_dir', 'private_dir', 'imports'} or not config['imports']:
        raise ValueError('invalid_config')
    source_root, private = Path(config['source_dir']), Path(config['private_dir'])
    results = []
    for item in config['imports']:
        if set(item) not in ({'source_file', 'mapping_file'}, {'source_file', 'mapping_file', 'archive_member'}):
            raise ValueError('invalid_import_entry')
        name = item['source_file']
        if not isinstance(name, str) or not name or any(c in name for c in '/\\:') or name in ('.', '..'):
            raise ValueError('source_basename_required')
        source = source_root / name
        private = checked_private(private, source, workspace_root)
        mapping_path = Path(item['mapping_file'])
        if not mapping_path.is_absolute() or not mapping_path.resolve().is_relative_to(private):
            raise ValueError('mapping_must_be_private')
        mapping = WorkbookMapping.model_validate_json(mapping_path.read_text('utf-8'))
        container_hash = None
        if 'archive_member' in item:
            if source.suffix.lower() != '.zip' or not regular_output(source) or source.stat().st_size > 128 * 1024**2:
                raise ValueError('invalid_source_archive')
            for ancestor in (source, *source.parents):
                if ancestor.is_symlink() or getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400:
                    raise ValueError('source_link_forbidden')
            container_hash = sha256(source)
            with ZipFile(source) as archive:
                check_archive(archive)
                member = item['archive_member']
                if not member.lower().endswith('.xlsx') or archive.getinfo(member).file_size > 128 * 1024**2:
                    raise ValueError('selected_member_limit_or_type')
                data = archive.read(member)
            digest = hashlib.sha256(data).hexdigest()
            incoming = private / 'xlsx_sources'
            incoming.mkdir(exist_ok=True)
            if incoming.is_symlink() or getattr(incoming.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError('output_link_forbidden')
            extracted = incoming / (digest + '.xlsx')
            if extracted.exists():
                if not regular_output(extracted) or sha256(extracted) != digest:
                    raise ValueError('extracted_source_integrity')
            else:
                with extracted.open('xb') as stream:
                    stream.write(data)
            provenance = incoming / (digest + '-' + container_hash + '.json')
            receipt = canonical({'source_archive_sha256': container_hash, 'member': member, 'member_sha256': digest})
            if provenance.exists():
                if not regular_output(provenance) or provenance.read_text('utf-8') != receipt:
                    raise ValueError('provenance_integrity')
            else:
                with provenance.open('x', encoding='utf-8') as stream:
                    stream.write(receipt)
            input_path = extracted
        else:
            input_path = source
        target = private / 'xlsx_snapshots'
        target.mkdir(exist_ok=True)
        results.append(import_workbook(input_path, mapping, target, workspace_root=workspace_root))
        if container_hash is not None and sha256(source) != container_hash:
            raise ValueError('source_archive_changed')
    return results


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid_arguments')


def main():
    try:
        parser = StatusParser(add_help=False)
        parser.add_argument('--config', required=True, type=Path)
        args = parser.parse_args()
        print('XLSX_IMPORT_STARTED', flush=True)
        run_batch(args.config, workspace_root=Path(__file__).resolve().parents[1])
    except KeyboardInterrupt:
        print('XLSX_IMPORT_INTERRUPTED')
        return 130
    except Exception:
        print('XLSX_IMPORT_FAILED')
        return 3
    print('XLSX_IMPORT_COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
