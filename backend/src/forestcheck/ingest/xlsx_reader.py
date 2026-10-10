"""Bounded, read-only OOXML parser. No evaluation, extraction or external resources."""
import re
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
MAX_XML = 512 * 1024**2


class SafeStream:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.total, self.tail = stream, limit, 0, b''

    def read(self, size=-1):
        block = self.stream.read(min(size if size >= 0 else self.limit + 1, self.limit + 1))
        self.total += len(block)
        probe = (self.tail + block).upper()
        if self.total > self.limit or b'\x00' in probe or b'<!DOCTYPE' in probe or b'<!ENTITY' in probe:
            raise ValueError('unsafe_or_excessive_xml')
        self.tail = probe[-16:]
        return block


def xml(archive, name, limit=4 * 1024**2):
    if archive.getinfo(name).file_size > limit:
        raise ValueError('metadata_limit')
    with archive.open(name) as stream:
        return ET.fromstring(SafeStream(stream, limit).read())


def check_archive(archive):
    infos = archive.infolist()
    if len(infos) > 4096 or len({i.filename for i in infos}) != len(infos):
        raise ValueError('archive_entry_limit_or_duplicate')
    if sum(i.file_size for i in infos) > 1024**3:
        raise ValueError('archive_expansion_limit')
    for i in infos:
        name = i.filename
        p = PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name
                or i.flag_bits & 1 or i.file_size > MAX_XML
                or i.file_size > max(1, i.compress_size) * 1000
                or (i.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError('unsafe_archive_member')
        if any(token in name.lower() for token in ('vbaproject', 'externallinks/', 'embeddings/', 'activex/')):
            raise ValueError('active_workbook_unsupported')
        if name.endswith('.rels'):
            if any(r.get('TargetMode') == 'External' for r in xml(archive, name)):
                raise ValueError('external_relationship_unsupported')


@contextmanager
def open_book(source: Path):
    with ZipFile(source) as archive:
        check_archive(archive)
        book = xml(archive, 'xl/workbook.xml')
        targets = {r.attrib['Id']: r for r in xml(archive, 'xl/_rels/workbook.xml.rels')}
        sheets = {}
        for sheet in book.findall(f'{NS}sheets/{NS}sheet'):
            rel = targets[sheet.attrib[f'{REL}id']]
            target = rel.attrib['Target']
            member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            if '..' in PurePosixPath(member).parts or member not in archive.namelist():
                raise ValueError('invalid_worksheet_target')
            if sheet.attrib['name'] in sheets:
                raise ValueError('duplicate_sheet_name')
            sheets[sheet.attrib['name']] = member
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = xml(archive, 'xl/sharedStrings.xml', 64 * 1024**2)
            for item in root:
                strings.append(''.join(t.text or '' for t in item.iter(NS + 't')))
            if len(strings) > 1000000:
                raise ValueError('shared_strings_limit')
        metadata = {'workbook': ET.tostring(book, encoding='unicode'),
                    'styles': ET.tostring(xml(archive, 'xl/styles.xml'), encoding='unicode')
                    if 'xl/styles.xml' in archive.namelist() else None}
        yield archive, sheets, strings, metadata


def rows(archive, member, strings):
    with archive.open(member) as stream:
        parser = ET.iterparse(SafeStream(stream, MAX_XML), events=('start', 'end'))
        root = None
        previous, count = 0, 0
        for event, element in parser:
            if event == 'start' and element.tag == NS + 'sheetData':
                root = element
            if event != 'end' or element.tag != NS + 'row':
                continue
            number = int(element.attrib['r'])
            count += 1
            if number <= previous or number > 1048576 or count > 1048576:
                raise ValueError('invalid_row_order_or_limit')
            previous = number
            cells = {}
            for cell in element.findall(NS + 'c'):
                ref = re.fullmatch(r'([A-Z]{1,3})([1-9][0-9]*)', cell.attrib['r'])
                if ref is None or int(ref[2]) != number:
                    raise ValueError('invalid_cell_reference')
                position = 0
                for char in ref[1]:
                    position = position * 26 + ord(char) - 64
                if position > 2048 or position in cells:
                    raise ValueError('column_limit_or_duplicate')
                kind = cell.get('t', 'n')
                literal = cell.findtext(NS + 'v', '')
                formula = cell.find(NS + 'f')
                if kind == 's':
                    index = int(literal)
                    if not 0 <= index < len(strings):
                        raise ValueError('invalid_shared_string')
                    value = strings[index]
                elif kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.iter(NS + 't'))
                elif kind in ('n', 'b', 'e', 'str', 'd'):
                    value = literal
                else:
                    raise ValueError('unsupported_cell_type')
                if len(value) > 1024**2:
                    raise ValueError('cell_text_limit')
                cells[position] = {'value': value, 'type': kind, 'style': cell.get('s'),
                                   'literal': literal, 'formula': None if formula is None else (formula.text or ''),
                                   'formula_attributes': None if formula is None else formula.attrib}
            yield number, cells
            element.clear()
            if root is not None:
                root.clear()
