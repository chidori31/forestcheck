# Ingest

Реализован позиционный CSV-импорт: inspect_source, validate_mapping,
normalize_dataset и строгий ImportMapping v1.0. PyArrow устанавливается через
extra analytics. CSV-загрузка подключена к существующему API.

Структурированный XLSX-адаптер: xlsx_import.py, xlsx_reader.py и
scripts/import_xlsx_local.py. Листы остаются отдельными уровнями, parent-child
связи хранятся отдельно, повторения диагностируются без удаления. Позиционные
поля с одинаковыми заголовками не объединяются. Подробности — docs/xlsx_import.md.
XLSX-профили ещё не подключены к API/UI; предметные неоднозначности блокируют анализ.
