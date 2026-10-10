# Rules

RuleEngine проверяет Record/FieldValue по версиям RuleDefinition и возвращает
RuleOutcome; Finding создаётся только при violation. Pure Python, без ML/API.
Локальный runner передаёт проверкам только зафиксированные development-записи;
подробные результаты остаются в private_dir. Документация — docs/rule_catalog.md.
Реальная грамматика состава и словари кодов не подтверждены; synthetic fixtures
не являются предметными нормами. Неизвестные основания не считаются passed.
