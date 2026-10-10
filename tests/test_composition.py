"""Invented composition syntax only; no forestry reference or dataset access."""

import importlib
import importlib.util

import pytest
from pydantic import ValidationError


def composition_module():
    name = "forestcheck.rules.composition"
    assert importlib.util.find_spec(name) is not None, "composition parser is not implemented"
    return importlib.import_module(name)


def grammar_payload(**changes):
    return {
        "schema_version": "1.0",
        "reference_id": "synthetic-composition-demo",
        "reference_version": "demo-1",
        "source_locator": "synthetic://test_composition/declared-demo-1",
        "verified": True,
        "synthetic": True,
        "syntax": "integer_prefix_ascii_code_v1",
        "known_codes": ("AA", "BB", "CC", "DD"),
        "separator": ",",
        "admixture_marker": "+",
        "additional_marker": ";",
        **changes,
    }


def synthetic_grammar(**changes):
    return composition_module().CompositionGrammar.model_validate(grammar_payload(**changes))


@pytest.mark.parametrize("raw", [" 03AA?+ZZ;BB ", "", "0", "неизвестный состав"])
def test_missing_grammar_preserves_raw_and_refuses_to_infer(raw):
    result = composition_module().parse_composition(raw)
    assert result.status == "unavailable_reference"
    assert result.raw == raw
    assert result.tokens == ()
    assert result.grammar is None


@pytest.mark.parametrize(
    ("changes", "allow_synthetic", "reason"),
    [
        ({}, False, "synthetic_reference_disabled"),
        ({"verified": False}, True, "unverified_reference"),
        ({"verified": False, "synthetic": False}, False, "unverified_reference"),
    ],
)
def test_unusable_reference_is_reported_instead_of_success(changes, allow_synthetic, reason):
    grammar = synthetic_grammar(**changes)
    result = composition_module().parse_composition("03AA+BB", grammar,
                                                   allow_synthetic=allow_synthetic)
    assert result.status == "unavailable_reference"
    assert result.reason == reason
    assert result.raw == "03AA+BB"
    assert result.tokens == ()
    assert result.grammar == grammar


def test_declared_synthetic_syntax_keeps_unknown_codes_markers_and_source_spans():
    grammar = synthetic_grammar()
    result = composition_module().parse_composition("03AA,0BB+ZZ;DD", grammar,
                                                   allow_synthetic=True)
    assert result.status == "parsed"
    assert result.grammar == grammar
    assert [(t.start, t.end, t.raw, t.coefficient, t.code, t.known, t.kind)
            for t in result.tokens] == [
        (0, 4, "03AA", "03", "AA", True, "main"),
        (4, 8, ",0BB", "0", "BB", True, "main"),
        (8, 11, "+ZZ", None, "ZZ", False, "admixture"),
        (11, 14, ";DD", None, "DD", True, "additional"),
    ]
    assert "".join(token.raw for token in result.tokens) == result.raw
    assert result.schema_version == "1.0"


def test_coefficients_have_no_universal_target_or_leading_zero_normalization():
    raw = "0000AA,237BB"
    result = composition_module().parse_composition(raw, synthetic_grammar(), allow_synthetic=True)
    assert result.status == "parsed"
    assert [token.coefficient for token in result.tokens] == ["0000", "237"]
    assert result.raw == raw


@pytest.mark.parametrize("raw", ["2AA??+BB", "2AA/8BB", "2AA,,BB", "+BB", "2AA;DD+CC", ""])
def test_unrecognized_syntax_remains_lossless_and_is_not_success(raw):
    result = composition_module().parse_composition(raw, synthetic_grammar(), allow_synthetic=True)
    assert result.status == "unrecognized_syntax"
    assert result.raw == raw
    assert "".join(token.raw for token in result.tokens) == raw
    for token in result.tokens:
        assert raw[token.start:token.end] == token.raw
    if raw:
        assert any(token.kind == "unknown" for token in result.tokens)


def test_whitespace_and_unrecognized_fragment_are_preserved_at_exact_offsets():
    result = composition_module().parse_composition(" 2AA , ?? + CC ", synthetic_grammar(),
                                                   allow_synthetic=True)
    assert result.status == "unrecognized_syntax"
    assert [(t.start, t.end, t.raw, t.kind) for t in result.tokens] == [
        (0, 5, " 2AA ", "main"),
        (5, 10, ", ?? ", "unknown"),
        (10, 15, "+ CC ", "admixture"),
    ]


@pytest.mark.parametrize("version", [None, "2.0", "0.9", 1])
def test_grammar_rejects_incompatible_schema_versions(version):
    with pytest.raises(ValidationError):
        synthetic_grammar(schema_version=version)


@pytest.mark.parametrize(
    "changes",
    [
        {"source_locator": ""},
        {"source_locator": " \t"},
        {"reference_id": " \t"},
        {"reference_version": ""},
        {"verified": "true"},
        {"synthetic": 1},
        {"syntax": "arbitrary_regex"},
        {"separator": "/"},
        {"known_codes": ("AA", "AA")},
        {"known_codes": ("A+",)},
        {"coefficient_target": 10},
        {"regex": ".*"},
    ],
)
def test_grammar_rejects_undeclared_fields_and_unsupported_syntax(changes):
    with pytest.raises(ValidationError):
        synthetic_grammar(**changes)


def test_result_roundtrip_rejects_unknown_version_fields_and_broken_source_spans():
    module = composition_module()
    result = module.parse_composition("01AA", synthetic_grammar(), allow_synthetic=True)
    assert module.CompositionResult.model_validate_json(result.model_dump_json()) == result
    payload = result.model_dump()
    for change in ({"schema_version": "2.0"}, {"probability": 1}, {"raw": "different"}):
        with pytest.raises(ValidationError):
            module.CompositionResult.model_validate({**payload, **change})
    with pytest.raises(ValidationError):
        result.raw = "changed"


def test_synthetic_opt_in_must_be_a_boolean():
    with pytest.raises(TypeError):
        composition_module().parse_composition("2AA", synthetic_grammar(), allow_synthetic="false")
