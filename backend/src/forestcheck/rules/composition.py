"""Lossless, explicitly declared composition syntax; no real forestry grammar shipped.

The only supported syntax is a deliberately small demonstration: optional ASCII
integer prefixes, uppercase ASCII codes, comma-separated elements, one optional
``+`` admixture group followed by one optional ``;`` additional group. Horizontal
spaces are retained. Unknown codes are preserved, not classified as row errors.

A caller must supply a verified, versioned grammar and its source locator. The
``verified`` flag records the caller's assertion, not independent source checking.
Synthetic references require an explicit opt-in. No coefficient total, forestry
codebook, Finding, reference loading, or source mutation is implemented here.
"""

import re
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictStr, model_validator

from forestcheck.domain.contracts import Contract, Text

Index = Annotated[int, Field(strict=True, ge=0)]
Digits = Annotated[str, Field(strict=True, pattern=r"^[0-9]+$")]
TokenKind = Literal["main", "admixture", "additional", "unknown"]
_ELEMENT = re.compile(r"[ \t]*([0-9]+)?([A-Z]+)[ \t]*")


class CompositionGrammar(Contract):
    """Declarative reference for the one supported syntax; never executable input."""

    reference_id: Text
    reference_version: Text
    source_locator: Text
    verified: StrictBool
    synthetic: StrictBool
    syntax: Literal["integer_prefix_ascii_code_v1"]
    known_codes: tuple[Text, ...]
    separator: Literal[","]
    admixture_marker: Literal["+"]
    additional_marker: Literal[";"]

    @model_validator(mode="after")
    def valid_codebook(self) -> Self:
        if not all(
            value.strip()
            for value in (self.reference_id, self.reference_version, self.source_locator)
        ):
            raise ValueError("composition_reference_evidence_required")
        if len(self.known_codes) != len(set(self.known_codes)):
            raise ValueError("duplicate_composition_code")
        if any(re.fullmatch(r"[A-Z]+", code) is None for code in self.known_codes):
            raise ValueError("unsupported_composition_code")
        return self


class CompositionToken(Contract):
    """Zero-based, end-exclusive character span, including its preceding delimiter.

    Coefficient is the exact digit lexeme, not an inferred share or normalized
    number. The raw span retains whitespace, markers, and unrecognized fragments.
    """

    start: Index
    end: Index
    raw: StrictStr
    coefficient: Digits | None
    code: StrictStr | None
    known: StrictBool
    kind: TokenKind


class CompositionResult(Contract):
    raw: StrictStr
    status: Literal["parsed", "unavailable_reference", "unrecognized_syntax"]
    reason: Literal[
        "parsed",
        "missing_reference",
        "unverified_reference",
        "synthetic_reference_disabled",
        "unrecognized_syntax",
    ]
    grammar: CompositionGrammar | None
    tokens: tuple[CompositionToken, ...]

    @model_validator(mode="after")
    def lossless_spans(self) -> Self:
        if self.status == "unavailable_reference":
            if self.tokens:
                raise ValueError("unavailable_reference_has_no_interpreted_tokens")
            return self
        if self.grammar is None or not self.grammar.verified:
            raise ValueError("interpreted_composition_requires_verified_reference")
        offset = 0
        for token in self.tokens:
            if (
                token.start != offset
                or token.end <= token.start
                or token.raw != self.raw[token.start : token.end]
                or token.end > len(self.raw)
            ):
                raise ValueError("invalid_composition_source_span")
            offset = token.end
        if offset != len(self.raw):
            raise ValueError("incomplete_composition_source_spans")
        if self.status == "parsed" and (
            not self.tokens or any(token.kind == "unknown" for token in self.tokens)
        ):
            raise ValueError("parsed_composition_requires_recognized_syntax")
        return self


def parse_composition(
    raw: str,
    grammar: CompositionGrammar | None = None,
    *,
    allow_synthetic: bool = False,
) -> CompositionResult:
    """Preserve all input; interpretation is conditional on an explicit reference."""
    if not isinstance(raw, str):
        raise TypeError("composition_raw_must_be_text")
    if type(allow_synthetic) is not bool:
        raise TypeError("allow_synthetic_must_be_boolean")
    if grammar is not None and not isinstance(grammar, CompositionGrammar):
        raise TypeError("composition_grammar_must_be_validated")

    reason = None
    if grammar is None:
        reason = "missing_reference"
    elif not grammar.verified:
        reason = "unverified_reference"
    elif grammar.synthetic and not allow_synthetic:
        reason = "synthetic_reference_disabled"
    if reason is not None:
        return CompositionResult(
            schema_version="1.0",
            raw=raw,
            status="unavailable_reference",
            reason=reason,
            grammar=grammar,
            tokens=(),
        )

    boundaries = [0, *(m.start() for m in re.finditer(r"[,;+]", raw) if m.start()), len(raw)]
    tokens = []
    kind: TokenKind = "main"
    known_codes = set(grammar.known_codes)
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            continue
        fragment = raw[start:end]
        body = fragment
        valid_marker = True
        if start:
            marker, body = fragment[0], fragment[1:]
            if marker == grammar.admixture_marker:
                valid_marker = kind == "main"
                if valid_marker:
                    kind = "admixture"
            elif marker == grammar.additional_marker:
                valid_marker = kind in ("main", "admixture")
                if valid_marker:
                    kind = "additional"
        match = _ELEMENT.fullmatch(body) if valid_marker else None
        coefficient, code = match.groups() if match else (None, None)
        tokens.append(
            CompositionToken(
                schema_version="1.0",
                start=start,
                end=end,
                raw=fragment,
                coefficient=coefficient,
                code=code,
                known=code in known_codes,
                kind=kind if match else "unknown",
            )
        )
    status = (
        "parsed" if tokens and all(t.kind != "unknown" for t in tokens) else "unrecognized_syntax"
    )
    return CompositionResult(
        schema_version="1.0",
        raw=raw,
        status=status,
        reason=status,
        grammar=grammar,
        tokens=tuple(tokens),
    )
