"""Parse and conservatively repair JSON emitted by an LLM."""

import json
import re
from collections.abc import Callable
from typing import Any


# -- string-boundary helpers -------------------------------------------------

def _string_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of double-quoted JSON string regions."""
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            start = i
            i += 1
            while i < len(text):
                if text[i] == '\\':
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            ranges.append((start, i))
        else:
            i += 1
    return ranges


def _replace_outside_strings(
    text: str, pattern: re.Pattern[str], repl: str | Callable[[re.Match[str]], str],
) -> str:
    """Apply *pattern* replacements only where they do not fall inside a quoted string."""
    ranges = _string_ranges(text)
    result: list[str] = []
    last_end = 0
    for match in pattern.finditer(text):
        if any(start <= match.start() < end for start, end in ranges):
            continue
        result.append(text[last_end:match.start()])
        if callable(repl):
            result.append(repl(match))
        else:
            result.append(match.expand(repl))
        last_end = match.end()
    result.append(text[last_end:])
    return ''.join(result)


# -- repair helpers ----------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """Strip Markdown fences and isolate the outermost JSON object or array."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    bracket_start = cleaned.find("[")
    if start < 0 or (0 <= bracket_start < start):
        start = bracket_start
    end = cleaned.rfind("}")
    bracket_end = cleaned.rfind("]")
    if end < start or bracket_end > end:
        end = bracket_end
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _repair_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]``."""
    return _replace_outside_strings(text, _TRAILING_COMMA_RE, r"\1")


_PYTHON_LITERAL_RE = re.compile(r"\b(True|False|None)\b")
_LITERAL_MAP = {"True": "true", "False": "false", "None": "null"}


def _repair_python_literals(text: str) -> str:
    """Replace Python capitalised literals with JSON lowercase ones."""
    return _replace_outside_strings(
        text, _PYTHON_LITERAL_RE, lambda m: _LITERAL_MAP[m.group(1)],
    )


_UNQUOTED_KEY_RE = re.compile(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:')


def _repair_unquoted_keys(text: str) -> str:
    """Add double quotes around bare-word object keys.

    Matches identifiers that appear immediately after ``{`` or ``,`` and
    are followed by a colon.  This is intentionally conservative: it will
    not fix every possible unquoted-key syntax, but it handles the common
    LLM output ``{key: value}`` pattern without touching quoted strings.
    """
    return _replace_outside_strings(text, _UNQUOTED_KEY_RE, r'"\1":')


def _ensure_dict(payload: Any) -> dict[str, Any]:
    """Raise if *payload* is not a JSON object."""
    if not isinstance(payload, dict):
        raise ValueError("LLM output must decode to a JSON object")
    return payload


# -- public entry point -------------------------------------------------------

def parse_or_repair_json(raw_output: str) -> dict[str, Any]:
    """Parse JSON, applying a cascade of conservative repairs.

    Repairs are attempted in order of safety:
    1. Direct parse after stripping Markdown fences.
    2. Trailing-comma removal.
    3. Unquoted-key repair + Python-literal normalisation.
    4. If all fail, raise ``ValueError`` with the original decoder message.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("LLM output must be a non-empty string")

    text = _extract_json_block(raw_output)

    # Attempt 1: direct parse
    try:
        return _ensure_dict(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Attempt 2: trailing commas
    try:
        return _ensure_dict(json.loads(_repair_trailing_commas(text)))
    except json.JSONDecodeError:
        pass

    # Attempt 3: trailing commas + unquoted keys + Python literals
    repaired = _repair_trailing_commas(text)
    repaired = _repair_unquoted_keys(repaired)
    repaired = _repair_python_literals(repaired)
    try:
        return _ensure_dict(json.loads(repaired))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc.msg}") from exc


__all__ = ["parse_or_repair_json"]
