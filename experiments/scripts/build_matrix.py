"""Enumerate the 4D experiment matrix and emit a ``matrix.json`` manifest.

Axes:
    A. ASR backend          faster-whisper | funasr | whisperx
    B. Overlap / speaker    pyannote | energy_fallback
    C. LLM resolver         none | openai  (openai is wired to DeepSeek)
    D. Speech separation    none | sepformer

Cardinality: 3 × 2 × 2 × 2 = **24 main cells**.

A ``mock`` sanity cell (asr=mock, others=none) is appended for end-to-end
smoke tests; it is harmless because it bypasses every heavy backend.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "experiments" / "matrix.json"

# Keep in sync with aggregate.py:CELL_AXES
AXES: dict[str, list[str]] = {
    "asr": ["faster-whisper", "funasr", "whisperx"],
    "osd": ["pyannote", "energy_fallback"],
    "resolver": ["none", "openai"],
    "separation": ["none", "sepformer"],
}

# Cell defaults that aren't part of the 4-axis product.
DEFAULTS: dict[str, Any] = {
    "language": "zh",
    "asr_device": "cuda",
    "asr_compute_type": "float16",
    "faster_whisper_model": "small",
}


def _cell_id(cell: dict[str, Any]) -> str:
    return (
        f"asr={cell.get('asr', 'mock')}"
        f"_osd={cell.get('osd', 'pyannote')}"
        f"_resolver={cell.get('resolver', 'none')}"
        f"_sep={cell.get('separation', 'none')}"
    )


def build_cells(include_mock: bool = True) -> list[dict[str, Any]]:
    """Cartesian product over the four axes, with sensible defaults."""
    cells: list[dict[str, Any]] = []
    for combo in product(AXES["asr"], AXES["osd"], AXES["resolver"], AXES["separation"]):
        asr, osd, resolver, sep = combo
        cell = {
            "asr": asr,
            "osd": osd,
            "resolver": resolver,
            "separation": sep,
            **DEFAULTS,
        }
        cells.append(cell)
    if include_mock:
        cells.append({
            "asr": "mock",
            "osd": "pyannote",
            "resolver": "none",
            "separation": "none",
            "language": "und",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "faster_whisper_model": "small",
        })
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-mock", action="store_true", help="omit the mock sanity cell")
    args = ap.parse_args()

    cells = build_cells(include_mock=not args.no_mock)
    out = {
        "axes": AXES,
        "defaults": DEFAULTS,
        "cells": [{"cell_id": _cell_id(c), **c} for c in cells],
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(out['cells'])} cells → {args.out}")
    for c in out["cells"]:
        print(f"  {c['cell_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
