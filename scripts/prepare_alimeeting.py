"""Convert AliMeeting (Praat TextGrid) annotations into the project format.

For each meeting this produces:

* speaker turns ``{meeting_id, speaker, start_time, end_time}``; and
* ground-truth overlap regions — spans where two or more speakers are active at
  once — used as the reference labels for Phase 2 / Step 7b overlap-threshold
  calibration.

The TextGrid parser targets the standard Praat ``ooTextFile`` text format: each
speaker is an ``IntervalTier`` whose non-empty intervals mark that speaker
talking. This is the format AliMeeting ships, but **validate the output against a
real ``Eval_Ali`` file** before trusting a full run — the pure functions here are
unit-tested only against synthetic fixtures (no audio required).

Usage::

    export ALIMEETING_ROOT=/path/to/Eval_Ali
    python scripts/prepare_alimeeting.py
    # or: python scripts/prepare_alimeeting.py --root /path/to/Eval_Ali --out data/alimeeting/annotations
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

Turn = dict[str, Any]
Region = tuple[float, float]

_INTERVAL_TIER_RE = re.compile(r'class\s*=\s*"IntervalTier"')
_NAME_RE = re.compile(r'name\s*=\s*"([^"]*)"')
_INTERVAL_RE = re.compile(
    r"xmin\s*=\s*([0-9.]+)\s*"
    r"xmax\s*=\s*([0-9.]+)\s*"
    r'text\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def parse_textgrid(text: str) -> dict[str, list[Region]]:
    """Parse a Praat TextGrid into ``{speaker_tier: [(start, end), ...]}``.

    Only ``IntervalTier`` tiers are read; only intervals with non-empty text are
    kept (empty text means silence for that speaker). The tier name is used as
    the speaker label.
    """
    tiers: dict[str, list[Region]] = {}
    # Split into per-tier blocks at each "IntervalTier" marker so tier names and
    # their intervals stay associated.
    markers = [m.start() for m in _INTERVAL_TIER_RE.finditer(text)]
    for index, start in enumerate(markers):
        end = markers[index + 1] if index + 1 < len(markers) else len(text)
        block = text[start:end]
        name_match = _NAME_RE.search(block)
        speaker = name_match.group(1).strip() if name_match else f"tier_{index}"
        regions: list[Region] = []
        for interval in _INTERVAL_RE.finditer(block):
            xmin, xmax, content = interval.groups()
            if content.strip():
                regions.append((float(xmin), float(xmax)))
        if regions:
            tiers.setdefault(speaker, []).extend(regions)
    return tiers


def textgrid_to_turns(text: str, meeting_id: str) -> list[Turn]:
    """Flatten a TextGrid into time-sorted speaker turns in the project format."""
    turns: list[Turn] = []
    for speaker, regions in parse_textgrid(text).items():
        for start, end in regions:
            if end > start:
                turns.append({
                    "meeting_id": meeting_id,
                    "speaker": speaker,
                    "start_time": round(float(start), 3),
                    "end_time": round(float(end), 3),
                })
    turns.sort(key=lambda turn: (turn["start_time"], turn["end_time"]))
    return turns


def overlap_regions_from_turns(turns: list[Turn]) -> list[Region]:
    """Return spans where >= 2 **distinct speakers** are simultaneously active.

    Activity is tracked per speaker, so a single speaker with duplicate or
    self-overlapping turns never counts as multi-speaker overlap.
    """
    events: list[tuple[float, int, str]] = []
    for turn in turns:
        start = float(turn["start_time"])
        end = float(turn["end_time"])
        if end > start:
            speaker = str(turn.get("speaker", ""))
            events.append((start, 1, speaker))
            events.append((end, -1, speaker))
    if not events:
        return []

    # Process starts (+1) before ends (-1) at the same timestamp so a turn that
    # begins exactly when another ends does not register as overlap.
    events.sort(key=lambda item: (item[0], -item[1]))
    active: dict[str, int] = {}

    def num_speakers() -> int:
        return sum(1 for count in active.values() if count > 0)

    regions: list[Region] = []
    region_start: float | None = None
    previous = events[0][0]
    for timestamp, delta, speaker in events:
        if num_speakers() >= 2 and timestamp > previous and region_start is not None:
            regions.append((region_start, timestamp))
            region_start = None
        active[speaker] = active.get(speaker, 0) + delta
        if num_speakers() >= 2 and region_start is None:
            region_start = timestamp
        elif num_speakers() < 2:
            region_start = None
        previous = timestamp
    return _merge_regions(regions)


def _merge_regions(regions: list[Region]) -> list[Region]:
    """Merge touching/overlapping regions into a minimal disjoint set."""
    ordered = sorted((float(s), float(e)) for s, e in regions if e > s)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def convert_meeting(text: str, meeting_id: str) -> dict[str, Any]:
    """Build the prepared annotation record for one meeting."""
    turns = textgrid_to_turns(text, meeting_id)
    regions = overlap_regions_from_turns(turns)
    total_overlap = round(sum(end - start for start, end in regions), 3)
    return {
        "meeting_id": meeting_id,
        "num_speakers": len({turn["speaker"] for turn in turns}),
        "turns": turns,
        "overlap_regions": [{"start_time": round(s, 3), "end_time": round(e, 3)} for s, e in regions],
        "overlap_seconds": total_overlap,
    }


def _meeting_id_from_path(path: Path) -> str:
    return path.stem


def prepare_root(root: Path, out_dir: Path) -> list[Path]:
    """Convert every ``*.TextGrid`` under ``root`` and write one JSON per meeting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    textgrids = sorted(root.rglob("*.TextGrid"))
    if not textgrids:
        raise FileNotFoundError(f"no .TextGrid files found under {root}")
    seen: dict[str, Path] = {}
    for textgrid in textgrids:
        meeting_id = _meeting_id_from_path(textgrid)
        if meeting_id in seen:
            raise ValueError(
                f"duplicate meeting_id '{meeting_id}' from {seen[meeting_id]} and {textgrid}; "
                "point --root at a single TextGrid directory (e.g. Eval_Ali_far/textgrid_dir) "
                "so far/near copies do not silently overwrite each other"
            )
        seen[meeting_id] = textgrid
        record = convert_meeting(textgrid.read_text(encoding="utf-8"), meeting_id)
        out_path = out_dir / f"{meeting_id}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_path)
        print(f"{meeting_id}: {len(record['turns'])} turns, "
              f"{len(record['overlap_regions'])} overlap regions, "
              f"{record['overlap_seconds']}s overlap -> {out_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("ALIMEETING_ROOT"),
        help="AliMeeting Eval root (defaults to $ALIMEETING_ROOT)",
    )
    parser.add_argument(
        "--out",
        default="data/alimeeting/annotations",
        help="output directory for prepared per-meeting JSON",
    )
    args = parser.parse_args()
    if not args.root:
        parser.error("set ALIMEETING_ROOT or pass --root")

    written = prepare_root(Path(args.root), Path(args.out))
    print(f"\nprepared {len(written)} meeting annotation file(s) in {args.out}")


if __name__ == "__main__":
    main()
