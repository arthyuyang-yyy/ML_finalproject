"""Pre-fill the annotation CSV from AliMeeting TextGrids (issue #58 B1 / #61).

Annotators must not start from a blank sheet. This script turns the corpus's
near-field per-speaker TextGrids (clean transcripts) into the pre-filled
``annotation_template.csv`` layout that ``build_annotation_set.py`` consumes — so
people only add the *semantic* columns (``event_type`` / ``content`` /
``owner`` / ``deadline``).

What it does that ``prepare_alimeeting.py`` does not:

1. **Merges per meeting.** Near-field ships one TextGrid per speaker
   (e.g. ``R8001_M8004_N_SPK8013.TextGrid``); files are grouped by the
   ``R<x>_M<y>`` meeting id and time-sorted into one segment sequence.
2. **Keeps the transcript text** (the parser in ``prepare_alimeeting`` drops it).
3. **口径 B clustering (issue #61).** Utterances from *different* speakers that
   fall inside the same ground-truth overlap region are merged into ONE shared
   ``segment_id`` with one row per candidate (``speaker``/``text`` per row), so a
   high-overlap segment keeps its multi-candidate structure. Single-speaker
   utterances each become their own low-overlap segment.
4. **Computes ``overlap_type``** (none / partial / full) from the fraction of a
   segment covered by overlap. The partial↔full cut and the minimum overlap to
   count at all are **provisional** (issue #61 待办③, not yet ratified) and are
   exposed as CLI flags.

The semantic columns are left blank on purpose — that is the human's job. Feed
the result to ``build_annotation_set.py`` once filled.

Usage::

    python scripts/prefill_annotation_csv.py --root /path/to/Eval_Ali_near/textgrid_dir
    python scripts/prefill_annotation_csv.py --root TGDIR --out-dir data/annotations/prefilled \\
        --full-threshold 0.5 --min-overlap-seconds 0.0
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_alimeeting import (  # noqa: E402
    _INTERVAL_RE,
    _INTERVAL_TIER_RE,
    _NAME_RE,
    overlap_regions_from_turns,
)

Region = tuple[float, float]
Utterance = dict[str, Any]

# Column order of data/annotations/annotation_template.csv (build_annotation_set
# reads these names; semantic columns are filled by annotators).
HEADER = [
    "meeting_id", "segment_id", "start_time", "end_time", "speaker", "text",
    "is_overlap", "overlap_type", "topic", "decision", "action_item",
    "event_type", "content", "owner", "deadline", "uncertainty_note",
]

_MEETING_ID_RE = re.compile(r"^(R\d+_M\d+)")


# --------------------------------------------------------------------------- #
# Pure functions (no I/O; unit-tested)
# --------------------------------------------------------------------------- #
def meeting_id_of(stem: str) -> str:
    """Meeting id from a TextGrid stem, stripping the mic/speaker suffix.

    ``R8001_M8004_N_SPK8013`` and ``R8001_M8004_MS801`` both map to
    ``R8001_M8004``; a stem that does not match the pattern is its own meeting.
    """
    match = _MEETING_ID_RE.match(stem)
    return match.group(1) if match else stem


def speaker_token_of(stem: str, meeting_id: str) -> str:
    """The per-file speaker tag, i.e. the stem with the meeting-id prefix removed.

    ``R8001_M8004_N_SPK8013`` -> ``N_SPK8013``; returns ``""`` when the stem *is*
    the meeting id (a single multi-speaker TextGrid for the whole meeting).
    """
    if stem == meeting_id:
        return ""
    return stem[len(meeting_id):].lstrip("_")


def parse_utterances(text: str, meeting_id: str, file_token: str = "") -> list[Utterance]:
    """Parse one TextGrid into utterances **keeping the transcript text**.

    Each non-empty interval of an ``IntervalTier`` becomes an utterance. The
    speaker is the tier name, except when the file holds a single speaker tier and
    a ``file_token`` is given (the near-field per-speaker case): then the token is
    the speaker id, which stays unique across that meeting's files.
    """
    tiers: list[tuple[str, list[tuple[float, float, str]]]] = []
    markers = [m.start() for m in _INTERVAL_TIER_RE.finditer(text)]
    for index, start in enumerate(markers):
        end = markers[index + 1] if index + 1 < len(markers) else len(text)
        block = text[start:end]
        name_match = _NAME_RE.search(block)
        name = name_match.group(1).strip() if name_match else f"tier_{index}"
        intervals = [
            (float(xmin), float(xmax), content.strip())
            for xmin, xmax, content in _INTERVAL_RE.findall(block)
            if content.strip()
        ]
        if intervals:
            tiers.append((name, intervals))

    single_speaker = len(tiers) == 1 and bool(file_token)
    utterances: list[Utterance] = []
    for name, intervals in tiers:
        speaker = file_token if single_speaker else name
        for xmin, xmax, content in intervals:
            if xmax > xmin:
                utterances.append({
                    "meeting_id": meeting_id,
                    "speaker": speaker,
                    "start_time": round(xmin, 3),
                    "end_time": round(xmax, 3),
                    "text": content,
                })
    return utterances


def _covered_seconds(start: float, end: float, regions: list[Region]) -> float:
    """Seconds of ``[start, end]`` covered by any overlap region."""
    total = 0.0
    for region_start, region_end in regions:
        lo = max(start, region_start)
        hi = min(end, region_end)
        if hi > lo:
            total += hi - lo
    return total


def overlap_type_of(coverage: float, full_threshold: float) -> str:
    """Bucket an overlap-coverage fraction into ``none`` / ``partial`` / ``full``."""
    if coverage <= 0.0:
        return "none"
    return "full" if coverage >= full_threshold else "partial"


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def cluster_segments(
    utterances: list[Utterance],
    full_threshold: float = 0.5,
    min_overlap_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Group utterances into segments under 口径 B and tag each ``overlap_type``.

    A segment is *high-overlap* when >= 2 distinct speakers share a ground-truth
    overlap region (one row per candidate); otherwise each utterance is its own
    *low-overlap* segment. Segments are returned time-sorted, without ids.
    """
    utts = sorted(utterances, key=lambda u: (u["start_time"], u["end_time"]))
    regions = overlap_regions_from_turns(utts)
    if min_overlap_seconds > 0:
        regions = [(s, e) for s, e in regions if e - s >= min_overlap_seconds]

    union = _UnionFind(len(regions))
    per_utt_regions: list[list[int]] = []
    for utt in utts:
        hit = [
            i for i, (rs, re_) in enumerate(regions)
            if min(utt["end_time"], re_) > max(utt["start_time"], rs)
        ]
        for other in hit[1:]:
            union.union(hit[0], other)
        per_utt_regions.append(hit)

    clusters: dict[int, list[Utterance]] = {}
    singles: list[Utterance] = []
    for utt, hit in zip(utts, per_utt_regions):
        if hit:
            clusters.setdefault(union.find(hit[0]), []).append(utt)
        else:
            singles.append(utt)

    segments: list[dict[str, Any]] = []
    for members in clusters.values():
        speakers = {m["speaker"] for m in members}
        if len(speakers) < 2:  # defensive: a lone speaker is not real overlap
            singles.extend(members)
            continue
        seg_start = min(m["start_time"] for m in members)
        seg_end = max(m["end_time"] for m in members)
        coverage = _covered_seconds(seg_start, seg_end, regions) / (seg_end - seg_start)
        segments.append({
            "start_time": seg_start,
            "end_time": seg_end,
            "is_overlap": True,
            "overlap_type": overlap_type_of(coverage, full_threshold),
            "rows": [
                {"speaker": m["speaker"], "text": m["text"]}
                for m in sorted(members, key=lambda m: (m["start_time"], m["speaker"]))
            ],
        })

    for utt in singles:
        segments.append({
            "start_time": utt["start_time"],
            "end_time": utt["end_time"],
            "is_overlap": False,
            "overlap_type": "none",
            "rows": [{"speaker": utt["speaker"], "text": utt["text"]}],
        })

    segments.sort(key=lambda s: (s["start_time"], s["end_time"]))
    return segments


def segments_to_csv_rows(meeting_id: str, segments: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flatten segments into template rows (one row per candidate; shared id)."""
    rows: list[dict[str, str]] = []
    for index, segment in enumerate(segments, start=1):
        segment_id = f"{meeting_id}_seg{index:04d}"
        for candidate in segment["rows"]:
            row = {key: "" for key in HEADER}
            row.update(
                meeting_id=meeting_id,
                segment_id=segment_id,
                start_time=f"{segment['start_time']:.3f}",
                end_time=f"{segment['end_time']:.3f}",
                speaker=candidate["speaker"],
                text=candidate["text"],
                is_overlap="True" if segment["is_overlap"] else "False",
                overlap_type=segment["overlap_type"],
            )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# I/O layer
# --------------------------------------------------------------------------- #
def read_meeting_utterances(root: Path) -> dict[str, list[Utterance]]:
    """Parse every ``*.TextGrid`` under ``root``, merged and grouped by meeting."""
    textgrids = sorted(root.rglob("*.TextGrid"))
    if not textgrids:
        raise FileNotFoundError(f"no .TextGrid files found under {root}")
    by_meeting: dict[str, list[Utterance]] = {}
    for textgrid in textgrids:
        stem = textgrid.stem
        meeting_id = meeting_id_of(stem)
        token = speaker_token_of(stem, meeting_id)
        utterances = parse_utterances(textgrid.read_text(encoding="utf-8"), meeting_id, token)
        by_meeting.setdefault(meeting_id, []).extend(utterances)
    return by_meeting


def write_meeting_csv(out_path: Path, rows: list[dict[str, str]]) -> None:
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path,
                        help="directory of AliMeeting TextGrids (near-field textgrid_dir)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "annotations" / "prefilled",
                        help="directory to write one <meeting_id>.csv per meeting")
    parser.add_argument("--full-threshold", type=float, default=0.5,
                        help="overlap-coverage at/above which overlap_type=full (provisional, issue #61)")
    parser.add_argument("--min-overlap-seconds", type=float, default=0.0,
                        help="ignore overlap regions shorter than this before clustering (provisional)")
    args = parser.parse_args(argv)

    by_meeting = read_meeting_utterances(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_segments = total_high = 0
    for meeting_id, utterances in sorted(by_meeting.items()):
        segments = cluster_segments(utterances, args.full_threshold, args.min_overlap_seconds)
        rows = segments_to_csv_rows(meeting_id, segments)
        out_path = args.out_dir / f"{meeting_id}.csv"
        write_meeting_csv(out_path, rows)
        high = sum(1 for s in segments if s["is_overlap"])
        total_segments += len(segments)
        total_high += high
        print(f"[ok] {meeting_id}: {len(segments)} segments ({high} high-overlap), "
              f"{len(rows)} rows -> {out_path}")
    print(f"\nsummary: {len(by_meeting)} meetings, {total_segments} segments "
          f"({total_high} high-overlap). Semantic columns left blank for annotators.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
