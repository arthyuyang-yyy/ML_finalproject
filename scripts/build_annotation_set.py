"""Convert a flat annotation CSV into the canonical evaluation-set JSON.

Annotators fill ``data/annotations/annotation_template.csv`` (one row per
segment) in Excel/WPS. This script assembles, **per meeting**, the nested
``{"evidence_segments": [...], "gold_events": [...]}`` document consumed by the
Step 16b event-extraction experiment (same shape as
``experiments/event_extraction/annotations.json``), and runs the project's own
``validate_meeting_events_document`` so the output is schema-legal **by
construction**: missing evidence IDs, an ``action_item`` whose ``owner`` is not
supported by the cited evidence, a high-overlap segment presented as a confident
decision, etc. are reported immediately instead of leaking into experiments.

Transcription, speaker, timing and overlap come from the corpus (e.g.
AliMeeting via ``scripts/prepare_alimeeting.py``) and are pre-filled into the
CSV upstream; annotators only add the semantic layer (``event_type`` and, for
action items, ``owner``/``deadline``). See ``docs/annotation_guide.zh-CN.md``.

Usage::

    python scripts/build_annotation_set.py --csv data/annotations/meeting_xxx.csv
    python scripts/build_annotation_set.py --csv labels.csv --out-dir data/annotations/built
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm.event_validator import validate_meeting_events_document  # noqa: E402

LOW_OVERLAP = "low_overlap_cluster"
HIGH_OVERLAP = "high_overlap_candidate"
ACTION_ITEM = "action_item"
UNCERTAINTY = "uncertainty"

_DEFAULT_CANDIDATE_CONFIDENCE = 0.4


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "t", "是"}


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read annotation rows, dropping ``#`` comment lines and blank rows."""
    text = csv_path.read_text(encoding="utf-8-sig")
    kept = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(kept)))
    rows: list[dict[str, str]] = []
    for raw in reader:
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k is not None}
        if row.get("meeting_id") and row.get("segment_id"):
            rows.append(row)
    return rows


def _group_by_segment(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["segment_id"], []).append(row)
    return groups


def _is_high_overlap(group: list[dict[str, str]]) -> bool:
    return any(
        _truthy(row.get("is_overlap", "")) or row.get("overlap_type", "").lower() in {"partial", "full"}
        for row in group
    )


def _build_segment(segment_id: str, group: list[dict[str, str]], high: bool) -> dict[str, Any]:
    head = group[0]
    meeting_id = head["meeting_id"]
    start_time = min(_to_float(r.get("start_time", ""), 0.0) for r in group)
    end_time = max(_to_float(r.get("end_time", ""), 0.0) for r in group)
    language = head.get("language", "") or "zh"
    overlap_score = _to_float(head.get("overlap_score", ""), 0.8 if high else 0.1)
    segment: dict[str, Any] = {
        "meeting_id": meeting_id,
        "segment_id": segment_id,
        "evidence_id": segment_id,
        "start_time": start_time,
        "end_time": end_time,
        "overlap_score": overlap_score,
        "audio_clip_path": head.get("audio_clip_path", ""),
        "source_audio_path": head.get("source_audio_path", ""),
        "language": language,
    }
    if high:
        candidates = []
        for index, row in enumerate(group, start=1):
            if not row.get("text"):
                continue
            candidates.append(
                {
                    "candidate_id": f"{segment_id}_c{index}",
                    "speaker": row.get("speaker", ""),
                    "text": row["text"],
                    "confidence": _to_float(row.get("confidence", ""), _DEFAULT_CANDIDATE_CONFIDENCE),
                    "uncertainty_note": "high-overlap",
                }
            )
        segment.update(
            speaker="MIXED",
            text="",
            processing_path=HIGH_OVERLAP,
            route_reason="overlap_score above threshold",
            asr_confidence=0.0,
            speaker_confidence=_to_float(head.get("speaker_confidence", ""), 0.35),
            candidates=candidates,
            uncertainty_note=head.get("uncertainty_note", "")
            or "High-overlap segment; speaker attribution is uncertain.",
        )
    else:
        segment.update(
            speaker=head.get("speaker", ""),
            text=head.get("text", ""),
            processing_path=LOW_OVERLAP,
            route_reason="overlap_score below threshold",
            asr_confidence=_to_float(head.get("asr_confidence", ""), 0.9),
            speaker_confidence=_to_float(head.get("speaker_confidence", ""), 0.9),
            candidates=[],
            uncertainty_note=head.get("uncertainty_note", ""),
        )
    return segment


def _build_gold_event(segment_id: str, group: list[dict[str, str]], high: bool) -> dict[str, Any] | None:
    head = group[0]
    event_type = head.get("event_type", "").strip()
    if high:
        # Hard rule: high-overlap evidence is always an uncertainty event.
        event_type = UNCERTAINTY
    if not event_type:
        return None
    content = head.get("content", "").strip() or head.get("text", "").strip()
    event: dict[str, Any] = {
        "event_type": event_type,
        "evidence_ids": [segment_id],
        "content": content,
    }
    if event_type == ACTION_ITEM:
        event["owner"] = head.get("owner", "").strip() or "uncertain"
        deadline = head.get("deadline", "").strip()
        if deadline:
            event["deadline"] = deadline
    return event


def _meeting_summary(segments: list[dict[str, Any]]) -> str:
    spoken = [s["text"].strip() for s in segments if s["processing_path"] == LOW_OVERLAP and s["text"].strip()]
    high = sum(1 for s in segments if s["processing_path"] == HIGH_OVERLAP)
    parts = list(spoken)
    if high:
        parts.append(f"{high} high-overlap segment(s) remain uncertain.")
    return " ".join(parts).strip() or "No reliable transcript is available."


def _gold_to_validatable(meeting_id: str, gold_events: list[dict[str, Any]],
                         segment_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand simple gold events into the full event shape the validator expects."""
    events: list[dict[str, Any]] = []
    for index, gold in enumerate(gold_events, start=1):
        segment = segment_by_id[gold["evidence_ids"][0]]
        speakers = (
            [] if segment["processing_path"] == HIGH_OVERLAP else [segment["speaker"]]
        )
        event: dict[str, Any] = {
            "event_id": f"{meeting_id}_e{index}",
            "event_type": gold["event_type"],
            "evidence_ids": list(gold["evidence_ids"]),
            "content": gold["content"],
            "confidence": "low" if gold["event_type"] == UNCERTAINTY else "high",
            "speakers": speakers,
        }
        if gold["event_type"] == ACTION_ITEM:
            event["task"] = gold["content"]
            event["owner"] = gold.get("owner", "uncertain")
            if "deadline" in gold:
                event["deadline"] = gold["deadline"]
        events.append(event)
    return events


def build_meeting(meeting_id: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Build and validate one meeting's annotation document.

    Raises ``ValueError`` (with meeting context) if the result is not schema-legal.
    """
    groups = _group_by_segment(rows)
    evidence_segments: list[dict[str, Any]] = []
    gold_events: list[dict[str, Any]] = []
    for segment_id, group in groups.items():
        high = _is_high_overlap(group)
        evidence_segments.append(_build_segment(segment_id, group, high))
        event = _build_gold_event(segment_id, group, high)
        if event is not None:
            gold_events.append(event)

    segment_by_id = {s["evidence_id"]: s for s in evidence_segments}
    summary = _meeting_summary(evidence_segments)
    document = {
        "meeting_id": meeting_id,
        "meeting_summary": summary,
        "events": _gold_to_validatable(meeting_id, gold_events, segment_by_id),
    }
    try:
        validate_meeting_events_document(document, evidence_segments)
    except ValueError as exc:
        raise ValueError(f"meeting '{meeting_id}': {exc}") from exc

    return {
        "description": (
            f"Annotation set for meeting '{meeting_id}', built from CSV via "
            "scripts/build_annotation_set.py and validated with "
            "validate_meeting_events_document."
        ),
        "meeting_id": meeting_id,
        "meeting_summary": summary,
        "evidence_segments": evidence_segments,
        "gold_events": gold_events,
    }


def build_all(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Group rows by meeting and build one validated document per meeting."""
    by_meeting: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_meeting.setdefault(row["meeting_id"], []).append(row)
    return {meeting_id: build_meeting(meeting_id, meeting_rows)
            for meeting_id, meeting_rows in by_meeting.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="annotation CSV path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "annotations" / "built",
        help="directory to write one <meeting_id>.json per meeting",
    )
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)
    if not rows:
        print(f"no annotation rows found in {args.csv}", file=sys.stderr)
        return 1

    documents = build_all(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for meeting_id, document in documents.items():
        out_path = args.out_dir / f"{meeting_id}.json"
        out_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[ok] {meeting_id}: {len(document['evidence_segments'])} segments, "
            f"{len(document['gold_events'])} events -> {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
