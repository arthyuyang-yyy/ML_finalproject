"""Build the canonical evidence representation from both processing paths."""

import json
from pathlib import Path
from typing import Any

from src.overlap.router import route_segment
from src.evidence.validator import validate_meeting, validate_metadata_segment
from src.utils import required_text

LOW_OVERLAP_PATH = "low_overlap_cluster"
HIGH_OVERLAP_PATH = "high_overlap_candidate"
VALID_PROCESSING_PATHS = {LOW_OVERLAP_PATH, HIGH_OVERLAP_PATH}


def build_metadata_segment(
    meeting_id: str,
    segment_id: str,
    speaker: str,
    start_time: float,
    end_time: float,
    text: str,
    processing_path: str,
    overlap_score: float,
    asr_confidence: float,
    speaker_confidence: float,
    candidates: list[dict[str, Any]] | None = None,
    uncertainty_note: str = "",
    evidence_id: str | None = None,
    audio_clip_path: str = "",
    source_audio_path: str = "",
    language: str = "und",
    route_reason: str = "",
    source: str = "",
    decision_reason: str = "",
) -> dict[str, Any]:
    """Build and validate one canonical evidence segment."""
    normalized_candidates = _normalize_candidates(candidates or [], segment_id, uncertainty_note)
    record = {
        "meeting_id": required_text(meeting_id, "meeting_id"),
        "segment_id": required_text(segment_id, "segment_id"),
        "evidence_id": required_text(evidence_id or segment_id, "evidence_id"),
        "speaker": required_text(speaker, "speaker"),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "text": str(text),
        "processing_path": str(processing_path),
        "route_reason": route_reason or f"overlap_score routed to {processing_path}",
        "overlap_score": float(overlap_score),
        "asr_confidence": float(asr_confidence),
        "speaker_confidence": float(speaker_confidence),
        "audio_clip_path": str(audio_clip_path),
        "source_audio_path": str(source_audio_path),
        "language": str(language or "und"),
        "candidates": normalized_candidates,
        "uncertainty_note": str(uncertainty_note),
    }
    if source:
        record["source"] = str(source)
    if decision_reason:
        record["decision_reason"] = str(decision_reason)
    return validate_metadata_segment(record)


def build_evidence_segments(
    low_overlap_segments: list[dict[str, Any]],
    high_overlap_segments: list[dict[str, Any]],
    *,
    meeting_id: str | None = None,
    source_audio_path: str = "",
    language: str = "und",
    overlap_threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Merge low/high-overlap results into sorted canonical evidence records."""
    _require_segment_list(low_overlap_segments, "low_overlap_segments")
    _require_segment_list(high_overlap_segments, "high_overlap_segments")

    evidence = [
        _build_from_processed_segment(
            segment,
            expected_path=LOW_OVERLAP_PATH,
            meeting_id=meeting_id,
            source_audio_path=source_audio_path,
            language=language,
            overlap_threshold=overlap_threshold,
        )
        for segment in low_overlap_segments
    ]
    evidence.extend(
        _build_from_processed_segment(
            segment,
            expected_path=HIGH_OVERLAP_PATH,
            meeting_id=meeting_id,
            source_audio_path=source_audio_path,
            language=language,
            overlap_threshold=overlap_threshold,
        )
        for segment in high_overlap_segments
    )
    evidence.sort(key=lambda segment: (segment["start_time"], segment["end_time"], segment["segment_id"]))
    _validate_unique_ids(evidence)
    return validate_meeting(evidence) if evidence else []


def build_evidence_file(
    low_overlap_path: str | Path,
    high_overlap_path: str | Path,
    output_path: str | Path,
    **builder_kwargs: Any,
) -> list[dict[str, Any]]:
    """Build ``evidence_segments.json`` from low/high-overlap JSON artifacts."""
    low_segments = _read_segment_list(low_overlap_path)
    high_segments = _read_segment_list(high_overlap_path)
    evidence = build_evidence_segments(low_segments, high_segments, **builder_kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def _build_from_processed_segment(
    segment: dict[str, Any],
    *,
    expected_path: str,
    meeting_id: str | None,
    source_audio_path: str,
    language: str,
    overlap_threshold: float,
) -> dict[str, Any]:
    if not isinstance(segment, dict):
        raise ValueError(f"{expected_path} entries must be dictionaries")

    actual_path = str(segment.get("processing_path") or expected_path)
    if actual_path != expected_path:
        raise ValueError(f"segment {segment.get('segment_id', '<unknown>')} belongs to {actual_path}, not {expected_path}")

    segment_meeting_id = meeting_id or segment.get("meeting_id")
    if not segment_meeting_id:
        raise ValueError(f"segment {segment.get('segment_id', '<unknown>')} is missing meeting_id")

    overlap_score = float(segment.get("overlap_score", 0.0))
    generated_route_reason = _route_reason(overlap_score, overlap_threshold, expected_path)
    route_reason = str(segment.get("route_reason") or generated_route_reason)
    return build_metadata_segment(
        meeting_id=str(segment_meeting_id),
        segment_id=str(segment.get("segment_id", "")),
        evidence_id=segment.get("evidence_id"),
        speaker=str(segment.get("speaker", "")),
        start_time=float(segment.get("start_time", 0.0)),
        end_time=float(segment.get("end_time", 0.0)),
        text=str(segment.get("text", "")),
        processing_path=expected_path,
        route_reason=route_reason,
        overlap_score=overlap_score,
        asr_confidence=float(segment.get("asr_confidence", 0.0)),
        speaker_confidence=float(segment.get("speaker_confidence", 0.0)),
        candidates=list(segment.get("candidates", [])),
        uncertainty_note=str(segment.get("uncertainty_note", "")),
        audio_clip_path=str(segment.get("audio_clip_path", "")),
        source_audio_path=str(segment.get("source_audio_path") or source_audio_path),
        language=str(segment.get("language") or language),
        source=str(segment.get("source", "")),
        decision_reason=str(segment.get("decision_reason", "")),
    )


def _normalize_candidates(
    candidates: list[dict[str, Any]],
    segment_id: str,
    segment_uncertainty_note: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate[{index - 1}] must be a dictionary")
        normalized.append({
            **candidate,
            "candidate_id": str(candidate.get("candidate_id") or f"{segment_id}_c{index}"),
            "speaker": required_text(candidate.get("speaker"), f"candidate[{index - 1}].speaker"),
            "text": required_text(candidate.get("text"), f"candidate[{index - 1}].text"),
            "confidence": float(candidate.get("confidence", 0.0)),
            "uncertainty_note": str(
                candidate.get("uncertainty_note")
                or segment_uncertainty_note
                or "High-overlap candidate; interpretation is uncertain."
            ),
        })
    return normalized


def _route_reason(score: float, threshold: float, processing_path: str) -> str:
    routed_path = route_segment(score, threshold=threshold)
    if routed_path != processing_path:
        raise ValueError(
            f"overlap_score={score:.3f} routes to {routed_path}, "
            f"but segment was supplied as {processing_path}"
        )
    comparator = ">=" if score >= threshold else "<"
    return f"overlap_score={score:.3f} {comparator} threshold={threshold:.3f}; routed to {processing_path}"


def _validate_unique_ids(segments: list[dict[str, Any]]) -> None:
    segment_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for segment in segments:
        segment_id = segment["segment_id"]
        evidence_id = segment["evidence_id"]
        if segment_id in segment_ids:
            raise ValueError(f"duplicate segment_id: {segment_id}")
        if evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")
        segment_ids.add(segment_id)
        evidence_ids.add(evidence_id)


def _require_segment_list(value: Any, name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")


def _read_segment_list(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


__all__ = [
    "HIGH_OVERLAP_PATH",
    "LOW_OVERLAP_PATH",
    "build_evidence_file",
    "build_evidence_segments",
    "build_metadata_segment",
]
