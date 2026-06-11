"""Deterministic event extraction when LLM is unavailable or output is invalid."""

from typing import Any

from src.utils import confidence_level


def fallback_event_document(
    evidence_segments: list[dict[str, Any]],
    event_index: int,
) -> dict[str, Any]:
    """Build a deterministic event document from evidence segments.

    Low-overlap segments become ``speaker_stance`` events.  High-overlap
    segments become ``uncertainty`` events.
    """
    from src.llm.event_validator import validate_meeting_events_document

    meeting_id = str(evidence_segments[0]["meeting_id"])
    low_segments = [
        segment
        for segment in evidence_segments
        if segment.get("processing_path") == "low_overlap_cluster" and str(segment.get("text", "")).strip()
    ]
    high_segments = [
        segment
        for segment in evidence_segments
        if segment.get("processing_path") == "high_overlap_candidate"
    ]

    summary_parts = [str(segment["text"]).strip() for segment in low_segments]
    if high_segments:
        summary_parts.append(f"{len(high_segments)} high-overlap segment(s) remain uncertain.")
    meeting_summary = " ".join(summary_parts).strip() or "No reliable transcript is available."

    events: list[dict[str, Any]] = []
    next_index = event_index
    for segment in low_segments:
        confidence = confidence_level(
            float(segment.get("asr_confidence", 0.0))
            * float(segment.get("speaker_confidence", 0.0))
        )
        events.append({
            "event_id": f"ev_{next_index:03d}",
            "event_type": "speaker_stance",
            "content": str(segment["text"]).strip(),
            "speakers": [str(segment.get("speaker", "UNKNOWN"))],
            "evidence_ids": [str(segment["evidence_id"])],
            "confidence": confidence,
        })
        next_index += 1

    for segment in high_segments:
        candidate_text = " / ".join(
            str(candidate.get("text", "")).strip()
            for candidate in segment.get("candidates", [])
            if str(candidate.get("text", "")).strip()
        )
        content = str(segment.get("uncertainty_note", "")).strip()
        if candidate_text:
            content = f"{content} Candidate interpretations: {candidate_text}".strip()
        events.append({
            "event_id": f"ev_{next_index:03d}",
            "event_type": "uncertainty",
            "content": content or "High-overlap speech could not be attributed reliably.",
            "speakers": [],
            "evidence_ids": [str(segment["evidence_id"])],
            "confidence": "low",
        })
        next_index += 1

    return validate_meeting_events_document(
        {
            "meeting_id": meeting_id,
            "meeting_summary": meeting_summary,
            "events": events,
        },
        evidence_segments,
    )


__all__ = ["fallback_event_document"]
