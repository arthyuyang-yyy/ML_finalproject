"""Evidence-aware meeting event extraction."""

from typing import Any

from .event_validator import validate_meeting_event
from .gemma_client import GemmaClient
from .prompts import build_event_extraction_prompt


def extract_meeting_events(
    evidence_segments: list[dict[str, Any]],
    client: GemmaClient | None = None,
) -> list[dict[str, Any]]:
    """Extract meeting events, falling back to one evidence-backed event."""
    if not evidence_segments:
        return []

    known_evidence_ids = {str(segment["evidence_id"]) for segment in evidence_segments}
    if client is not None:
        prompt = build_event_extraction_prompt(evidence_segments)
        response = client.generate_json(prompt)
        events = response.get("events", [])
        if events:
            return [validate_meeting_event(event, known_evidence_ids) for event in events]

    meeting_id = str(evidence_segments[0]["meeting_id"])
    summary = " ".join(str(segment.get("text", "")).strip() for segment in evidence_segments).strip()
    confidence_values = [float(segment.get("asr_confidence", 0.0)) for segment in evidence_segments]
    confidence = sum(confidence_values) / len(confidence_values)
    fallback = {
        "meeting_id": meeting_id,
        "event_id": f"{meeting_id}_event_0001",
        "summary": summary or "No transcript available.",
        "evidence_ids": sorted(known_evidence_ids),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "uncertainty_note": "; ".join(
            str(segment.get("uncertainty_note", ""))
            for segment in evidence_segments
            if segment.get("uncertainty_note")
        ),
    }
    return [validate_meeting_event(fallback, known_evidence_ids)]
