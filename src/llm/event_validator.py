"""Validation for extracted meeting events."""

from typing import Any


REQUIRED_EVENT_FIELDS = {
    "meeting_id": str,
    "event_id": str,
    "summary": str,
    "evidence_ids": list,
    "confidence": (int, float),
    "uncertainty_note": str,
}


def validate_meeting_event(event: Any, known_evidence_ids: set[str] | None = None) -> dict[str, Any]:
    """Validate one event and ensure evidence citations point to known IDs."""
    if not isinstance(event, dict):
        raise ValueError(f"event must be a dict, got {type(event).__name__}")

    for field, expected_type in REQUIRED_EVENT_FIELDS.items():
        if field not in event:
            raise ValueError(f"event is missing required field '{field}'")
        if not isinstance(event[field], expected_type):
            raise ValueError(f"event.{field} has invalid type")

    confidence = float(event["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("event.confidence must be between 0.0 and 1.0")

    if known_evidence_ids is not None:
        missing = [evidence_id for evidence_id in event["evidence_ids"] if evidence_id not in known_evidence_ids]
        if missing:
            raise ValueError(f"event cites unknown evidence_id values: {missing}")
    return event
