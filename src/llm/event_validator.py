"""Validation and conservative normalization for LLM meeting events."""

from typing import Any

from src.utils import confidence_level, required_text

ALLOWED_EVENT_TYPES = {
    "decision",
    "action_item",
    "open_question",
    "disagreement",
    "uncertainty",
    "speaker_stance",
    "topic_transition",
}


def validate_meeting_event(
    event: Any,
    known_evidence_ids: set[str] | None = None,
    evidence_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one event and conservatively normalize its confidence."""
    if not isinstance(event, dict):
        raise ValueError(f"event must be a dict, got {type(event).__name__}")

    required = {
        "event_id": str,
        "event_type": str,
        "evidence_ids": list,
        "confidence": (str, int, float),
    }
    for field, expected_type in required.items():
        if field not in event:
            raise ValueError(f"event is missing required field '{field}'")
        if not isinstance(event[field], expected_type) or isinstance(event[field], bool):
            raise ValueError(f"event.{field} has invalid type")

    normalized = dict(event)
    normalized["event_id"] = required_text(event["event_id"], "event.event_id")
    event_type = required_text(event["event_type"], "event.event_type")
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"event.event_type must be one of {sorted(ALLOWED_EVENT_TYPES)}")
    normalized["event_type"] = event_type

    content = event.get("content", event.get("summary", ""))
    normalized["content"] = required_text(content, "event.content")
    normalized.pop("summary", None)

    raw_ids = event["evidence_ids"]
    if not raw_ids:
        raise ValueError("event.evidence_ids must contain at least one evidence ID")
    evidence_ids: list[str] = []
    for value in raw_ids:
        evidence_id = required_text(value, "event.evidence_ids item")
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
    if known_evidence_ids is not None:
        missing = [value for value in evidence_ids if value not in known_evidence_ids]
        if missing:
            raise ValueError(f"event cites unknown evidence_id values: {missing}")
    normalized["evidence_ids"] = evidence_ids

    speakers = event.get("speakers", [])
    if not isinstance(speakers, list) or any(not isinstance(value, str) for value in speakers):
        raise ValueError("event.speakers must be a list of strings")
    normalized["speakers"] = [value.strip() for value in speakers if value.strip()]
    normalized["confidence"] = confidence_level(event["confidence"])

    if event_type == "action_item":
        normalized["task"] = required_text(event.get("task"), "action_item.task")
        normalized["owner"] = required_text(event.get("owner"), "action_item.owner")
        if "deadline" in event and event["deadline"] is not None:
            normalized["deadline"] = required_text(event["deadline"], "action_item.deadline")

    cited_evidence = [
        evidence_by_id[value]
        for value in evidence_ids
        if evidence_by_id is not None and value in evidence_by_id
    ]
    supported_speakers = _supported_speakers(cited_evidence)
    unsupported_speakers = [
        speaker for speaker in normalized["speakers"] if speaker not in supported_speakers
    ]
    if cited_evidence and unsupported_speakers:
        raise ValueError(f"event cites unsupported speakers: {unsupported_speakers}")
    if event_type == "action_item":
        owner = normalized["owner"]
        if cited_evidence and owner != "uncertain" and owner not in supported_speakers:
            raise ValueError(f"action_item.owner is not supported by cited evidence: {owner}")

    cites_high_overlap = any(
        item.get("processing_path") == "high_overlap_candidate" for item in cited_evidence
    )
    if cites_high_overlap and normalized["confidence"] == "high":
        normalized["confidence"] = "low" if event_type == "uncertainty" else "medium"
        normalized["uncertainty_note"] = _append_note(
            str(event.get("uncertainty_note", "")),
            "Confidence reduced because the event cites high-overlap evidence.",
        )
    if event_type == "uncertainty":
        normalized["confidence"] = "low"

    return normalized


def validate_meeting_events_document(
    document: Any,
    evidence_segments: list[dict[str, Any]],
    *,
    drop_invalid_events: bool = False,
) -> dict[str, Any]:
    """Validate a meeting event document and optionally discard invalid events."""
    if not isinstance(document, dict):
        raise ValueError("meeting event output must be a JSON object")
    if not evidence_segments:
        raise ValueError("evidence_segments must not be empty")

    meeting_ids = {str(segment.get("meeting_id", "")) for segment in evidence_segments}
    if len(meeting_ids) != 1 or "" in meeting_ids:
        raise ValueError("evidence segments must share one non-empty meeting_id")
    expected_meeting_id = next(iter(meeting_ids))
    meeting_id = required_text(document.get("meeting_id"), "meeting_id")
    if meeting_id != expected_meeting_id:
        raise ValueError(f"meeting_id must be '{expected_meeting_id}'")

    meeting_summary = required_text(document.get("meeting_summary"), "meeting_summary")
    raw_events = document.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("events must be a list")

    evidence_by_id = {str(segment["evidence_id"]): segment for segment in evidence_segments}
    valid_events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    errors: list[str] = []
    for index, event in enumerate(raw_events):
        try:
            validated = validate_meeting_event(
                event,
                known_evidence_ids=set(evidence_by_id),
                evidence_by_id=evidence_by_id,
            )
            if validated["event_id"] in event_ids:
                raise ValueError(f"duplicate event_id: {validated['event_id']}")
            event_ids.add(validated["event_id"])
            valid_events.append(validated)
        except ValueError as exc:
            if not drop_invalid_events:
                raise ValueError(f"events[{index}]: {exc}") from exc
            errors.append(f"events[{index}]: {exc}")

    result = {
        "meeting_id": meeting_id,
        "meeting_summary": meeting_summary,
        "events": valid_events,
    }
    if errors:
        result["validation_warnings"] = errors
    return result


def _append_note(current: str, note: str) -> str:
    return f"{current.rstrip('; ')}; {note}" if current.strip() else note


def _supported_speakers(evidence: list[dict[str, Any]]) -> set[str]:
    supported: set[str] = set()
    for segment in evidence:
        speaker = str(segment.get("speaker", "")).strip()
        if speaker and speaker not in {"MIXED", "UNKNOWN"}:
            supported.add(speaker)
        for candidate in segment.get("candidates", []):
            candidate_speaker = str(candidate.get("speaker", "")).strip()
            if candidate_speaker and candidate_speaker not in {"MIXED", "UNKNOWN"}:
                supported.add(candidate_speaker)
    return supported


__all__ = [
    "ALLOWED_EVENT_TYPES",
    "validate_meeting_event",
    "validate_meeting_events_document",
]
