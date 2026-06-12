"""Evidence-grounded structured meeting event extraction."""

import json
import logging
from pathlib import Path
from typing import Any

from .event_validator import validate_meeting_events_document
from .gemma_client import GemmaClient
from .json_repair import parse_or_repair_json
from .prompts import build_event_extraction_prompt, build_event_repair_prompt
from src.fallbacks.events import fallback_event_document

logger = logging.getLogger(__name__)


def extract_meeting_events(
    evidence_segments: list[dict[str, Any]],
    client: GemmaClient | None = None,
    event_index: int = 1,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Extract a validated meeting event document from evidence segments.

    LLM output is parsed and validated strictly. A configured client receives
    one repair attempt by default. If validation still fails, valid events are
    salvaged when possible; otherwise a deterministic evidence-only fallback is
    returned.
    """
    if not evidence_segments:
        return {"meeting_id": "", "meeting_summary": "", "events": []}
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    if client is not None:
        prompt = build_event_extraction_prompt(evidence_segments)
        previous_output = ""
        previous_error = ""
        last_document: dict[str, Any] | None = None
        for attempt in range(max_attempts):
            if attempt:
                prompt = build_event_repair_prompt(
                    evidence_segments,
                    previous_output,
                    previous_error,
                )
            try:
                raw_output = client.generate_json(prompt)
            except Exception as exc:
                logger.warning("Gemma event extraction failed; using deterministic fallback: %s", exc)
                break
            previous_output = _serialize_output(raw_output)
            try:
                document = _coerce_document(raw_output)
                last_document = document
                return validate_meeting_events_document(document, evidence_segments)
            except ValueError as exc:
                previous_error = str(exc)

        if last_document is not None:
            try:
                salvaged = validate_meeting_events_document(
                    last_document,
                    evidence_segments,
                    drop_invalid_events=True,
                )
                if salvaged["events"]:
                    return salvaged
            except ValueError:
                pass

    return fallback_event_document(evidence_segments, event_index=event_index)


def extract_meeting_events_file(
    evidence_path: str | Path,
    output_path: str | Path,
    client: GemmaClient | None = None,
    **extractor_kwargs: Any,
) -> dict[str, Any]:
    """Extract and write ``meeting_events.json`` from ``evidence_segments.json``."""
    payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evidence_segments.json must contain a JSON list")
    document = extract_meeting_events(payload, client=client, **extractor_kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document


def _coerce_document(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        return parse_or_repair_json(raw_output)
    raise ValueError(f"Gemma output must be a dict or JSON string, got {type(raw_output).__name__}")


def _serialize_output(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output
    try:
        return json.dumps(raw_output, ensure_ascii=False)
    except TypeError:
        return repr(raw_output)


__all__ = ["extract_meeting_events", "extract_meeting_events_file", "_serialize_output"]
