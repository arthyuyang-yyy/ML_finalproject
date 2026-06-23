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

# Rough character-to-token ratio for mixed Chinese/English meeting transcripts.
_CHARS_PER_TOKEN = 2.5
# Default batch token budget. For small local models (gemma3:4b ctx=4096) keep
# this small; the early-exit mechanism will fall back to deterministic after the
# first batch fails. For large cloud models (DeepSeek 128K, GPT-4o 128K) pass a
# higher value via ``batch_token_budget``.
_DEFAULT_BATCH_TOKEN_BUDGET = 30000


def extract_meeting_events(
    evidence_segments: list[dict[str, Any]],
    client: GemmaClient | None = None,
    event_index: int = 1,
    max_attempts: int = 2,
    batch_token_budget: int = _DEFAULT_BATCH_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Extract a validated meeting event document from evidence segments.

    When the evidence is too large for a single LLM prompt, segments are split
    into token-budgeted batches. Each batch is extracted independently, then
    the per-batch documents are merged and validated against the full evidence
    list. If any batch fails after all repair attempts, the function falls back
    to a deterministic evidence-only document.
    """
    if not evidence_segments:
        return {"meeting_id": "", "meeting_summary": "", "events": []}
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    if client is not None:
        batches = _split_evidence_into_batches(evidence_segments, batch_token_budget)
        batch_documents: list[dict[str, Any]] = []
        for batch in batches:
            document = _extract_single_batch(batch, client, max_attempts)
            if document is None:
                logger.warning(
                    "Event extraction batch failed; using deterministic fallback "
                    "for all %d evidence segments", len(evidence_segments),
                )
                break
            batch_documents.append(document)

        if len(batch_documents) == len(batches) and batch_documents:
            merged = _merge_batch_documents(batch_documents, evidence_segments)
            try:
                return validate_meeting_events_document(merged, evidence_segments)
            except ValueError:
                pass
            try:
                salvaged = validate_meeting_events_document(
                    merged, evidence_segments, drop_invalid_events=True,
                )
                if salvaged["events"]:
                    return salvaged
            except ValueError:
                pass

    return fallback_event_document(evidence_segments, event_index=event_index)


def _extract_single_batch(
    batch: list[dict[str, Any]],
    client: GemmaClient,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Extract events for one batch with up to *max_attempts* repair tries.

    Returns the last successfully-parsed document (even if it fails strict
    validation) so the merge step can re-validate against the full evidence
    list. Returns ``None`` only when the backend itself fails.
    """
    previous_output = ""
    previous_error = ""
    last_document: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        if attempt:
            prompt = build_event_repair_prompt(batch, previous_output, previous_error)
        else:
            prompt = build_event_extraction_prompt(batch)
        try:
            raw_output = client.generate_json(prompt)
        except Exception as exc:
            logger.warning("Gemma event extraction failed for batch: %s", exc)
            return None
        if isinstance(raw_output, dict) and raw_output.get("fallback"):
            logger.warning("Gemma event extraction returned fallback for batch; skipping LLM")
            return None
        previous_output = _serialize_output(raw_output)
        try:
            document = _coerce_document(raw_output)
            last_document = document
            return validate_meeting_events_document(document, batch)
        except ValueError as exc:
            previous_error = str(exc)

    return last_document


def _split_evidence_into_batches(
    evidence_segments: list[dict[str, Any]],
    token_budget: int,
) -> list[list[dict[str, Any]]]:
    """Split evidence segments into batches that fit within a token budget."""
    char_budget = int(token_budget * _CHARS_PER_TOKEN)
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for segment in evidence_segments:
        seg_chars = len(json.dumps(segment, ensure_ascii=False))
        if current and current_chars + seg_chars > char_budget:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += seg_chars
    if current:
        batches.append(current)
    return batches


def _merge_batch_documents(
    batch_documents: list[dict[str, Any]],
    evidence_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge per-batch event documents into a single document."""
    meeting_id = str(evidence_segments[0].get("meeting_id", ""))
    summaries = [str(doc.get("meeting_summary", "")).strip() for doc in batch_documents]
    events: list[dict[str, Any]] = []
    for doc in batch_documents:
        events.extend(doc.get("events", []))
    event_id_counter = 1
    for event in events:
        event["event_id"] = f"ev_{event_id_counter:03d}"
        event_id_counter += 1
    return {
        "meeting_id": meeting_id,
        "meeting_summary": " ".join(summary for summary in summaries if summary),
        "events": events,
    }


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
