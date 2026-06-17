"""Resolve high-overlap candidates into a final evidence segment."""

import json
from typing import Any

from .gemma_client import GemmaClient
from .json_repair import parse_or_repair_json


def resolve_high_overlap_segments(
    segments: list[dict[str, Any]],
    client: GemmaClient | None = None,
) -> list[dict[str, Any]]:
    """Resolve high-overlap segments with an LLM, or use a deterministic fallback."""
    return [resolve_high_overlap_segment(segment, client=client) for segment in segments]


def resolve_high_overlap_segment(
    segment: dict[str, Any],
    client: GemmaClient | None = None,
) -> dict[str, Any]:
    """Choose or merge candidates for one high-overlap segment."""
    candidates = list(segment.get("candidates", []))
    if not candidates:
        return {
            **segment,
            "source": "unresolved",
            "decision_reason": "No candidates were available for this high-overlap segment.",
        }

    if client is not None:
        try:
            resolved = _validate_resolution(_coerce_resolution(client.generate_json(_build_prompt(segment))))
            return _apply_resolution(segment, resolved, source="llm_resolved")
        except (TypeError, ValueError):
            pass

    best = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
    speaker = str(best.get("speaker", "") or "").strip()
    if not speaker or speaker == "UNKNOWN":
        speaker = str(segment.get("speaker", "MIXED") or "MIXED")
    fallback = {
        "speaker": speaker,
        "text": str(best.get("text", "")),
        "confidence": float(best.get("confidence", segment.get("asr_confidence", 0.0))),
        "decision_reason": (
            "Fallback resolver selected the highest-confidence candidate while "
            "preserving all alternatives for review."
        ),
    }
    return _apply_resolution(segment, fallback, source="fallback_resolved")


def _apply_resolution(segment: dict[str, Any], resolution: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        **segment,
        "speaker": resolution["speaker"],
        "text": resolution["text"],
        "asr_confidence": resolution["confidence"],
        "source": source,
        "decision_reason": resolution["decision_reason"],
    }


def _build_prompt(segment: dict[str, Any]) -> str:
    payload = {
        "segment_id": segment.get("segment_id"),
        "time_range": [segment.get("start_time"), segment.get("end_time")],
        "overlap_score": segment.get("overlap_score"),
        "candidates": segment.get("candidates", []),
    }
    return (
        "Resolve this high-overlap meeting segment using only the candidates. "
        "Return JSON with speaker, text, confidence, and decision_reason. "
        "Do not invent content outside the candidates.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _coerce_resolution(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        return parse_or_repair_json(raw_output)
    raise ValueError(f"resolver output must be a dict or JSON string, got {type(raw_output).__name__}")


def _validate_resolution(payload: dict[str, Any]) -> dict[str, Any]:
    speaker = str(payload.get("speaker", "")).strip()
    text = str(payload.get("text", "")).strip()
    reason = str(payload.get("decision_reason", "")).strip()
    if not speaker:
        raise ValueError("resolved speaker must be non-empty")
    if not text:
        raise ValueError("resolved text must be non-empty")
    if not reason:
        raise ValueError("decision_reason must be non-empty")
    confidence = float(payload.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return {
        "speaker": speaker,
        "text": text,
        "confidence": round(confidence, 3),
        "decision_reason": reason,
    }


__all__ = ["resolve_high_overlap_segment", "resolve_high_overlap_segments"]
