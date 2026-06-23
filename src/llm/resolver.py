"""Resolve high-overlap candidates into a final evidence segment."""

from typing import Any

from .gemma_client import GemmaClient
from .json_repair import parse_or_repair_json


def resolve_high_overlap_segments(
    segments: list[dict[str, Any]],
    client: GemmaClient | None = None,
    context_segments: list[dict[str, Any]] | None = None,
    context_window: int = 2,
    suspected_min_confidence_gain: float = 0.15,
    suspected_max_text_cer: float = 0.35,
) -> list[dict[str, Any]]:
    """Resolve high-overlap segments with an LLM, or use a deterministic fallback."""
    return [
        resolve_high_overlap_segment(
            segment,
            client=client,
            context_segments=_nearby_context(segment, context_segments or [], context_window),
            suspected_min_confidence_gain=suspected_min_confidence_gain,
            suspected_max_text_cer=suspected_max_text_cer,
        )
        for segment in segments
    ]


def resolve_high_overlap_segment(
    segment: dict[str, Any],
    client: GemmaClient | None = None,
    context_segments: list[dict[str, Any]] | None = None,
    suspected_min_confidence_gain: float = 0.15,
    suspected_max_text_cer: float = 0.35,
) -> dict[str, Any]:
    """Synthesize the most plausible high-overlap transcript from candidates."""
    candidates = list(segment.get("candidates", []))
    usable_candidates = [candidate for candidate in candidates if not _is_placeholder_candidate(candidate)]
    if segment.get("route_mode") == "suspected_high_overlap" and str(segment.get("baseline_text", "")).strip():
        return _resolve_suspected_high_overlap(
            segment,
            usable_candidates,
            min_confidence_gain=suspected_min_confidence_gain,
            max_text_cer=suspected_max_text_cer,
        )
    if not usable_candidates:
        return {
            **segment,
            "source": "unresolved",
            "decision_reason": "No usable transcript candidates were available for this high-overlap segment.",
        }

    if client is not None:
        try:
            prompt_segment = {
                **segment,
                "candidates": usable_candidates,
                "context_segments": context_segments or [],
            }
            resolved = _validate_resolution(_coerce_resolution(client.generate_json(_build_prompt(prompt_segment))))
            source = "unresolved" if resolved["resolution_mode"] == "unresolved" else "llm_resolved"
            return _apply_resolution(segment, resolved, source=source)
        except (TypeError, ValueError):
            pass

    best = max(usable_candidates, key=lambda item: float(item.get("confidence", 0.0)))
    speaker = str(best.get("speaker", "") or "").strip()
    if not speaker or speaker == "UNKNOWN":
        speaker = str(segment.get("speaker", "MIXED") or "MIXED")
    fallback = {
        "speaker": speaker,
        "text": str(best.get("text", "")),
        "confidence": float(best.get("confidence", segment.get("asr_confidence", 0.0))),
        "resolution_mode": "selected",
        "source_candidate_ids": [str(best.get("candidate_id", ""))],
        "decision_reason": (
            "Fallback resolver selected the highest-confidence candidate while "
            "preserving all alternatives for review."
        ),
    }
    return _apply_resolution(segment, fallback, source="fallback_resolved")


def _resolve_suspected_high_overlap(
    segment: dict[str, Any],
    usable_candidates: list[dict[str, Any]],
    *,
    min_confidence_gain: float,
    max_text_cer: float,
) -> dict[str, Any]:
    """Preserve baseline text unless a suspected-overlap candidate is low risk."""
    baseline_text = str(segment.get("baseline_text", "")).strip()
    baseline_speaker = str(segment.get("baseline_speaker", "") or segment.get("speaker", "UNKNOWN")).strip()
    baseline_confidence = float(segment.get("baseline_asr_confidence", segment.get("asr_confidence", 0.0)))
    baseline_speaker_confidence = float(segment.get("baseline_speaker_confidence", segment.get("speaker_confidence", 0.0)))
    if not usable_candidates:
        return _preserve_baseline_resolution(
            segment,
            baseline_text=baseline_text,
            baseline_speaker=baseline_speaker,
            baseline_confidence=baseline_confidence,
            baseline_speaker_confidence=baseline_speaker_confidence,
            reason="No usable high-overlap candidates were available; preserved low-overlap ASR baseline.",
        )

    best = max(usable_candidates, key=lambda item: float(item.get("confidence", 0.0)))
    best_text = str(best.get("text", "")).strip()
    best_confidence = float(best.get("confidence", 0.0))
    text_cer = _character_error_ratio(best_text, baseline_text)
    if best_text and best_confidence - baseline_confidence >= min_confidence_gain and text_cer <= max_text_cer:
        speaker = str(best.get("speaker", "") or "").strip() or baseline_speaker or "MIXED"
        resolution = {
            "speaker": speaker,
            "text": best_text,
            "confidence": best_confidence,
            "resolution_mode": "selected",
            "source_candidate_ids": [str(best.get("candidate_id", ""))],
            "decision_reason": (
                "Suspected high-overlap candidate replaced baseline because confidence gain "
                f"{best_confidence - baseline_confidence:.3f} >= {min_confidence_gain:.3f} "
                f"and candidate/baseline CER {text_cer:.3f} <= {max_text_cer:.3f}."
            ),
        }
        return _apply_resolution(segment, resolution, source="fallback_resolved")

    return _preserve_baseline_resolution(
        segment,
        baseline_text=baseline_text,
        baseline_speaker=baseline_speaker,
        baseline_confidence=baseline_confidence,
        baseline_speaker_confidence=baseline_speaker_confidence,
        reason=(
            "Preserved low-overlap ASR baseline for suspected high-overlap segment because "
            f"best candidate confidence gain was {best_confidence - baseline_confidence:.3f} "
            f"and candidate/baseline CER was {text_cer:.3f}."
        ),
    )


def _preserve_baseline_resolution(
    segment: dict[str, Any],
    *,
    baseline_text: str,
    baseline_speaker: str,
    baseline_confidence: float,
    baseline_speaker_confidence: float,
    reason: str,
) -> dict[str, Any]:
    return {
        **segment,
        "speaker": baseline_speaker or "UNKNOWN",
        "text": baseline_text,
        "asr_confidence": baseline_confidence,
        "speaker_confidence": baseline_speaker_confidence,
        "source": "baseline_preserved",
        "decision_reason": reason,
        "resolution_mode": "baseline_preserved",
        "source_candidate_ids": [],
    }


def _apply_resolution(segment: dict[str, Any], resolution: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        **segment,
        "speaker": resolution["speaker"],
        "text": resolution["text"],
        "asr_confidence": resolution["confidence"],
        "source": source,
        "decision_reason": resolution["decision_reason"],
        "resolution_mode": resolution["resolution_mode"],
        "source_candidate_ids": resolution["source_candidate_ids"],
    }


def _is_placeholder_candidate(candidate: dict[str, Any]) -> bool:
    text = str(candidate.get("text", "")).strip().lower()
    return "pending asr decode" in text


def _character_error_ratio(candidate_text: str, baseline_text: str) -> float:
    """Return edit distance normalized by baseline length."""
    candidate = "".join(candidate_text.split())
    baseline = "".join(baseline_text.split())
    if not baseline:
        return 0.0 if not candidate else 1.0
    if not candidate:
        return 1.0
    previous = list(range(len(baseline) + 1))
    for i, char in enumerate(candidate, start=1):
        current = [i]
        for j, baseline_char in enumerate(baseline, start=1):
            substitution = previous[j - 1] + (char != baseline_char)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return min(1.0, previous[-1] / len(baseline))


def _build_prompt(segment: dict[str, Any]) -> str:
    candidates = segment.get("candidates", [])
    candidate_lines: list[str] = []
    for idx, c in enumerate(candidates, 1):
        cid = c.get("candidate_id", f"c{idx}")
        spk = c.get("speaker", "UNKNOWN")
        txt = c.get("text", "")
        conf = float(c.get("confidence", 0.0))
        candidate_lines.append(f"[{idx}] id={cid} speaker={spk} text={txt!r} confidence={conf:.3f}")
    candidate_block = "\n".join(candidate_lines)
    context_lines: list[str] = []
    for item in segment.get("context_segments", []):
        context_lines.append(
            "[{start:.3f}-{end:.3f}] speaker={speaker} text={text!r}".format(
                start=float(item.get("start_time", 0.0)),
                end=float(item.get("end_time", 0.0)),
                speaker=item.get("speaker", "UNKNOWN"),
                text=str(item.get("text", "")),
            )
        )
    context_block = "\n".join(context_lines) if context_lines else "(no reliable adjacent context)"

    instruction = (
        "你需要根据重叠语音片段的多个候选转写、说话人假设、置信度和相邻上下文，"
        "综合生成一版最合理的最终转写。可以选择某一个候选，也可以合并或轻微修正多个候选；"
        "不要编造候选和上下文都不支持的内容。如果候选都不可信，返回 unresolved 和空 text。"
        "只返回一个 JSON 对象，不要输出任何解释文字。\n"
        "\n"
        "Synthesize the most plausible transcript from the candidates, speaker hypotheses, "
        "confidence scores, and adjacent context. You may select, merge, or lightly correct "
        "candidate text, but do not invent unsupported content. If no candidate is usable, "
        "return resolution_mode='unresolved' and text=''. "
        "Respond with ONLY a JSON object, no explanation.\n"
        "\n"
        "Output format:\n"
        '{"speaker": "SPEAKER_XX", "text": "...", "confidence": 0.XX, '
        '"resolution_mode": "selected|merged|corrected|unresolved", '
        '"source_candidate_ids": ["..."], "decision_reason": "..."}\n'
    )

    return (
        f"{instruction}\n"
        f"segment_id: {segment.get('segment_id')}\n"
        f"time_range: [{float(segment.get('start_time', 0.0)):.3f}, {float(segment.get('end_time', 0.0)):.3f}]\n"
        f"overlap_score: {segment.get('overlap_score', 0.0):.3f}\n"
        "\n"
        "Adjacent reliable context:\n"
        f"{context_block}\n"
        "\n"
        "Candidate transcriptions:\n"
        f"{candidate_block}\n"
        "\n"
        "JSON:"
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
    resolution_mode = str(payload.get("resolution_mode", "selected")).strip() or "selected"
    if resolution_mode not in {"selected", "merged", "corrected", "unresolved"}:
        raise ValueError("resolution_mode must be selected, merged, corrected, or unresolved")
    if not speaker:
        raise ValueError("resolved speaker must be non-empty")
    if not text and resolution_mode != "unresolved":
        raise ValueError("resolved text must be non-empty")
    if not reason:
        raise ValueError("decision_reason must be non-empty")
    source_candidate_ids = payload.get("source_candidate_ids", [])
    if source_candidate_ids is None:
        source_candidate_ids = []
    if not isinstance(source_candidate_ids, list) or any(not isinstance(value, str) for value in source_candidate_ids):
        raise ValueError("source_candidate_ids must be a list of strings")
    confidence = float(payload.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return {
        "speaker": speaker,
        "text": text,
        "confidence": round(confidence, 3),
        "resolution_mode": resolution_mode,
        "source_candidate_ids": [value.strip() for value in source_candidate_ids if value.strip()],
        "decision_reason": reason,
    }


def _nearby_context(
    segment: dict[str, Any],
    context_segments: list[dict[str, Any]],
    context_window: int,
) -> list[dict[str, Any]]:
    if context_window <= 0:
        return []
    start = float(segment.get("start_time", 0.0))
    before = [
        item for item in context_segments
        if str(item.get("text", "")).strip() and float(item.get("end_time", 0.0)) <= start
    ]
    after = [
        item for item in context_segments
        if str(item.get("text", "")).strip() and float(item.get("start_time", 0.0)) >= start
    ]
    selected = before[-context_window:] + after[:context_window]
    return sorted(selected, key=lambda item: (float(item.get("start_time", 0.0)), float(item.get("end_time", 0.0))))


__all__ = ["resolve_high_overlap_segment", "resolve_high_overlap_segments"]
