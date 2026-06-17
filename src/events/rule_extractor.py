"""Rule-based structured event extraction baseline (Step 16a).

A dependency-free, deterministic baseline that classifies each low-overlap
evidence segment into one structured event type using bilingual (zh/en) keyword
and pattern matching, extracts action-item owners and deadlines, and represents
high-overlap segments as ``uncertainty`` events. It emits the same schema as the
LLM extractor and is checked by the shared validator, giving the project a
non-LLM baseline to compare against plain and evidence-constrained LLM
extraction (Step 16b).
"""

import re
from typing import Any

from src.llm.event_validator import validate_meeting_events_document
from src.utils import confidence_level

LOW_OVERLAP_PATH = "low_overlap_cluster"
HIGH_OVERLAP_PATH = "high_overlap_candidate"
UNCERTAIN_SPEAKERS = {"", "MIXED", "UNKNOWN"}

# Bilingual surface cues. Order of the checks below encodes priority:
# decision > action_item > open_question > speaker_stance (default).
DECISION_CUES = (
    "决定", "决议", "敲定", "拍板", "通过", "批准", "同意采用", "达成一致",
    "decided", "decision", "we will go with", "let's go with", "lets go with",
    "agreed to", "we agree", "approve", "approved", "go ahead with", "final call",
)
ACTION_CUES = (
    "负责", "跟进", "待办", "需要完成", "请务必", "务必", "提交", "安排", "落实",
    "action item", "action:", "todo", "to-do", "to do", "follow up", "follow-up",
    "will handle", "take care of", "assigned to", "assign", "deliver", "prepare",
    "please", "need to", "responsible for",
)
OPEN_QUESTION_CUES = (
    "待确认", "待定", "需要确认", "还不清楚", "尚未确定", "是否", "有待讨论",
    "open question", "tbd", "to be decided", "to be determined", "not sure whether",
    "unclear whether", "question remains", "still open",
)

_DEADLINE_PATTERNS = (
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),
    re.compile(r"\d{1,2}\s*[月/]\s*\d{1,2}\s*[日号]?"),
    re.compile(r"(今天|明天|后天|大后天|本周[一二三四五六日天]?|下周[一二三四五六日天]?|"
               r"周[一二三四五六日天]|月底|本月底|下个?月底?)"),
    re.compile(r"(?i)\b(today|tomorrow|tonight|this week|next week|next month|"
               r"end of (?:day|week|month)|eod|eow)\b"),
    re.compile(r"(?i)\bby\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|"
               r"saturday|sunday|\w+day|the\s+\d{1,2}(?:st|nd|rd|th)?)\b"),
)


def extract_rule_based_events(
    evidence_segments: list[dict[str, Any]],
    event_index: int = 1,
) -> dict[str, Any]:
    """Build a validated meeting-event document from evidence with rules only."""
    if not evidence_segments:
        return {"meeting_id": "", "meeting_summary": "", "events": []}

    meeting_id = str(evidence_segments[0]["meeting_id"])
    events: list[dict[str, Any]] = []
    summary_parts: list[str] = []
    high_overlap_count = 0
    next_index = event_index

    for segment in evidence_segments:
        path = segment.get("processing_path")
        if path == HIGH_OVERLAP_PATH:
            events.append(_uncertainty_event(segment, next_index))
            high_overlap_count += 1
            next_index += 1
            continue
        if path != LOW_OVERLAP_PATH:
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        events.append(_low_overlap_event(segment, text, next_index))
        summary_parts.append(text)
        next_index += 1

    if high_overlap_count:
        summary_parts.append(f"{high_overlap_count} high-overlap segment(s) remain uncertain.")
    meeting_summary = " ".join(summary_parts).strip() or "No reliable transcript is available."

    return validate_meeting_events_document(
        {"meeting_id": meeting_id, "meeting_summary": meeting_summary, "events": events},
        evidence_segments,
    )


def _low_overlap_event(segment: dict[str, Any], text: str, index: int) -> dict[str, Any]:
    """Classify one low-overlap segment into a structured event."""
    speaker = str(segment.get("speaker", "")).strip()
    confidence = confidence_level(
        float(segment.get("asr_confidence", 0.0)) * float(segment.get("speaker_confidence", 0.0))
    )
    event: dict[str, Any] = {
        "event_id": f"ev_{index:03d}",
        "event_type": classify_event_type(text),
        "topic": _topic(text),
        "content": text,
        "speakers": [speaker] if speaker and speaker not in UNCERTAIN_SPEAKERS else [],
        "evidence_ids": [str(segment["evidence_id"])],
        "confidence": confidence,
    }
    if event["event_type"] == "action_item":
        event["task"] = text
        event["owner"] = speaker if speaker and speaker not in UNCERTAIN_SPEAKERS else "uncertain"
        deadline = extract_deadline(text)
        if deadline:
            event["deadline"] = deadline
    return event


def _uncertainty_event(segment: dict[str, Any], index: int) -> dict[str, Any]:
    """Represent a high-overlap segment as a low-confidence uncertainty event."""
    candidate_text = " / ".join(
        str(candidate.get("text", "")).strip()
        for candidate in segment.get("candidates", [])
        if str(candidate.get("text", "")).strip()
    )
    content = str(segment.get("uncertainty_note", "")).strip()
    if candidate_text:
        content = f"{content} Candidate interpretations: {candidate_text}".strip()
    return {
        "event_id": f"ev_{index:03d}",
        "event_type": "uncertainty",
        "topic": "high-overlap",
        "content": content or "High-overlap speech could not be attributed reliably.",
        "speakers": [],
        "evidence_ids": [str(segment["evidence_id"])],
        "confidence": "low",
    }


def classify_event_type(text: str) -> str:
    """Return the structured event type for a transcript using cue priority."""
    lowered = text.lower()
    if _contains_cue(text, lowered, DECISION_CUES):
        return "decision"
    if _contains_cue(text, lowered, ACTION_CUES):
        return "action_item"
    if text.rstrip().endswith(("?", "？")) or _contains_cue(text, lowered, OPEN_QUESTION_CUES):
        return "open_question"
    return "speaker_stance"


def extract_deadline(text: str) -> str | None:
    """Return the first deadline-like expression found, or ``None``."""
    for pattern in _DEADLINE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


def _contains_cue(text: str, lowered: str, cues: tuple[str, ...]) -> bool:
    """Match ASCII cues case-insensitively and CJK cues against raw text."""
    for cue in cues:
        if cue.isascii():
            if cue in lowered:
                return True
        elif cue in text:
            return True
    return False


def _topic(text: str) -> str:
    """A short topic label derived from the leading words of the transcript."""
    compact = " ".join(text.split())
    return compact[:24]


__all__ = ["classify_event_type", "extract_deadline", "extract_rule_based_events"]
