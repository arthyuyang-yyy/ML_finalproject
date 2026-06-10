"""Canonical evidence-segment schema declarations."""

from typing import Any, NotRequired, TypedDict


class Candidate(TypedDict):
    candidate_id: str
    speaker: str
    text: str
    confidence: float
    uncertainty_note: str
    decode_config: NotRequired[dict[str, Any]]


class EvidenceSegment(TypedDict):
    meeting_id: str
    segment_id: str
    evidence_id: str
    speaker: str
    start_time: float
    end_time: float
    text: str
    processing_path: str
    route_reason: str
    overlap_score: float
    asr_confidence: float
    speaker_confidence: float
    audio_clip_path: str
    source_audio_path: str
    language: str
    candidates: list[Candidate]
    uncertainty_note: str


EVIDENCE_SEGMENT_FIELDS = tuple(EvidenceSegment.__annotations__)

__all__ = ["Candidate", "EVIDENCE_SEGMENT_FIELDS", "EvidenceSegment"]
