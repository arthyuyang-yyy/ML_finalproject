"""Canonical evidence-segment schema declarations and validation constants."""

from typing import Any, NotRequired, TypedDict

VALID_PROCESSING_PATHS = {"low_overlap_cluster", "high_overlap_candidate"}

REQUIRED_SEGMENT_FIELDS: dict[str, Any] = {
    "meeting_id": str,
    "segment_id": str,
    "evidence_id": str,
    "speaker": str,
    "start_time": (int, float),
    "end_time": (int, float),
    "text": str,
    "processing_path": str,
    "route_reason": str,
    "overlap_score": (int, float),
    "asr_confidence": (int, float),
    "speaker_confidence": (int, float),
    "audio_clip_path": str,
    "source_audio_path": str,
    "language": str,
    "candidates": list,
    "uncertainty_note": str,
}

REQUIRED_CANDIDATE_FIELDS: dict[str, Any] = {
    "candidate_id": str,
    "speaker": str,
    "text": str,
    "confidence": (int, float),
    "uncertainty_note": str,
}


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
    speaker_posterior: NotRequired[dict[str, float]]


EVIDENCE_SEGMENT_FIELDS = tuple(EvidenceSegment.__annotations__)


def _type_name(expected_type: Any) -> str:
    """Return a readable name for a type or tuple of types."""
    if isinstance(expected_type, tuple):
        return " or ".join(t.__name__ for t in expected_type)
    return expected_type.__name__


__all__ = [
    "Candidate",
    "EVIDENCE_SEGMENT_FIELDS",
    "EvidenceSegment",
    "REQUIRED_CANDIDATE_FIELDS",
    "REQUIRED_SEGMENT_FIELDS",
    "VALID_PROCESSING_PATHS",
]
