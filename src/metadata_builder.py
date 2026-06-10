"""Build the shared metadata representation used across the pipeline."""

from typing import Any


def build_metadata_segment(
    meeting_id: str,
    segment_id: str,
    speaker: str,
    start_time: float,
    end_time: float,
    text: str,
    processing_path: str,
    overlap_score: float,
    asr_confidence: float,
    speaker_confidence: float,
    candidates: list[dict] | None = None,
    uncertainty_note: str = "",
    evidence_id: str | None = None,
    audio_clip_path: str = "",
    source_audio_path: str = "",
    language: str = "und",
    route_reason: str = "",
) -> dict[str, Any]:
    """Return a normalized metadata record for one meeting segment."""
    if end_time < start_time:
        raise ValueError("end_time must not be earlier than start_time")

    valid_paths = {"low_overlap_cluster", "high_overlap_candidate"}
    if processing_path not in valid_paths:
        raise ValueError(f"processing_path must be one of {sorted(valid_paths)}")

    scores = {
        "overlap_score": overlap_score,
        "asr_confidence": asr_confidence,
        "speaker_confidence": speaker_confidence,
    }
    for name, score in scores.items():
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")

    return {
        "meeting_id": meeting_id,
        "segment_id": segment_id,
        "evidence_id": evidence_id or segment_id,
        "speaker": speaker,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "processing_path": processing_path,
        "route_reason": route_reason or f"overlap_score routed to {processing_path}",
        "overlap_score": overlap_score,
        "asr_confidence": asr_confidence,
        "speaker_confidence": speaker_confidence,
        "audio_clip_path": audio_clip_path,
        "source_audio_path": source_audio_path,
        "language": language,
        "candidates": candidates or [],
        "uncertainty_note": uncertainty_note,
    }
