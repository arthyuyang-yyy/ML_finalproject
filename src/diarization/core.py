"""Speaker diarization and attribution adapters."""

import logging
import os
from functools import lru_cache
from typing import Any

DEFAULT_SPEAKER_CONFIDENCE = 0.78
MIN_SPEAKER_COVERAGE = 0.70
MIXED_SPEAKER_COVERAGE = 0.20
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str) -> list[dict]:
    """Return timestamped speaker labels and attribution confidence."""
    pyannote_turns = diarize_with_pyannote(audio_path)
    if pyannote_turns:
        return pyannote_turns

    from ..audio.preprocess import segment_audio
    return cluster_speakers(segment_audio(audio_path))


def cluster_speakers(segments: list[dict]) -> list[dict]:
    """Assign deterministic speaker labels when no diarization backend exists."""
    clustered: list[dict] = []
    for segment in segments:
        speaker = segment.get("speaker") or "UNKNOWN"
        confidence = float(segment.get("speaker_confidence", DEFAULT_SPEAKER_CONFIDENCE))
        clustered.append({
            **segment,
            "speaker": speaker,
            "speaker_confidence": max(0.0, min(1.0, confidence)),
        })
    return clustered


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Attach speaker labels and confidence to segment timestamps."""
    if not segments:
        return []
    if not diarization_turns:
        return cluster_speakers(segments)

    assigned: list[dict[str, Any]] = []
    for segment in segments:
        coverage_by_speaker = _speaker_coverage(segment, diarization_turns)
        ordered = sorted(coverage_by_speaker.items(), key=lambda item: item[1], reverse=True)
        best_speaker, coverage = ordered[0] if ordered else (None, 0.0)
        significant = [speaker for speaker, value in ordered if value >= MIXED_SPEAKER_COVERAGE]
        if coverage >= MIN_SPEAKER_COVERAGE and best_speaker is not None:
            speaker = best_speaker
            confidence = coverage
        elif len(significant) >= 2:
            speaker = "MIXED"
            confidence = min(1.0, sum(coverage_by_speaker.values()))
        else:
            speaker = "UNKNOWN"
            confidence = coverage
        assigned.append({
            **segment,
            "speaker": speaker,
            "speaker_confidence": round(max(0.0, min(1.0, confidence)), 3),
        })
    return assigned


def diarize_with_pyannote(
    audio_path: str,
    model_name: str = PYANNOTE_DIARIZATION_MODEL,
    auth_token: str | None = None,
) -> list[dict[str, Any]] | None:
    """Return pyannote speaker turns, or ``None`` if the backend is unavailable."""
    token = auth_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        logger.info("pyannote diarization disabled because no Hugging Face token is configured")
        return None

    try:
        pipeline = _load_pyannote_pipeline(model_name, token)
    except ImportError:
        logger.warning("pyannote diarization unavailable because pyannote.audio is not installed")
        return None

    try:
        output = pipeline(audio_path)
    except Exception as exc:
        logger.warning("pyannote diarization failed; using deterministic fallback: %s", exc)
        return None

    turns: list[dict[str, Any]] = []
    if hasattr(output, "itertracks"):
        for turn, _, speaker in output.itertracks(yield_label=True):
            turns.append({
                "speaker": str(speaker),
                "start_time": round(float(turn.start), 3),
                "end_time": round(float(turn.end), 3),
                "speaker_confidence": 1.0,
            })
    return turns


@lru_cache(maxsize=4)
def _load_pyannote_pipeline(model_name: str, token: str) -> Any:
    """Load each pyannote model once per process."""
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(model_name, use_auth_token=token)


def _best_speaker_for_segment(
    segment: dict[str, Any],
    diarization_turns: list[dict[str, Any]],
) -> tuple[str | None, float]:
    """Return the speaker covering the largest fraction of a segment."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    duration = max(0.0, end - start)
    if duration == 0.0:
        return None, 0.0

    coverage_by_speaker = _speaker_coverage(segment, diarization_turns)
    if not coverage_by_speaker:
        return None, 0.0
    speaker, coverage = max(coverage_by_speaker.items(), key=lambda item: item[1])
    return speaker, coverage


def _speaker_coverage(
    segment: dict[str, Any],
    diarization_turns: list[dict[str, Any]],
) -> dict[str, float]:
    """Return per-speaker fractions of the segment duration."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    duration = max(0.0, end - start)
    if duration == 0.0:
        return {}

    covered_seconds: dict[str, float] = {}
    for turn in diarization_turns:
        turn_start = float(turn["start_time"])
        turn_end = float(turn["end_time"])
        overlap_start = max(start, turn_start)
        overlap_end = min(end, turn_end)
        if overlap_end <= overlap_start:
            continue
        speaker = str(turn["speaker"])
        covered_seconds[speaker] = covered_seconds.get(speaker, 0.0) + (overlap_end - overlap_start)
    return {speaker: min(1.0, covered / duration) for speaker, covered in covered_seconds.items()}
