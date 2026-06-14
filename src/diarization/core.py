"""Speaker diarization and attribution adapters."""

import logging
import os
from functools import lru_cache
from typing import Any

from src.errors import BackendExecutionError, BackendUnavailableError
from src.fallbacks.diarization import cluster_speakers

DEFAULT_SPEAKER_CONFIDENCE = 0.78
MIN_SPEAKER_COVERAGE = 0.70
MIXED_SPEAKER_COVERAGE = 0.20
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str) -> list[dict]:
    """Return timestamped speaker labels and attribution confidence.

    pyannote remains authoritative when configured. Otherwise the audio is
    segmented and run through unsupervised speaker-embedding clustering, so the
    public Step 5 interface recovers real speakers in the lightweight mode
    instead of always returning placeholder labels.
    """
    pyannote_turns = diarize_with_pyannote(audio_path)
    if pyannote_turns:
        return pyannote_turns

    from ..audio.preprocess import load_audio, segment_waveform
    from .embedding_cluster import cluster_segments

    samples, sample_rate = load_audio(audio_path)
    segments = segment_waveform(samples, sample_rate)
    if not segments:
        return []
    return cluster_segments(samples, segments, sample_rate)


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]] | None = None,
    *,
    samples: Any | None = None,
    sample_rate: int | None = None,
) -> list[dict[str, Any]]:
    """Attach speaker labels and confidence to segment timestamps.

    When pyannote diarization turns are available they remain authoritative.
    Otherwise, if the waveform is supplied, speakers are recovered by
    unsupervised speaker-embedding clustering (which also attaches a
    ``cluster_similarity_distribution``); failing that, the deterministic
    no-model fallback assigns placeholder labels.
    """
    if not segments:
        return []
    if not diarization_turns:
        if samples is not None and len(samples) > 0:
            from .embedding_cluster import cluster_segments

            from ..audio.preprocess import TARGET_SAMPLE_RATE

            rate = sample_rate or TARGET_SAMPLE_RATE
            return cluster_segments(samples, segments, rate)
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
    """Return pyannote speaker turns when configured.

    A missing token means the optional backend is disabled and returns
    ``None``. Once configured, dependency, model-loading, and inference
    failures are surfaced because diarization is then a required service.
    """
    token = auth_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        logger.info("pyannote diarization disabled because no Hugging Face token is configured")
        return None

    try:
        pipeline = load_pyannote_pipeline(model_name, token)
        output = pipeline(audio_path)
    except ImportError as exc:
        raise BackendUnavailableError(
            "pyannote diarization is configured but pyannote.audio is unavailable"
        ) from exc
    except Exception as exc:
        raise BackendExecutionError(
            f"pyannote diarization failed for model '{model_name}'"
        ) from exc

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
def load_pyannote_pipeline(model_name: str, token: str) -> Any:
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
