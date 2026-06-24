"""Speaker diarization and attribution adapters."""

import logging
import os
from functools import lru_cache
from typing import Any

import numpy as np

from src.errors import BackendExecutionError, BackendUnavailableError
from src.fallbacks.diarization import cluster_speakers

DEFAULT_SPEAKER_CONFIDENCE = 0.78
MIN_SPEAKER_COVERAGE = 0.70
MIXED_SPEAKER_COVERAGE = 0.20
SHORT_SEGMENT_MAX_SECONDS = 2.0
SHORT_SEGMENT_MIN_SPEAKER_COVERAGE = 0.35
SHORT_SEGMENT_MIN_SPEAKER_SECONDS = 0.25
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
        covered_seconds_by_speaker = _speaker_covered_seconds(segment, diarization_turns)
        ordered = sorted(coverage_by_speaker.items(), key=lambda item: item[1], reverse=True)
        best_speaker, coverage = ordered[0] if ordered else (None, 0.0)
        best_seconds = covered_seconds_by_speaker.get(best_speaker, 0.0) if best_speaker else 0.0
        significant = [speaker for speaker, value in ordered if value >= MIXED_SPEAKER_COVERAGE]
        if coverage >= MIN_SPEAKER_COVERAGE and best_speaker is not None:
            speaker = best_speaker
            confidence = coverage
        elif len(significant) >= 2:
            speaker = "MIXED"
            confidence = min(1.0, sum(coverage_by_speaker.values()))
        elif (
            best_speaker is not None
            and _segment_duration(segment) <= SHORT_SEGMENT_MAX_SECONDS
            and (
                coverage >= SHORT_SEGMENT_MIN_SPEAKER_COVERAGE
                or best_seconds >= SHORT_SEGMENT_MIN_SPEAKER_SECONDS
            )
        ):
            speaker = best_speaker
            confidence = coverage
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

    The audio is loaded with the project's own :func:`load_audio` (soundfile
    based, 16 kHz mono float32) and fed to the pipeline as an in-memory
    waveform. This avoids pyannote.audio 4.x's hard dependency on
    ``torchcodec``/FFmpeg for file decoding, which frequently breaks on
    macOS where the system FFmpeg major version does not match the one
    torchcodec was built against.
    """
    token = auth_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        logger.info("pyannote diarization disabled because no Hugging Face token is configured")
        return None

    try:
        pipeline = load_pyannote_pipeline(model_name, token)
    except ImportError as exc:
        raise BackendUnavailableError(
            "pyannote diarization is configured but pyannote.audio is unavailable"
        ) from exc
    except (BackendUnavailableError, BackendExecutionError):
        raise
    except Exception as exc:
        raise BackendExecutionError(
            f"pyannote diarization failed for model '{model_name}'"
        ) from exc

    try:
        import torch

        from ..audio.preprocess import TARGET_SAMPLE_RATE, load_audio

        samples, sample_rate = load_audio(audio_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            from ..audio.preprocess import resample

            samples = resample(samples, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        output = pipeline({"waveform": waveform, "sample_rate": int(sample_rate)})
    except ImportError as exc:
        raise BackendUnavailableError(
            "pyannote diarization is configured but pyannote.audio is unavailable"
        ) from exc
    except (BackendUnavailableError, BackendExecutionError):
        raise
    except Exception as exc:
        raise BackendExecutionError(
            f"pyannote diarization failed for model '{model_name}'"
        ) from exc

    return _pyannote_turns_to_dicts(_annotation_from_output(output))


def _annotation_from_output(output: Any) -> Any:
    """Return the pyannote ``Annotation`` from a pipeline output.

    pyannote.audio 4.x returns a ``DiarizeOutput`` namedtuple whose
    ``speaker_diarization`` field holds the classic :class:`Annotation`, while
    older versions returned the ``Annotation`` (or timeline-like object)
    directly. This normalises both so callers can rely on ``itertracks``.
    """
    for attr in ("speaker_diarization", "annotation"):
        candidate = getattr(output, attr, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate
    return output


def _pyannote_turns_to_dicts(annotation: Any) -> list[dict[str, Any]]:
    """Convert a pyannote ``Annotation`` into the project's turn dicts."""
    turns: list[dict[str, Any]] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append({
                "speaker": str(speaker),
                "start_time": round(float(turn.start), 3),
                "end_time": round(float(turn.end), 3),
                "speaker_confidence": 1.0,
            })
    return turns


@lru_cache(maxsize=4)
def load_pyannote_pipeline(model_name: str, token: str) -> Any:
    """Load each pyannote model once per process and move it to the best accelerator.

    pyannote 3.x runs on CPU by default, which is prohibitively slow for long
    meetings (tens of minutes of audio can take tens of minutes to diarize).
    The pipeline is moved to CUDA/MPS when available so inference does not
    appear to hang; failures here are non-fatal and fall back to CPU.
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_name, token=token)
    _move_pipeline_to_accelerator(pipeline)
    return pipeline


def _move_pipeline_to_accelerator(pipeline: Any) -> None:
    """Move a pyannote pipeline to CUDA/MPS when available, else keep CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
    except Exception as exc:
        logger.warning("could not move pyannote pipeline to accelerator; using CPU: %s", exc)


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
    duration = _segment_duration(segment)
    if duration == 0.0:
        return {}

    return {
        speaker: min(1.0, covered / duration)
        for speaker, covered in _speaker_covered_seconds(segment, diarization_turns).items()
    }


def _speaker_covered_seconds(
    segment: dict[str, Any],
    diarization_turns: list[dict[str, Any]],
) -> dict[str, float]:
    """Return per-speaker covered seconds within one segment."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    if end <= start:
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
    return covered_seconds


def _segment_duration(segment: dict[str, Any]) -> float:
    return max(0.0, float(segment["end_time"]) - float(segment["start_time"]))
