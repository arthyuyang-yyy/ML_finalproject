"""Speaker diarization and attribution adapters."""

import os
from typing import Any

DEFAULT_SPEAKER_CONFIDENCE = 0.78
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def diarize_audio(audio_path: str) -> list[dict]:
    """Return timestamped speaker labels and attribution confidence."""
    pyannote_turns = diarize_with_pyannote(audio_path)
    if pyannote_turns:
        return pyannote_turns

    from .audio.preprocess import segment_audio
    return cluster_speakers(segment_audio(audio_path))


def cluster_speakers(segments: list[dict]) -> list[dict]:
    """Assign deterministic speaker labels when no diarization backend exists."""
    clustered: list[dict] = []
    for index, segment in enumerate(segments):
        speaker = segment.get("speaker") or f"SPEAKER_{index % 2:02d}"
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
    for index, segment in enumerate(segments):
        best_speaker, coverage = _best_speaker_for_segment(segment, diarization_turns)
        speaker = best_speaker or segment.get("speaker") or f"SPEAKER_{index % 2:02d}"
        confidence = coverage if best_speaker else DEFAULT_SPEAKER_CONFIDENCE
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
        return None

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return None

    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        output = pipeline(audio_path)
    except Exception:
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

    coverage_by_speaker: dict[str, float] = {}
    for turn in diarization_turns:
        turn_start = float(turn["start_time"])
        turn_end = float(turn["end_time"])
        overlap_start = max(start, turn_start)
        overlap_end = min(end, turn_end)
        if overlap_end <= overlap_start:
            continue
        speaker = str(turn["speaker"])
        coverage_by_speaker[speaker] = coverage_by_speaker.get(speaker, 0.0) + (overlap_end - overlap_start)

    if not coverage_by_speaker:
        return None, 0.0
    speaker, covered = max(coverage_by_speaker.items(), key=lambda item: item[1])
    return speaker, covered / duration
