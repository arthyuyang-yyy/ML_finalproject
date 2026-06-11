"""Assign diarization turns to VAD segments."""

from typing import Any

from .core import assign_speakers_to_segments


def assign_speaker_to_segments(
    segments: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach speaker labels and confidence based on timestamp overlap."""
    return assign_speakers_to_segments(segments, diarization)


__all__ = ["assign_speaker_to_segments", "assign_speakers_to_segments"]
