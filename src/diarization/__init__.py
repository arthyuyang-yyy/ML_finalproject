"""Speaker diarization and segment-attribution interfaces."""

from .core import (
    DEFAULT_SPEAKER_CONFIDENCE,
    PYANNOTE_DIARIZATION_MODEL,
    _best_speaker_for_segment,
    assign_speakers_to_segments,
    cluster_speakers,
    diarize_audio,
    diarize_with_pyannote,
)

assign_speaker_to_segments = assign_speakers_to_segments

__all__ = [
    "DEFAULT_SPEAKER_CONFIDENCE",
    "PYANNOTE_DIARIZATION_MODEL",
    "_best_speaker_for_segment",
    "assign_speaker_to_segments",
    "assign_speakers_to_segments",
    "cluster_speakers",
    "diarize_audio",
    "diarize_with_pyannote",
]
