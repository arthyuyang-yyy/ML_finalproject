"""pyannote speaker diarization backend."""

from .core import PYANNOTE_DIARIZATION_MODEL, diarize_audio, diarize_with_pyannote

__all__ = ["PYANNOTE_DIARIZATION_MODEL", "diarize_audio", "diarize_with_pyannote"]
