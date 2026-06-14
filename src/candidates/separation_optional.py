"""Optional speech-separation facade for the high-overlap path.

Re-exports the dependency-free NMF separation baseline and its pluggable
backends from :mod:`src.speech_separation`. The high-overlap path consumes these
through ``process_high_overlap_segments(..., separate=True)``.
"""

from src.speech_separation import (
    AsteroidSeparationBackend,
    NmfSeparationBackend,
    SpeechSeparator,
    separate_speakers,
    separate_waveform,
)

__all__ = [
    "AsteroidSeparationBackend",
    "NmfSeparationBackend",
    "SpeechSeparator",
    "separate_speakers",
    "separate_waveform",
]
