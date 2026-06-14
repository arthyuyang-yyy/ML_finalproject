"""Compatibility facade for optional speech separation."""

from src.speech_separation import (
    get_separation_adapter,
    separate_speakers,
    separate_waveform,
)

__all__ = ["get_separation_adapter", "separate_speakers", "separate_waveform"]
