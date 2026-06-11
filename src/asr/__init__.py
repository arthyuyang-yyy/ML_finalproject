"""Pluggable ASR backends used by the meeting pipeline."""

from .core import (
    ASRAdapter,
    FunASRAdapter,
    MockASRAdapter,
    WhisperAdapter,
    WhisperXAdapter,
    _aggregate_confidence,
    _from_funasr_result,
    _from_whisper_result,
    _from_whisperx_result,
    get_adapter,
    logprob_to_confidence,
    transcribe_audio,
    transcribe_segments,
)

__all__ = [
    "ASRAdapter",
    "FunASRAdapter",
    "MockASRAdapter",
    "WhisperAdapter",
    "WhisperXAdapter",
    "_aggregate_confidence",
    "_from_funasr_result",
    "_from_whisper_result",
    "_from_whisperx_result",
    "get_adapter",
    "logprob_to_confidence",
    "transcribe_audio",
    "transcribe_segments",
]
