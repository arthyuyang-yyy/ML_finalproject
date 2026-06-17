"""Whisper-family ASR entry points."""

from typing import Any

from .core import FasterWhisperAdapter, WhisperAdapter, WhisperXAdapter


def transcribe_clip(audio_clip_path: str, language: str | None = None, **adapter_kwargs: Any) -> dict[str, Any]:
    """Transcribe one clip with the project's primary faster-whisper baseline."""
    return FasterWhisperAdapter(language=language, **adapter_kwargs).transcribe_file(audio_clip_path)


__all__ = ["FasterWhisperAdapter", "WhisperAdapter", "WhisperXAdapter", "transcribe_clip"]
