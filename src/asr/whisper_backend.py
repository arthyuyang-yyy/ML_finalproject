"""Whisper-family ASR entry points."""

from typing import Any

from .core import WhisperAdapter, WhisperXAdapter


def transcribe_clip(
    audio_clip_path: str,
    language: str | None = None,
    model: str = "whisperx",
    **adapter_kwargs: Any,
) -> dict[str, Any]:
    """Transcribe one clip with WhisperX or OpenAI Whisper."""
    adapter_class = WhisperXAdapter if model.lower() == "whisperx" else WhisperAdapter
    return adapter_class(language=language, **adapter_kwargs).transcribe_file(audio_clip_path)


__all__ = ["WhisperAdapter", "WhisperXAdapter", "transcribe_clip"]
