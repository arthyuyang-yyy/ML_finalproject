"""Chinese-oriented ASR backend facade.

The initial implementation uses the existing FunASR/Paraformer adapter. A
SenseVoice-specific adapter can replace it without changing pipeline callers.
"""

from typing import Any

from .core import FunASRAdapter


def transcribe_clip(audio_clip_path: str, **adapter_kwargs: Any) -> dict[str, Any]:
    """Transcribe one clip with the configured FunASR-compatible backend."""
    return FunASRAdapter(**adapter_kwargs).transcribe_file(audio_clip_path)


__all__ = ["FunASRAdapter", "transcribe_clip"]
