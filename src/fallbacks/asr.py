"""ASR backend auto-selection fallback.

Probes available production backends and falls back to mock when none are installed.
"""

import importlib.util
import logging

logger = logging.getLogger(__name__)

_AUTO_BACKENDS = (
    ("funasr", "funasr"),
    ("faster-whisper", "faster_whisper"),
    ("whisperx", "whisperx"),
    ("whisper", "whisper"),
)


def resolve_asr_backend() -> str:
    """Return the best available ASR backend name, defaulting to ``mock``."""
    for backend, module in _AUTO_BACKENDS:
        if importlib.util.find_spec(module) is not None:
            return backend
    logger.warning("No production ASR backend is installed; falling back to mock ASR")
    return "mock"


__all__ = ["resolve_asr_backend"]
