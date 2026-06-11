"""FunASR / Paraformer ASR entry points.

A SenseVoice-specific adapter is planned; the current implementation uses the
existing FunASR/Paraformer adapter. When a native SenseVoice adapter is added
it can replace the ``FunASRAdapter`` reference here without changing callers.
"""

from .core import FunASRAdapter

__all__ = ["FunASRAdapter"]
