"""Fallback strategies for every external model dependency.

Each module provides a deterministic downgrade path so the pipeline stays
functional in lightweight / no-model mode.  When the real backend is available
the main processing modules use it directly; these fallbacks serve as safety
nets.

Modules
-------
asr          Backend auto-probing → mock when none installed.
candidates   Uncertainty-preserving candidates without faster-whisper.
diarization  Speaker assignment without pyannote.
overlap      Energy-based overlap estimation.
events       Deterministic meeting-event extraction without LLM.
qa           Evidence-cited deterministic QA answer.
embeddings   Hash-based character n-gram embeddings.
"""

from .asr import resolve_asr_backend
from .candidates import fallback_candidates
from .diarization import cluster_speakers
from .overlap import energy_overlap_proxy, estimate_with_energy_fallback
from .events import fallback_event_document
from .qa import fallback_answer
from .embeddings import HashingEmbeddingBackend

__all__ = [
    "HashingEmbeddingBackend",
    "cluster_speakers",
    "energy_overlap_proxy",
    "estimate_with_energy_fallback",
    "fallback_answer",
    "fallback_candidates",
    "fallback_event_document",
    "resolve_asr_backend",
]
