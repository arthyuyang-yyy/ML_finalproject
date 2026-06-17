"""Speaker diarization and segment-attribution interfaces."""

from .core import (
    DEFAULT_SPEAKER_CONFIDENCE,
    PYANNOTE_DIARIZATION_MODEL,
    _best_speaker_for_segment,
    assign_speakers_to_segments,
    diarize_audio,
    diarize_with_pyannote,
    load_pyannote_pipeline,
)
from .embedding_cluster import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_TEMPERATURE,
    AcousticEmbeddingBackend,
    ResemblyzerEmbeddingBackend,
    SpeakerEmbeddingBackend,
    agglomerative_cluster,
    cluster_segments,
    cluster_similarity_distributions,
    cosine_distance_matrix,
)

__all__ = [
    "DEFAULT_DISTANCE_THRESHOLD",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_SPEAKER_CONFIDENCE",
    "DEFAULT_TEMPERATURE",
    "PYANNOTE_DIARIZATION_MODEL",
    "AcousticEmbeddingBackend",
    "ResemblyzerEmbeddingBackend",
    "SpeakerEmbeddingBackend",
    "_best_speaker_for_segment",
    "agglomerative_cluster",
    "assign_speakers_to_segments",
    "cluster_segments",
    "cluster_similarity_distributions",
    "cosine_distance_matrix",
    "diarize_audio",
    "diarize_with_pyannote",
    "load_pyannote_pipeline",
]
