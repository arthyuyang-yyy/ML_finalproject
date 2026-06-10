"""Episodic-memory construction, storage, and retrieval facade."""

from .episodic_store import (
    DEFAULT_MEMORY_PATH,
    build_episodes,
    build_episodes_file,
    read_episodes,
    upsert_episodes,
    write_episodes,
)
from .memory_schema import Episode, validate_episode, validate_episode_collection
from .retriever import (
    EmbeddingBackend,
    HashingEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    retrieve_episodes,
)

__all__ = [
    "DEFAULT_MEMORY_PATH",
    "Episode",
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "SentenceTransformerEmbeddingBackend",
    "build_episodes",
    "build_episodes_file",
    "read_episodes",
    "retrieve_episodes",
    "upsert_episodes",
    "validate_episode",
    "validate_episode_collection",
    "write_episodes",
]
