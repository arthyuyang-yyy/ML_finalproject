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
from .retriever import retrieve_episodes
from src.fallbacks.embeddings import HashingEmbeddingBackend

__all__ = [
    "DEFAULT_MEMORY_PATH",
    "Episode",
    "HashingEmbeddingBackend",
    "build_episodes",
    "build_episodes_file",
    "read_episodes",
    "retrieve_episodes",
    "upsert_episodes",
    "validate_episode",
    "validate_episode_collection",
    "write_episodes",
]
