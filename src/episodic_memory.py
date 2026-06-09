"""Create, store, and search traceable meeting episodes."""

from typing import Any


def create_episode_from_segments(segments: list[dict]) -> dict[str, Any]:
    """Create one coherent episode record from related metadata segments."""
    # TODO: group segments by event/topic and aggregate evidence and uncertainty.
    raise NotImplementedError("Episode creation is not implemented yet.")


def store_episode(episode: dict) -> None:
    """Persist an episode using the configured memory backend."""
    # TODO: begin with local JSONL storage, then add vector retrieval.
    raise NotImplementedError("Episode storage is not implemented yet.")


def search_episodes(query: str, top_k: int = 5) -> list[dict]:
    """Return the most relevant episodes for a query."""
    # TODO: combine semantic retrieval with meeting, speaker, and time filters.
    raise NotImplementedError("Episode search is not implemented yet.")
