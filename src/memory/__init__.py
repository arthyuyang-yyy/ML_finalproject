"""Memory package facade."""

from src.episodic_memory import (
    create_episode_from_segments,
    create_episodes_from_segments,
    search_episodes,
    store_episode,
    store_episodes,
)

__all__ = [
    "create_episode_from_segments",
    "create_episodes_from_segments",
    "search_episodes",
    "store_episode",
    "store_episodes",
]
