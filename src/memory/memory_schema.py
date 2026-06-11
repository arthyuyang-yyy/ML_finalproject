"""Canonical episodic-memory schema and validation."""

from datetime import datetime
from typing import Any, NotRequired, TypedDict

CONFIDENCE_LEVELS = {"high", "medium", "low"}


class Episode(TypedDict):
    episode_id: str
    meeting_id: str
    event_type: str
    topic: str
    content: str
    speakers: list[str]
    start_time: float
    end_time: float
    evidence_ids: list[str]
    evidence_text: str
    overlap_score: float
    confidence: str
    importance: float
    audio_clip_paths: list[str]
    uncertainty_note: str
    memory_timestamp: NotRequired[str]


def validate_episode(episode: Any) -> Episode:
    """Validate one persistent episodic-memory record."""
    if not isinstance(episode, dict):
        raise ValueError(f"episode must be a dict, got {type(episode).__name__}")

    required: dict[str, Any] = {
        "episode_id": str,
        "meeting_id": str,
        "event_type": str,
        "topic": str,
        "content": str,
        "speakers": list,
        "start_time": (int, float),
        "end_time": (int, float),
        "evidence_ids": list,
        "evidence_text": str,
        "overlap_score": (int, float),
        "confidence": str,
        "importance": (int, float),
        "audio_clip_paths": list,
        "uncertainty_note": str,
    }
    for field, expected_type in required.items():
        if field not in episode:
            raise ValueError(f"episode is missing required field '{field}'")
        if not isinstance(episode[field], expected_type) or (
            field in {"start_time", "end_time", "overlap_score", "importance"}
            and isinstance(episode[field], bool)
        ):
            raise ValueError(f"episode.{field} has invalid type")

    for field in ("episode_id", "meeting_id", "event_type", "topic", "content"):
        if not episode[field].strip():
            raise ValueError(f"episode.{field} must be a non-empty string")
    if float(episode["end_time"]) <= float(episode["start_time"]):
        raise ValueError("episode.end_time must be greater than episode.start_time")
    for field in ("overlap_score", "importance"):
        if not 0.0 <= float(episode[field]) <= 1.0:
            raise ValueError(f"episode.{field} must be between 0.0 and 1.0")
    if episode["confidence"] not in CONFIDENCE_LEVELS:
        raise ValueError("episode.confidence must be high, medium, or low")
    if not episode["evidence_ids"]:
        raise ValueError("episode.evidence_ids must not be empty")
    for field in ("speakers", "evidence_ids", "audio_clip_paths"):
        if any(not isinstance(value, str) or not value.strip() for value in episode[field]):
            raise ValueError(f"episode.{field} must contain non-empty strings")
    if episode["event_type"] == "uncertainty":
        if episode["confidence"] != "low":
            raise ValueError("uncertainty episodes must have low confidence")
        if not episode["uncertainty_note"].strip():
            raise ValueError("uncertainty episodes must explain their uncertainty")
    memory_timestamp = episode.get("memory_timestamp")
    if memory_timestamp is not None:
        if not isinstance(memory_timestamp, str) or not memory_timestamp.strip():
            raise ValueError("episode.memory_timestamp must be a non-empty ISO timestamp")
        try:
            datetime.fromisoformat(memory_timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("episode.memory_timestamp must be a valid ISO timestamp") from exc
    return episode


def validate_episode_collection(episodes: Any) -> list[Episode]:
    """Validate a list of episodes and enforce globally unique IDs."""
    if not isinstance(episodes, list):
        raise ValueError("episodic memory must be a list")
    validated = [validate_episode(episode) for episode in episodes]
    episode_ids = [episode["episode_id"] for episode in validated]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episodic memory contains duplicate episode_id values")
    return validated


__all__ = [
    "CONFIDENCE_LEVELS",
    "Episode",
    "validate_episode",
    "validate_episode_collection",
]
