"""Create, store, and search traceable meeting episodes."""

import json
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_PATH = Path("outputs") / "episodic_memory.jsonl"


def create_episode_from_segments(segments: list[dict]) -> dict[str, Any]:
    """Create one coherent episode record from related metadata segments."""
    if not segments:
        raise ValueError("segments must not be empty")

    meeting_id = str(segments[0]["meeting_id"])
    speakers = sorted({str(segment.get("speaker", "UNKNOWN")) for segment in segments})
    evidence_ids = [str(segment.get("evidence_id", segment.get("segment_id"))) for segment in segments]
    text = " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()
    confidence_values = [
        float(segment.get("asr_confidence", 0.0)) * float(segment.get("speaker_confidence", 0.0))
        for segment in segments
    ]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    return {
        "meeting_id": meeting_id,
        "episode_id": f"{meeting_id}_event_0001",
        "start_time": min(float(segment["start_time"]) for segment in segments),
        "end_time": max(float(segment["end_time"]) for segment in segments),
        "speakers": speakers,
        "topic": "meeting discussion",
        "summary": text or "No transcript available.",
        "evidence_ids": evidence_ids,
        "evidence": segments,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "uncertainty_note": "; ".join(
            str(segment.get("uncertainty_note", ""))
            for segment in segments
            if segment.get("uncertainty_note")
        ),
    }


def store_episode(episode: dict, path: str | Path = DEFAULT_MEMORY_PATH) -> None:
    """Persist an episode using the configured memory backend."""
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode, ensure_ascii=False) + "\n")


def search_episodes(query: str, top_k: int = 5, path: str | Path = DEFAULT_MEMORY_PATH) -> list[dict]:
    """Return the most relevant episodes for a query."""
    memory_path = Path(path)
    if not memory_path.exists():
        return []

    query_terms = {term.lower() for term in query.split() if term.strip()}
    episodes: list[tuple[int, dict]] = []
    for line in memory_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        searchable = " ".join([
            str(episode.get("summary", "")),
            str(episode.get("topic", "")),
            " ".join(episode.get("speakers", [])),
        ]).lower()
        score = sum(1 for term in query_terms if term in searchable)
        episodes.append((score, episode))

    episodes.sort(key=lambda item: item[0], reverse=True)
    return [episode for score, episode in episodes[:top_k] if score > 0] or [
        episode for _, episode in episodes[:top_k]
    ]
