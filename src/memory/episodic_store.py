"""Build and atomically persist traceable event-level episodic memory."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory_schema import Episode, validate_episode, validate_episode_collection
from src.utils import confidence_level, required_text

DEFAULT_MEMORY_PATH = Path("memory") / "episodic_memory.json"

DEFAULT_IMPORTANCE = {
    "decision": 0.90,
    "action_item": 0.90,
    "open_question": 0.75,
    "disagreement": 0.70,
    "uncertainty": 0.60,
    "speaker_stance": 0.65,
    "topic_transition": 0.70,
}


def build_episodes(
    meeting_events: dict[str, Any],
    evidence_segments: list[dict[str, Any]],
) -> list[Episode]:
    """Convert validated meeting events into persistent episodic memories."""
    if not isinstance(meeting_events, dict):
        raise ValueError("meeting_events must be a structured meeting event document")
    events = meeting_events.get("events")
    if not isinstance(events, list):
        raise ValueError("meeting_events.events must be a list")

    meeting_id = required_text(meeting_events.get("meeting_id"), "meeting_events.meeting_id")
    memory_timestamp = _memory_timestamp(meeting_events.get("memory_timestamp"))
    evidence_by_id = _index_evidence(evidence_segments, meeting_id)
    episodes: list[Episode] = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"meeting_events.events[{index - 1}] must be a dict")
        evidence_ids = _event_evidence_ids(event, index)
        missing = [value for value in evidence_ids if value not in evidence_by_id]
        if missing:
            raise ValueError(f"meeting event {index} cites unknown evidence IDs: {missing}")
        evidence = [evidence_by_id[value] for value in evidence_ids]
        episodes.append(
            _build_episode(
                meeting_id,
                index,
                event,
                evidence_ids,
                evidence,
                memory_timestamp,
            )
        )
    return validate_episode_collection(episodes)


def build_episodes_file(
    meeting_events_path: str | Path,
    evidence_segments_path: str | Path,
    output_path: str | Path = DEFAULT_MEMORY_PATH,
) -> list[Episode]:
    """Build episodes from JSON artifacts and upsert them into long-term memory."""
    meeting_events = json.loads(Path(meeting_events_path).read_text(encoding="utf-8"))
    evidence_segments = json.loads(Path(evidence_segments_path).read_text(encoding="utf-8"))
    episodes = build_episodes(meeting_events, evidence_segments)
    upsert_episodes(output_path, episodes, meeting_id=str(meeting_events["meeting_id"]))
    return episodes


def upsert_episodes(
    path: str | Path,
    episodes: list[dict[str, Any]],
    *,
    meeting_id: str | None = None,
) -> list[Episode]:
    """Atomically replace affected meetings while preserving other meetings."""
    validated_new = validate_episode_collection(episodes)
    target = Path(path)
    existing = read_episodes(target)
    replaced_meeting_ids = {episode["meeting_id"] for episode in validated_new}
    if meeting_id is not None:
        replaced_meeting_ids.add(required_text(meeting_id, "meeting_id"))
    if not replaced_meeting_ids:
        raise ValueError("upsert requires at least one episode or an explicit meeting_id")
    merged = [
        episode for episode in existing if episode["meeting_id"] not in replaced_meeting_ids
    ] + validated_new
    merged.sort(key=lambda item: (item["meeting_id"], item["start_time"], item["episode_id"]))
    _atomic_write_json(target, merged)
    return merged


def write_episodes(path: str | Path, episodes: list[dict[str, Any]]) -> None:
    """Atomically replace an episodic-memory JSON file."""
    _atomic_write_json(Path(path), validate_episode_collection(episodes))


def read_episodes(path: str | Path = DEFAULT_MEMORY_PATH) -> list[Episode]:
    """Read and validate persistent episodic memory."""
    target = Path(path)
    if not target.exists():
        return []
    raw = target.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return validate_episode_collection(json.loads(raw))


def _build_episode(
    meeting_id: str,
    index: int,
    event: dict[str, Any],
    evidence_ids: list[str],
    evidence: list[dict[str, Any]],
    memory_timestamp: str,
) -> Episode:
    original_event_type = required_text(event.get("event_type"), "event.event_type")
    if original_event_type not in DEFAULT_IMPORTANCE:
        raise ValueError(f"unsupported event_type for episodic memory: {original_event_type}")
    cites_high_overlap = any(
        item.get("processing_path") == "high_overlap_candidate" for item in evidence
    )
    event_type = "uncertainty" if cites_high_overlap else original_event_type
    confidence = "low" if cites_high_overlap else confidence_level(event.get("confidence"))
    speakers = _episode_speakers(event, evidence, cites_high_overlap)
    uncertainty_note = str(event.get("uncertainty_note", "")).strip()
    if cites_high_overlap:
        uncertainty_note = _join_notes(
            uncertainty_note,
            *[str(item.get("uncertainty_note", "")) for item in evidence],
            "High-overlap evidence is stored as uncertainty, not as a confirmed fact.",
        )

    content = required_text(event.get("content"), "event.content")
    if cites_high_overlap and original_event_type != "uncertainty":
        content = f"Uncertain interpretation: {content}"
    topic = str(event.get("topic", "")).strip() or _default_topic(original_event_type)
    importance = float(event.get("importance", DEFAULT_IMPORTANCE[event_type]))
    if cites_high_overlap:
        importance = min(importance, DEFAULT_IMPORTANCE["uncertainty"])
    evidence_text = _evidence_text(evidence)
    episode: Episode = {
        "episode_id": f"{meeting_id}_ep_{index:03d}",
        "meeting_id": meeting_id,
        "event_type": event_type,
        "topic": topic,
        "content": content,
        "speakers": speakers,
        "start_time": min(float(item["start_time"]) for item in evidence),
        "end_time": max(float(item["end_time"]) for item in evidence),
        "evidence_ids": evidence_ids,
        "evidence_text": evidence_text,
        "overlap_score": round(max(float(item.get("overlap_score", 0.0)) for item in evidence), 3),
        "confidence": confidence,
        "importance": importance,
        "audio_clip_paths": _unique_non_empty(
            str(item.get("audio_clip_path", "")) for item in evidence
        ),
        "uncertainty_note": uncertainty_note,
        "memory_timestamp": memory_timestamp,
    }
    return validate_episode(episode)


def _index_evidence(
    evidence_segments: list[dict[str, Any]],
    meeting_id: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(evidence_segments, list):
        raise ValueError("evidence_segments must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(evidence_segments):
        if not isinstance(segment, dict):
            raise ValueError(f"evidence_segments[{index}] must be a dict")
        if segment.get("meeting_id") != meeting_id:
            raise ValueError("meeting events and evidence segments must share one meeting_id")
        evidence_id = segment.get("evidence_id") or segment.get("segment_id")
        evidence_id = required_text(evidence_id, f"evidence_segments[{index}].evidence_id")
        if evidence_id in indexed:
            raise ValueError(f"duplicate evidence ID: {evidence_id}")
        indexed[evidence_id] = segment
    return indexed


def _event_evidence_ids(event: dict[str, Any], index: int) -> list[str]:
    raw_ids = event.get("evidence_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"meeting event {index} evidence_ids must be a non-empty list")
    ids = [required_text(value, f"meeting event {index} evidence ID") for value in raw_ids]
    return list(dict.fromkeys(ids))


def _episode_speakers(
    event: dict[str, Any],
    evidence: list[dict[str, Any]],
    cites_high_overlap: bool,
) -> list[str]:
    if cites_high_overlap:
        return ["MIXED"]
    event_speakers = event.get("speakers", [])
    if isinstance(event_speakers, list):
        speakers = _unique_non_empty(str(value) for value in event_speakers)
        if speakers:
            return speakers
    return _unique_non_empty(str(item.get("speaker", "UNKNOWN")) for item in evidence)


def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for segment in evidence:
        text = str(segment.get("text", "")).strip()
        if text:
            parts.append(text)
            continue
        candidates = [
            str(candidate.get("text", "")).strip()
            for candidate in segment.get("candidates", [])
            if str(candidate.get("text", "")).strip()
        ]
        if candidates:
            parts.append("Candidate interpretations: " + " / ".join(candidates))
    return " ".join(parts)


def _default_topic(event_type: str) -> str:
    return event_type.replace("_", " ")


def _memory_timestamp(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    timestamp = required_text(value, "meeting_events.memory_timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("meeting_events.memory_timestamp must be a valid ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _unique_non_empty(values: Any) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _join_notes(*notes: str) -> str:
    return "; ".join(dict.fromkeys(note.strip().rstrip(".;") for note in notes if note.strip())) + "."


def _atomic_write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "DEFAULT_MEMORY_PATH",
    "build_episodes",
    "build_episodes_file",
    "read_episodes",
    "upsert_episodes",
    "write_episodes",
]
