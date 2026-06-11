"""Validation and normalization for evidence-backed QA answers."""

import re
from typing import Any


def validate_qa_answer(
    answer: Any,
    question: str,
    retrieved_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject citations or entities not supported by retrieved episodes."""
    if not isinstance(answer, dict):
        raise ValueError("QA answer must be a JSON object")
    if answer.get("insufficient_evidence") is True:
        text = str(answer.get("answer", "")).strip()
        if not text:
            raise ValueError("insufficient answer must explain that it cannot determine")
        if not any(
            marker in text.lower()
            for marker in ("无法确定", "不能确定", "证据不足", "cannot determine", "insufficient evidence")
        ):
            raise ValueError("insufficient answer must explicitly say it cannot determine")
        return {
            "answer": text,
            "episode_ids": [],
            "evidence_ids": [],
            "citations": [],
            "evidence": [],
            "speakers": [],
            "speaker": "",
            "timestamp": "",
            "confidence": "low",
            "uncertainty_note": str(answer.get("uncertainty_note", "")).strip(),
            "insufficient_evidence": True,
            "question": question,
            "query": question,
        }

    episode_by_id = {str(item["episode_id"]): item for item in retrieved_episodes}
    citations = answer.get("citations")
    if not isinstance(citations, list) or not citations:
        raise ValueError("supported QA answers require at least one citation")

    normalized_citations: list[dict[str, Any]] = []
    cited_episode_ids: list[str] = []
    cited_evidence_ids: list[str] = []
    cited_episodes: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("each citation must be an object")
        episode_id = str(citation.get("episode_id", "")).strip()
        if episode_id not in episode_by_id:
            raise ValueError(f"citation references unknown episode_id: {episode_id}")
        episode = episode_by_id[episode_id]
        evidence_ids = citation.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("citation evidence_ids must be a non-empty list")
        normalized_ids = list(dict.fromkeys(str(value) for value in evidence_ids))
        allowed_ids = {str(value) for value in episode["evidence_ids"]}
        if not set(normalized_ids) <= allowed_ids:
            raise ValueError("citation references evidence outside its episode")
        start = float(citation.get("start_time"))
        end = float(citation.get("end_time"))
        if start != float(episode["start_time"]) or end != float(episode["end_time"]):
            raise ValueError("citation timestamp does not match its episode")
        normalized_citations.append({
            "episode_id": episode_id,
            "evidence_ids": normalized_ids,
            "start_time": start,
            "end_time": end,
        })
        cited_episode_ids.append(episode_id)
        cited_evidence_ids.extend(normalized_ids)
        cited_episodes.append(episode)

    declared_episode_ids = {str(value) for value in answer.get("episode_ids", [])}
    declared_evidence_ids = {str(value) for value in answer.get("evidence_ids", [])}
    if declared_episode_ids != set(cited_episode_ids):
        raise ValueError("episode_ids must exactly match citations")
    if declared_evidence_ids != set(cited_evidence_ids):
        raise ValueError("evidence_ids must exactly match citations")

    allowed_speakers = {
        str(speaker)
        for episode in cited_episodes
        for speaker in episode.get("speakers", [])
    }
    speakers = list(dict.fromkeys(str(value) for value in answer.get("speakers", [])))
    if not set(speakers) <= allowed_speakers:
        raise ValueError("answer contains a speaker not supported by cited episodes")

    confidence = str(answer.get("confidence", "")).lower()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError("answer confidence must be high, medium, or low")
    uncertain = any(_episode_is_uncertain(episode) for episode in cited_episodes)
    uncertainty_note = str(answer.get("uncertainty_note", "")).strip()
    if uncertain and (confidence != "low" or not uncertainty_note):
        raise ValueError("uncertain evidence requires low confidence and an uncertainty note")

    text = str(answer.get("answer", "")).strip()
    if not text:
        raise ValueError("answer must be a non-empty string")
    for evidence_id in set(cited_evidence_ids):
        if evidence_id not in text:
            raise ValueError("answer text must include every cited evidence_id")
    mentioned_speakers = set(re.findall(r"\bSPEAKER_[A-Za-z0-9]+\b", text))
    if not mentioned_speakers <= allowed_speakers:
        raise ValueError("answer text contains a speaker not supported by cited episodes")
    for citation in normalized_citations:
        if not _contains_timestamp(text, citation["start_time"], citation["end_time"]):
            raise ValueError("answer text must include every cited timestamp")
    if uncertain and not any(marker in text.lower() for marker in ("不确定", "uncertain", "低置信", "low confidence")):
        raise ValueError("answer text must explicitly mention uncertainty")

    timestamps = [
        f"{item['start_time']:.3f}-{item['end_time']:.3f}s"
        for item in normalized_citations
    ]
    return {
        "answer": text,
        "episode_ids": list(dict.fromkeys(cited_episode_ids)),
        "evidence_ids": list(dict.fromkeys(cited_evidence_ids)),
        "citations": normalized_citations,
        "evidence": normalized_citations,
        "speakers": speakers,
        "speaker": ", ".join(speakers),
        "timestamp": ", ".join(timestamps),
        "confidence": confidence,
        "uncertainty_note": uncertainty_note,
        "insufficient_evidence": False,
        "question": question,
        "query": question,
    }


def _episode_is_uncertain(episode: dict[str, Any]) -> bool:
    confidence = episode.get("confidence", "low")
    low_confidence = confidence == "low" or (
        isinstance(confidence, (int, float)) and float(confidence) < 0.5
    )
    return (
        episode.get("event_type") == "uncertainty"
        or float(episode.get("overlap_score", 0.0)) > 0.6
        or low_confidence
    )


def _contains_timestamp(text: str, start: float, end: float) -> bool:
    candidates = {
        f"{start}-{end}",
        f"{start:.1f}-{end:.1f}",
        f"{start:.2f}-{end:.2f}",
        f"{start:.3f}-{end:.3f}",
        f"{start}–{end}",
        f"{start:.1f}–{end:.1f}",
        f"{start:.2f}–{end:.2f}",
        f"{start:.3f}–{end:.3f}",
    }
    compact = text.replace(" ", "")
    return any(candidate in compact for candidate in candidates)


__all__ = ["validate_qa_answer"]
