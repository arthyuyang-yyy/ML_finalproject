"""Evidence-cited deterministic answer when LLM is unavailable."""

from typing import Any

from src.utils import confidence_level


def fallback_answer(
    question: str,
    retrieved_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic, evidence-cited answer from the top retrieved episode."""
    from src.qa.answer_validator import validate_qa_answer

    episode = retrieved_episodes[0]
    evidence_ids = list(dict.fromkeys(str(value) for value in episode["evidence_ids"]))
    start = float(episode["start_time"])
    end = float(episode["end_time"])
    confidence = confidence_level(episode.get("confidence", "low"))
    uncertain = _is_uncertain(episode)
    if uncertain:
        confidence = "low"

    content = str(episode["content"]).strip()
    evidence_text = str(episode.get("evidence_text", "")).strip()
    if not content and not evidence_text:
        return _insufficient_answer(question)

    statement = content or evidence_text
    citation_text = ", ".join(evidence_ids)
    timestamp = _format_timestamp(start, end)
    answer = (
        f"{statement} 证据来自 {citation_text}，时间范围是 {timestamp}，"
        f"置信度为 {confidence}。"
    )
    uncertainty_note = ""
    if uncertain:
        uncertainty_note = str(episode.get("uncertainty_note", "")).strip()
        if not uncertainty_note:
            uncertainty_note = "该证据存在高重叠或低置信度，结论具有不确定性。"
        answer += f" 注意：该结论不确定。{uncertainty_note}"

    return validate_qa_answer(
        {
            "answer": answer,
            "episode_ids": [str(episode["episode_id"])],
            "evidence_ids": evidence_ids,
            "citations": [{
                "episode_id": str(episode["episode_id"]),
                "evidence_ids": evidence_ids,
                "start_time": start,
                "end_time": end,
            }],
            "speakers": [str(value) for value in episode.get("speakers", [])],
            "confidence": confidence,
            "uncertainty_note": uncertainty_note,
            "insufficient_evidence": False,
        },
        question,
        retrieved_episodes,
    )


def _is_uncertain(episode: dict[str, Any]) -> bool:
    return (
        episode.get("event_type") == "uncertainty"
        or confidence_level(episode.get("confidence", "low")) == "low"
        or float(episode.get("overlap_score", 0.0)) > 0.6
    )


def _format_timestamp(start: float, end: float) -> str:
    return f"{start:.3f}-{end:.3f}s"


def _insufficient_answer(question: str) -> dict[str, Any]:
    return {
        "answer": "现有会议证据不足，无法确定。",
        "episode_ids": [],
        "evidence_ids": [],
        "citations": [],
        "evidence": [],
        "speakers": [],
        "speaker": "",
        "timestamp": "",
        "confidence": "low",
        "uncertainty_note": "No relevant episodic memory was retrieved.",
        "insufficient_evidence": True,
        "question": question,
        "query": question,
    }


__all__ = ["fallback_answer"]
