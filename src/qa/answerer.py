"""Evidence-constrained question answering over retrieved episodes."""

from typing import Any

from src.llm.gemma_client import GemmaClient
from src.llm.json_repair import parse_or_repair_json
from src.llm.event_extractor import _serialize_output
from src.utils import confidence_level

from .answer_validator import validate_qa_answer
from .prompts import build_qa_prompt, build_qa_repair_prompt


def answer_question(
    question: str,
    retrieved_episodes: list[dict[str, Any]],
    client: GemmaClient | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Generate an evidence-backed answer using only retrieved episodes.

    A configured Gemma client must return structured JSON. Its citations and
    uncertainty claims are validated against the supplied top-k episodes. An
    unavailable or invalid model falls back to a deterministic cited answer.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(retrieved_episodes, list):
        raise ValueError("retrieved_episodes must be a list")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if not retrieved_episodes:
        return _insufficient_answer(question)

    _validate_retrieved_episodes(retrieved_episodes)
    if client is not None:
        prompt = build_qa_prompt(question, retrieved_episodes)
        previous_output = ""
        previous_error = ""
        for attempt in range(max_attempts):
            if attempt:
                prompt = build_qa_repair_prompt(
                    question,
                    retrieved_episodes,
                    previous_output,
                    previous_error,
                )
            raw_output = client.generate_json(prompt)
            previous_output = _serialize_output(raw_output)
            try:
                payload = raw_output if isinstance(raw_output, dict) else parse_or_repair_json(raw_output)
                return validate_qa_answer(payload, question, retrieved_episodes)
            except (TypeError, ValueError) as exc:
                previous_error = str(exc)

    return _fallback_answer(question, retrieved_episodes)


def _fallback_answer(
    question: str,
    retrieved_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
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


def _validate_retrieved_episodes(episodes: list[dict[str, Any]]) -> None:
    required = {
        "episode_id",
        "content",
        "evidence_ids",
        "start_time",
        "end_time",
        "confidence",
    }
    seen_ids: set[str] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"retrieved_episodes[{index}] must be a dict")
        missing = required - episode.keys()
        if missing:
            raise ValueError(f"retrieved episode is missing fields: {sorted(missing)}")
        episode_id = str(episode["episode_id"]).strip()
        if not episode_id or episode_id in seen_ids:
            raise ValueError("retrieved episode IDs must be non-empty and unique")
        seen_ids.add(episode_id)
        evidence_ids = episode["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("retrieved episode evidence_ids must be a non-empty list")
        start = float(episode["start_time"])
        end = float(episode["end_time"])
        if end <= start:
            raise ValueError("retrieved episode end_time must be greater than start_time")
        confidence_level(episode["confidence"])


def _is_uncertain(episode: dict[str, Any]) -> bool:
    return (
        episode.get("event_type") == "uncertainty"
        or confidence_level(episode.get("confidence", "low")) == "low"
        or float(episode.get("overlap_score", 0.0)) > 0.6
    )


def _format_timestamp(start: float, end: float) -> str:
    return f"{start:.3f}-{end:.3f}s"


__all__ = ["answer_question"]
