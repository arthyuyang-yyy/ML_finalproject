"""Evidence-constrained question answering over retrieved episodes."""

import logging
from typing import Any

from src.llm.gemma_client import GemmaClient
from src.llm.json_repair import parse_or_repair_json
from src.llm.event_extractor import _serialize_output
from src.fallbacks.qa import fallback_answer
from src.utils import confidence_level

from .answer_validator import validate_qa_answer
from .prompts import build_qa_prompt, build_qa_repair_prompt

logger = logging.getLogger(__name__)


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
            try:
                raw_output = client.generate_json(prompt)
            except Exception as exc:
                logger.warning("Gemma QA failed; using deterministic fallback: %s", exc)
                break
            previous_output = _serialize_output(raw_output)
            try:
                payload = raw_output if isinstance(raw_output, dict) else parse_or_repair_json(raw_output)
                return validate_qa_answer(payload, question, retrieved_episodes)
            except (TypeError, ValueError) as exc:
                previous_error = str(exc)

    return fallback_answer(question, retrieved_episodes)


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


__all__ = ["answer_question"]
