"""Prompt construction for evidence-constrained meeting QA."""

import json
from typing import Any


def build_qa_prompt(question: str, retrieved_episodes: list[dict[str, Any]]) -> str:
    """Build a JSON-only QA prompt containing only retrieved episodes."""
    payload = json.dumps(retrieved_episodes, ensure_ascii=False, indent=2)
    return f"""You are the QA module of a verifiable meeting-memory system.

Answer the question using only the retrieved episodes below. Do not use outside knowledge.

Rules:
1. Use only the retrieved episodes.
2. Every factual claim must cite existing evidence_ids and the exact episode timestamp.
3. If the evidence is insufficient, set insufficient_evidence=true and say you cannot determine.
4. If any cited episode is high-overlap, event_type=uncertainty, or confidence=low, explicitly state the uncertainty and use confidence=low.
5. Do not invent speaker names, tasks, decisions, owners, or deadlines.
6. Cite only episode_id and evidence_id values present below.
7. Copy citation start_time and end_time exactly from the cited episode.
8. Return valid JSON only, without Markdown.

Return exactly this shape:
{{
  "answer": "answer text with evidence IDs and time ranges",
  "episode_ids": ["cited episode IDs"],
  "evidence_ids": ["cited evidence IDs"],
  "citations": [
    {{
      "episode_id": "existing episode ID",
      "evidence_ids": ["evidence IDs belonging to that episode"],
      "start_time": 0.0,
      "end_time": 0.0
    }}
  ],
  "speakers": ["supported speaker labels"],
  "confidence": "high|medium|low",
  "uncertainty_note": "required when cited evidence is uncertain",
  "insufficient_evidence": false
}}

Question:
{question}

Retrieved episodes:
{payload}
"""


def build_qa_repair_prompt(
    question: str,
    retrieved_episodes: list[dict[str, Any]],
    invalid_output: str,
    validation_error: str,
) -> str:
    return (
        build_qa_prompt(question, retrieved_episodes)
        + "\nThe previous answer failed validation. Regenerate the complete JSON object.\n"
        + f"Validation error: {validation_error}\n"
        + f"Previous output: {invalid_output[:4000]}\n"
    )


__all__ = ["build_qa_prompt", "build_qa_repair_prompt"]
