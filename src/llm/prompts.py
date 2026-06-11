"""Prompt builders for evidence-grounded structured event extraction."""

import json
from typing import Any


def build_event_extraction_prompt(evidence_segments: list[dict[str, Any]]) -> str:
    """Require Gemma to return only evidence-backed structured JSON."""
    payload = json.dumps(evidence_segments, ensure_ascii=False, indent=2)
    return f"""You are the structured event extraction module in a meeting understanding system.

Extract meaningful events from the supplied evidence segments. Do not freely summarize and do not invent facts.

Allowed event_type values:
- decision
- action_item
- open_question
- disagreement
- uncertainty
- speaker_stance
- topic_transition

Return one JSON object with exactly this top-level shape:
{{
  "meeting_id": "string",
  "meeting_summary": "string",
  "events": [
    {{
      "event_id": "string",
      "event_type": "decision|action_item|open_question|disagreement|uncertainty|speaker_stance|topic_transition",
      "topic": "short topic label",
      "content": "string",
      "speakers": ["speaker labels"],
      "evidence_ids": ["existing evidence_id values"],
      "confidence": "high|medium|low"
    }}
  ]
}}

Additional action_item fields:
- task: required non-empty string
- owner: required speaker label, or "uncertain" when the owner is unclear
- deadline: optional string

Rules:
1. Every event must cite at least one existing evidence_id.
2. Use only information present in the evidence segments.
3. Never mark an event "high" when any cited evidence has processing_path="high_overlap_candidate".
4. Represent ambiguous overlapped speech as event_type="uncertainty" with confidence="low".
5. Do not infer speaker names, owners, deadlines, decisions, or stances that are not supported.
6. Exclude small talk and unsupported conclusions.
7. Output valid JSON only. Do not use Markdown fences or explanatory text.
8. Include a concise topic for each event when it can be derived from evidence.

Evidence segments:
{payload}
"""


def build_event_repair_prompt(
    evidence_segments: list[dict[str, Any]],
    invalid_output: str,
    validation_error: str,
) -> str:
    """Ask Gemma to regenerate after JSON or schema validation fails."""
    return (
        build_event_extraction_prompt(evidence_segments)
        + "\nThe previous output was invalid. Regenerate the complete JSON object.\n"
        + f"Validation error: {validation_error}\n"
        + f"Previous output: {invalid_output[:4000]}\n"
    )


__all__ = ["build_event_extraction_prompt", "build_event_repair_prompt"]
