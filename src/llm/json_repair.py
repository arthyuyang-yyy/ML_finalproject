"""Parse and conservatively repair JSON emitted by an LLM."""

import json
import re
from typing import Any


def parse_or_repair_json(raw_output: str) -> dict[str, Any]:
    """Parse JSON, removing common Markdown fences and trailing commas."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("LLM output must be a non-empty string")

    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM output is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM output must decode to a JSON object")
    return payload


__all__ = ["parse_or_repair_json"]
