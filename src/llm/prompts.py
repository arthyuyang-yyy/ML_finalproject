"""Prompt builders for evidence-aware event extraction."""

import json
from typing import Any


def build_event_extraction_prompt(evidence_segments: list[dict[str, Any]]) -> str:
    """Build a compact prompt that requires evidence_id citations."""
    payload = json.dumps(evidence_segments, ensure_ascii=False, indent=2)
    return (
        "Extract meeting events as JSON. Every event must cite evidence_ids, "
        "preserve uncertainty for high-overlap candidates, and avoid unsupported claims.\n\n"
        f"Evidence segments:\n{payload}"
    )
