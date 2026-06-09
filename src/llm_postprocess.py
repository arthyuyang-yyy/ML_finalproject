"""Metadata-aware and uncertainty-preserving LLM post-processing interfaces."""

import json


SYSTEM_INSTRUCTIONS = """You process multi-speaker meeting records.
Do not invent information.
Preserve uncertainty in high-overlap regions.
If speaker attribution is uncertain, mark it as uncertain.
Every decision and action item must be linked to evidence timestamps.
Return only claims supported by the supplied segments and memory context."""


def build_llm_prompt_with_metadata(
    segments: list[dict], memory_context: list[dict] | None = None
) -> str:
    """Build a constrained prompt containing segment metadata and memory."""
    payload = {
        "segments": segments,
        "memory_context": memory_context or [],
    }
    return f"{SYSTEM_INSTRUCTIONS}\n\nINPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def uncertainty_aware_correction(segments: list[dict]) -> list[dict]:
    """Correct transcripts while retaining candidates and uncertainty markers."""
    # TODO: call an LLM with build_llm_prompt_with_metadata and validate output.
    raise NotImplementedError("LLM correction is not implemented yet.")


def generate_evidence_based_summary(segments: list[dict]) -> dict:
    """Generate a summary whose claims, decisions, and actions cite evidence."""
    # TODO: call an LLM and reject claims without timestamped evidence.
    raise NotImplementedError("Evidence-based summary generation is not implemented yet.")
