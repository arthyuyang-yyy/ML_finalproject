"""Simple evidence-citation metrics for meeting QA experiments."""

from typing import Any


def citation_rate(answers: list[dict[str, Any]]) -> float:
    """Return the fraction of answers carrying at least one evidence citation."""
    if not answers:
        return 0.0
    cited = sum(bool(answer.get("evidence_ids") or answer.get("evidence")) for answer in answers)
    return cited / len(answers)


def timestamp_citation_rate(answers: list[dict[str, Any]]) -> float:
    """Return the fraction of answers carrying a non-empty timestamp."""
    if not answers:
        return 0.0
    return sum(bool(answer.get("timestamp")) for answer in answers) / len(answers)


__all__ = ["citation_rate", "timestamp_citation_rate"]
