"""Shared lightweight utilities."""

from typing import Any


def validate_score(score: float, name: str = "score") -> float:
    """Validate and return a probability-like score in the range [0, 1]."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return score


def required_text(value: Any, name: str) -> str:
    """Validate that *value* is a non-empty string and return it stripped."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def confidence_level(value: Any) -> str:
    """Normalize a confidence indicator to ``"high"``, ``"medium"``, or ``"low"``."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized
        raise ValueError("confidence must be high, medium, or low")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
    raise ValueError("confidence must be high, medium, low, or a score between 0 and 1")
