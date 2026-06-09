"""Shared lightweight utilities."""


def validate_score(score: float, name: str = "score") -> float:
    """Validate and return a probability-like score in the range [0, 1]."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return score
