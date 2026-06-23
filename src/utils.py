"""Shared lightweight utilities."""

import os
from pathlib import Path
from typing import Any


def load_dotenv(dotenv_path: Path | str | None = None) -> None:
    """Load ``.env`` file into ``os.environ`` without any external dependency.

    If *dotenv_path* is not given, looks for ``.env`` in the project root
    (two levels above this module). Lines starting with ``#`` are comments,
    blank lines are skipped, and lines without ``=`` are ignored.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    env_file = Path(dotenv_path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


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
