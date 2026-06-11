"""Gemma client abstraction.

The default client is intentionally local and deterministic. Replace
``generate_json`` with an Ollama, llama.cpp, or hosted Gemma adapter when the
runtime target is fixed.
"""

from collections.abc import Callable
from typing import Any

GemmaGenerator = Callable[[str], dict[str, Any] | str]


class GemmaClient:
    """Minimal JSON-generation interface used by event extraction."""

    def __init__(self, generator: GemmaGenerator | None = None) -> None:
        self.generator = generator

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        """Run the configured Gemma generator or return an offline placeholder."""
        if self.generator is not None:
            return self.generator(prompt)
        return {
            "meeting_id": "",
            "meeting_summary": "",
            "events": [],
            "uncertainty_note": "Gemma backend is not configured; used deterministic fallback.",
            "prompt_length": len(prompt),
        }


__all__ = ["GemmaClient", "GemmaGenerator"]
