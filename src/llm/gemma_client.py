"""Gemma client abstraction.

The default client is intentionally local and deterministic. Replace
``generate_json`` with an Ollama, llama.cpp, or hosted Gemma adapter when the
runtime target is fixed.
"""

from typing import Any


class GemmaClient:
    """Minimal JSON-generation interface used by event extraction."""

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Return a deterministic placeholder response for offline demos."""
        return {
            "events": [],
            "uncertainty_note": "Gemma backend is not configured; used deterministic fallback.",
            "prompt_length": len(prompt),
        }
