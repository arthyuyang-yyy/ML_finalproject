"""Gemma client abstraction.

The default client is intentionally local and deterministic. Replace
``generate_json`` with an Ollama, llama.cpp, or hosted Gemma adapter when the
runtime target is fixed.
"""

from collections.abc import Callable
import json
from typing import Any
from urllib.request import Request, urlopen

GemmaGenerator = Callable[[str], dict[str, Any] | str]


class GemmaClient:
    """Minimal JSON-generation interface used by event extraction."""

    def __init__(self, generator: GemmaGenerator | None = None) -> None:
        self.generator = generator

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        """Run the configured Gemma generator."""
        if self.generator is None:
            raise ValueError("GemmaClient has no generator configured")
        return self.generator(prompt)


class OllamaGemmaClient(GemmaClient):
    """Gemma JSON client backed by a local Ollama server."""

    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result.get("response", ""))


def create_gemma_client(
    backend: str = "none",
    model: str = "gemma3:4b",
    base_url: str = "http://127.0.0.1:11434",
) -> GemmaClient | None:
    """Build a configured Gemma backend."""
    normalized = backend.lower()
    if normalized in {"", "none", "fallback"}:
        return None
    if normalized == "ollama":
        return OllamaGemmaClient(model=model, base_url=base_url)
    raise ValueError("gemma backend must be one of: none, ollama")


__all__ = ["GemmaClient", "GemmaGenerator", "OllamaGemmaClient", "create_gemma_client"]
