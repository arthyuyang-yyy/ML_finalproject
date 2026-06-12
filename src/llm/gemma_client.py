"""Gemma client abstraction with pluggable local and remote backends.

The default behaviour is intentionally local and deterministic.  When a real
backend is configured (via environment variables or explicit constructor args)
``generate_json`` will route prompts to it, parse the response, and surface
errors conservatively.

Quick start with Ollama::

    export LLM_BACKEND=ollama
    export OLLAMA_MODEL=gemma3
    python -c "from src.llm.gemma_client import GemmaClient; \
               c = GemmaClient(); print(c.generate_json('Say hi'))"

Quick start with an OpenAI-compatible API::

    export LLM_BACKEND=openai
    export OPENAI_API_KEY=sk-...
    export OPENAI_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from .backends import LLMBackend, OllamaBackend, auto_backend
from .json_repair import parse_or_repair_json

GemmaGenerator = Callable[[str], dict[str, Any] | str]
logger = logging.getLogger(__name__)


class _CallableBackend(LLMBackend):
    """Wrap a legacy callable generator into the :class:`LLMBackend` interface."""

    def __init__(self, generator: GemmaGenerator) -> None:
        self.generator = generator

    def generate(self, prompt: str, **kwargs: Any) -> str:
        result = self.generator(prompt)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


class GemmaClient:
    """JSON-generation interface used by event extraction and QA.

    Args:
        backend: An explicit :class:`LLMBackend` instance, or ``None`` to
            trigger :func:`auto_backend` on first use.
        generator: Legacy callable ``(prompt) -> dict | str``.  If provided
            and *backend* is omitted, the callable is wrapped as a backend.
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        generator: GemmaGenerator | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        elif generator is not None:
            self._backend = _CallableBackend(generator)
        else:
            self._backend = None
        self._resolved: LLMBackend | None | object = _UNSET

    @property
    def backend(self) -> LLMBackend | None:
        if self._resolved is _UNSET:
            self._resolved = self._backend if self._backend is not None else auto_backend()
        return self._resolved  # type: ignore[return-value]

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        """Run the configured backend and return parsed JSON or a fallback dict.

        When no backend is available a deterministic offline placeholder is
        returned so the pipeline never crashes on missing LLM dependencies.
        """
        backend = self.backend
        if backend is None:
            return _fallback_response(prompt)

        try:
            raw = backend.generate(prompt)
        except Exception as exc:
            return _error_response(prompt, exc)

        if not isinstance(raw, str) or not raw.strip():
            return _error_response(prompt, ValueError("backend returned empty output"))

        try:
            return parse_or_repair_json(raw)
        except ValueError as exc:
            return _error_response(prompt, exc, raw_output=raw)


class OllamaGemmaClient(GemmaClient):
    """Gemma JSON client backed by a local Ollama server.

    This is a backwards-compatible wrapper around :class:`OllamaBackend`.
    """

    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        backend = OllamaBackend(base_url=base_url, model=model, timeout=timeout)
        super().__init__(backend=backend)


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


def run_gemma(prompt: str, client: GemmaClient | None = None) -> str:
    """Run a configured Gemma backend and return raw text.

    When no backend is configured, or the configured backend is unavailable,
    return an empty JSON object so callers can continue through their
    deterministic evidence-only fallback.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    configured = client or create_gemma_client(
        os.environ.get("GEMMA_BACKEND", "none"),
        model=os.environ.get("GEMMA_MODEL", "gemma3:4b"),
        base_url=os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    if configured is None:
        return "{}"
    try:
        output = configured.generate_json(prompt)
    except Exception as exc:
        logger.warning("Gemma backend failed; using deterministic fallback: %s", exc)
        return "{}"
    return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)


_UNSET = object()


def _fallback_response(prompt: str) -> dict[str, Any]:
    return {
        "meeting_id": "",
        "meeting_summary": "",
        "events": [],
        "uncertainty_note": "Gemma backend is not configured; used deterministic fallback.",
        "prompt_length": len(prompt),
    }


def _error_response(
    prompt: str,
    exc: Exception,
    raw_output: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "meeting_id": "",
        "meeting_summary": "",
        "events": [],
        "uncertainty_note": f"LLM backend error: {exc}",
        "prompt_length": len(prompt),
        "error": str(exc),
        "fallback": True,
    }
    if raw_output is not None:
        result["raw_output_preview"] = raw_output[:1000]
    return result


__all__ = [
    "GemmaClient",
    "GemmaGenerator",
    "OllamaGemmaClient",
    "create_gemma_client",
    "run_gemma",
]
