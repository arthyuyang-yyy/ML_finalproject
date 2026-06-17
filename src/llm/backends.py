"""Pluggable LLM backends for local and remote inference.

Three concrete backends are provided:

- :class:`OllamaBackend` – HTTP API against a local (or remote) Ollama server.
- :class:`OpenAIBackend` – Any OpenAI-compatible chat API (OpenAI, DeepSeek,
  vLLM, etc.).
- :class:`TransformersBackend` – Direct ``transformers`` + ``torch`` loading
  for Gemma-style models.

All backends are imported lazily so that installing heavy dependencies
(torch, transformers, openai) remains optional.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """Minimal interface accepted by :class:`GemmaClient`."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return raw text output for *prompt*."""
        ...


class OllamaBackend(LLMBackend):
    """Ollama ``/api/generate`` backend with JSON-mode support.

    Environment variables:
        ``OLLAMA_URL`` – base URL (default ``http://localhost:11434``).
        ``OLLAMA_MODEL`` – model tag (default ``gemma3``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "gemma3")
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": float(kwargs.get("temperature", 0.0)),
            },
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc

        if response.get("done") is not True:
            raise RuntimeError(f"Ollama response incomplete: {response}")
        return str(response.get("response", ""))


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat-completions backend.

    Environment variables:
        ``OPENAI_API_KEY`` – required unless passed explicitly.
        ``OPENAI_BASE_URL`` – default ``https://api.openai.com/v1``.
        ``OPENAI_MODEL`` – default ``gpt-4o-mini``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "Install `openai` to use the OpenAI backend: pip install openai"
            ) from exc

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", 0.0)),
                response_format={"type": "json_object"},
            )
        except openai.OpenAIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        content = response.choices[0].message.content
        return content or ""


class TransformersBackend(LLMBackend):
    """Local HuggingFace ``transformers`` backend for Gemma-style models.

    The model and tokenizer are loaded lazily and cached.  Chat templates are
    applied automatically when available.

    Environment variables:
        ``TRANSFORMERS_MODEL`` – model id or path
        (default ``google/gemma-3-4b-it``).
        ``TRANSFORMERS_DEVICE`` – ``auto``, ``cpu``, ``cuda``, ``cuda:0``, …
        (default ``auto``).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "TRANSFORMERS_MODEL", "google/gemma-3-4b-it"
        )
        self.device = device or os.environ.get("TRANSFORMERS_DEVICE", "auto")
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install `transformers` and `torch` to use the Transformers backend."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Padding side needed for batched generation; safe for single prompts too.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map: str | None = "auto" if self.device == "auto" else None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None and self.device not in {None, "", "auto"}:
            model = model.to(self.device)

        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def generate(self, prompt: str, **kwargs: Any) -> str:
        model, tokenizer = self._ensure_model()

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        else:
            # Fallback for models without a chat template.
            text = f"User: {prompt}\nAssistant: "
            inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True)

        import torch

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        temperature = float(kwargs.get("temperature", 0.0))
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(kwargs.get("max_new_tokens", self.max_new_tokens)),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _ollama_available() -> bool:
    """Best-effort probe for a running Ollama server."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434",
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def auto_backend() -> LLMBackend | None:
    """Return a backend based on environment variables or auto-probing.

    Priority:
        1. ``LLM_BACKEND`` env var (``ollama``, ``openai``, ``transformers``).
        2. Auto-detect Ollama on localhost.
        3. Auto-detect ``OPENAI_API_KEY``.
        4. ``None`` → deterministic fallback.
    """
    backend_type = os.environ.get("LLM_BACKEND", "").lower()

    if backend_type == "ollama":
        return OllamaBackend()
    if backend_type == "openai":
        return OpenAIBackend()
    if backend_type == "transformers":
        return TransformersBackend()
    if backend_type in {"none", "mock"}:
        return None

    # Auto-detection (empty string or unset means auto)
    if os.environ.get("OLLAMA_URL") or _ollama_available():
        return OllamaBackend()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIBackend()
    return None


__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "TransformersBackend",
    "auto_backend",
]
