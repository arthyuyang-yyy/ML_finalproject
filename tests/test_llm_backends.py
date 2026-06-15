"""Tests for pluggable LLM backends."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from src.llm.backends import (
    LLMBackend,
    OllamaBackend,
    OpenAIBackend,
    TransformersBackend,
    auto_backend,
)
from src.llm.gemma_client import GemmaClient


class FakeBackend(LLMBackend):
    """Deterministic backend for unit tests."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses) if responses else []
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({"reply": prompt})


class OllamaBackendTests(unittest.TestCase):
    def test_generate_hits_api_and_returns_response(self) -> None:
        backend = OllamaBackend(base_url="http://fake-ollama:11434", model="gemma3")
        fake_response = {"response": '{"events": []}', "done": True}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                json.dumps(fake_response).encode("utf-8")
            )
            result = backend.generate("hello")

        self.assertEqual(result, '{"events": []}')
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://fake-ollama:11434/api/generate")
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "gemma3")
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])

    def test_generate_raises_on_http_error(self) -> None:
        backend = OllamaBackend(base_url="http://localhost:11434", model="gemma3")
        with patch("urllib.request.urlopen") as mock_urlopen:
            import urllib.error

            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="http://localhost:11434/api/generate",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=None,
            )
            with self.assertRaisesRegex(RuntimeError, "Ollama HTTP 500"):
                backend.generate("hello")

    def test_generate_raises_on_incomplete_response(self) -> None:
        backend = OllamaBackend()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                json.dumps({"response": "", "done": False}).encode("utf-8")
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                backend.generate("hello")

    def test_env_vars_override_defaults(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_URL": "http://ollama:11434", "OLLAMA_MODEL": "mistral"}):
            backend = OllamaBackend()
            self.assertEqual(backend.base_url, "http://ollama:11434")
            self.assertEqual(backend.model, "mistral")


class OpenAIBackendTests(unittest.TestCase):
    def test_generate_calls_chat_completions(self) -> None:
        backend = OpenAIBackend(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        )
        fake_message = MagicMock()
        fake_message.content = '{"events": []}'
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = backend.generate("hello", temperature=0.5)

        self.assertEqual(result, '{"events": []}')
        fake_client.chat.completions.create.assert_called_once()
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")
        self.assertEqual(call_kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(call_kwargs["messages"], [{"role": "user", "content": "hello"}])

    def test_generate_raises_on_api_error(self) -> None:
        backend = OpenAIBackend(api_key="sk-test")
        fake_openai = MagicMock()
        fake_openai.OpenAIError = type("OpenAIError", (Exception,), {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = fake_openai.OpenAIError("rate limit")
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            with self.assertRaisesRegex(RuntimeError, "OpenAI API error"):
                backend.generate("hello")

    def test_import_error_when_openai_missing(self) -> None:
        backend = OpenAIBackend(api_key="sk-test")
        with patch.dict("sys.modules", {"openai": None}):
            with self.assertRaises(ImportError):
                backend.generate("hello")


class TransformersBackendTests(unittest.TestCase):
    def test_generate_with_chat_template(self) -> None:
        backend = TransformersBackend(model_name="dummy", device="cpu")
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 2
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = "template"
        mock_inputs = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_tokenizer.apply_chat_template.return_value = mock_inputs
        mock_tokenizer.decode.return_value = '{"events": []}'

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_generated = MagicMock()
        mock_generated.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = mock_generated

        fake_torch = MagicMock()
        fake_torch.no_grad = MagicMock()
        fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with patch.object(backend, "_ensure_model", return_value=(mock_model, mock_tokenizer)):
                result = backend.generate("hello", temperature=0.0)

        self.assertEqual(result, '{"events": []}')
        mock_model.generate.assert_called_once()
        gen_kwargs = mock_model.generate.call_args.kwargs
        self.assertFalse(gen_kwargs["do_sample"])

    def test_generate_with_temperature(self) -> None:
        backend = TransformersBackend(model_name="dummy", device="cpu")
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_inputs = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = '{"events": []}'

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_generated = MagicMock()
        mock_generated.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = mock_generated

        fake_torch = MagicMock()
        fake_torch.no_grad = MagicMock()
        fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with patch.object(backend, "_ensure_model", return_value=(mock_model, mock_tokenizer)):
                backend.generate("hello", temperature=0.7)

        gen_kwargs = mock_model.generate.call_args.kwargs
        self.assertTrue(gen_kwargs["do_sample"])
        self.assertEqual(gen_kwargs["temperature"], 0.7)

    def test_import_error_when_transformers_missing(self) -> None:
        backend = TransformersBackend()
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with self.assertRaises(ImportError):
                backend.generate("hello")


class AutoBackendTests(unittest.TestCase):
    def test_explicit_ollama_env(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}, clear=False):
            backend = auto_backend()
        self.assertIsInstance(backend, OllamaBackend)

    def test_explicit_openai_env(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "openai"}, clear=False):
            backend = auto_backend()
        self.assertIsInstance(backend, OpenAIBackend)

    def test_explicit_transformers_env(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "transformers"}, clear=False):
            backend = auto_backend()
        self.assertIsInstance(backend, TransformersBackend)

    def test_none_returns_none(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "none"}, clear=False):
            backend = auto_backend()
        self.assertIsNone(backend)

    def test_auto_detects_ollama(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": ""}, clear=False):
            with patch("src.llm.backends._ollama_available", return_value=True):
                backend = auto_backend()
        self.assertIsInstance(backend, OllamaBackend)

    def test_auto_detects_openai_key(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "", "OPENAI_API_KEY": "sk-test"}, clear=False):
            with patch("src.llm.backends._ollama_available", return_value=False):
                backend = auto_backend()
        self.assertIsInstance(backend, OpenAIBackend)

    def test_auto_fallback_to_none(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": ""}, clear=False):
            with patch("src.llm.backends._ollama_available", return_value=False):
                backend = auto_backend()
        self.assertIsNone(backend)


class GemmaClientBackendTests(unittest.TestCase):
    def test_client_with_explicit_backend(self) -> None:
        fake = FakeBackend(['{"events": []}'])
        client = GemmaClient(backend=fake)
        result = client.generate_json("test")
        self.assertEqual(result, {"events": []})
        self.assertEqual(fake.prompts, ["test"])

    def test_client_with_legacy_generator(self) -> None:
        client = GemmaClient(generator=lambda p: {"echo": p})
        result = client.generate_json("hello")
        self.assertEqual(result, {"echo": "hello"})

    def test_client_backend_error_returns_safe_dict(self) -> None:
        class BrokenBackend(LLMBackend):
            def generate(self, prompt: str, **kwargs) -> str:
                raise RuntimeError("network down")

        client = GemmaClient(backend=BrokenBackend())
        result = client.generate_json("test")
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("network down", result["error"])
        self.assertTrue(result.get("fallback"))

    def test_client_empty_backend_response_returns_error(self) -> None:
        class EmptyBackend(LLMBackend):
            def generate(self, prompt: str, **kwargs) -> str:
                return "   "

        client = GemmaClient(backend=EmptyBackend())
        result = client.generate_json("test")
        self.assertIn("error", result)
        self.assertTrue(result.get("fallback"))

    def test_client_invalid_json_response_returns_error_with_preview(self) -> None:
        class BadJsonBackend(LLMBackend):
            def generate(self, prompt: str, **kwargs) -> str:
                return "not json at all"

        client = GemmaClient(backend=BadJsonBackend())
        result = client.generate_json("test")
        self.assertIn("error", result)
        self.assertIn("raw_output_preview", result)
        self.assertEqual(result["raw_output_preview"], "not json at all")

    def test_client_no_backend_returns_fallback(self) -> None:
        with patch.dict(os.environ, {"LLM_BACKEND": "none"}, clear=False):
            client = GemmaClient()
            result = client.generate_json("test")
        self.assertIn("fallback", str(result.get("uncertainty_note", "")).lower())


if __name__ == "__main__":
    unittest.main()
