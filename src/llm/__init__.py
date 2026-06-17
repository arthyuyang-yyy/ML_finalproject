"""LLM event extraction and QA package."""

from .backends import (
    LLMBackend,
    OllamaBackend,
    OpenAIBackend,
    TransformersBackend,
    auto_backend,
)
from .event_extractor import extract_meeting_events, extract_meeting_events_file
from .event_validator import validate_meeting_event, validate_meeting_events_document
from .gemma_client import (
    GemmaClient,
    OllamaGemmaClient,
    create_gemma_client,
    run_gemma,
)
from .json_repair import parse_or_repair_json

__all__ = [
    "auto_backend",
    "create_gemma_client",
    "extract_meeting_events",
    "extract_meeting_events_file",
    "GemmaClient",
    "LLMBackend",
    "OllamaBackend",
    "OllamaGemmaClient",
    "OpenAIBackend",
    "parse_or_repair_json",
    "run_gemma",
    "TransformersBackend",
    "validate_meeting_event",
    "validate_meeting_events_document",
]
