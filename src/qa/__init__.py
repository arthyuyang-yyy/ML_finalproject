"""Question-answering package facade."""

from typing import Any

from src.llm.gemma_client import GemmaClient
from src.memory.retriever import EmbeddingBackend, retrieve_episodes

from .answerer import answer_question
from .answer_validator import validate_qa_answer
from .prompts import build_qa_prompt, build_qa_repair_prompt


def retrieve_relevant_memory(
    query: str,
    top_k: int = 5,
    *,
    embedding_backend: EmbeddingBackend | None = None,
) -> list[dict[str, Any]]:
    return retrieve_episodes(
        question=query,
        top_k=top_k,
        embedding_backend=embedding_backend,
    )


def answer_question_with_evidence(
    query: str,
    retrieved_episodes: list[dict[str, Any]],
    client: GemmaClient | None = None,
) -> dict[str, Any]:
    return answer_question(query, retrieved_episodes, client=client)

__all__ = [
    "answer_question",
    "answer_question_with_evidence",
    "build_qa_prompt",
    "build_qa_repair_prompt",
    "retrieve_relevant_memory",
    "validate_qa_answer",
]
