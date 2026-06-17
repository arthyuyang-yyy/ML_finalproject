"""Lightweight retrieval over episodic memory using hash embeddings."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from .episodic_store import DEFAULT_MEMORY_PATH, read_episodes
from src.fallbacks.embeddings import HashingEmbeddingBackend

# MVP weights: keep hash-embedding similarity as the primary signal while using
# keyword hits as a deterministic boost. Recalibrate with retrieval metrics
# before adding recency, importance, or overlap penalties back into the score.
EMBEDDING_WEIGHT = 0.70
KEYWORD_WEIGHT = 0.30
DEFAULT_MIN_RELEVANCE_SCORE = 0.15
INDEX_FIELDS = ("content", "topic", "event_type", "speakers", "evidence_text")
QUERY_ALIASES = {
    "不确定": ("uncertainty", "overlap", "重叠"),
    "重叠": ("uncertainty", "overlap", "不确定"),
    "决定": ("decision",),
    "决策": ("decision",),
    "负责": ("action_item", "task", "owner"),
    "谁负责": ("action_item", "task", "owner"),
}


class EmbeddingBackend(Protocol):
    """Minimal embedding interface used by the retriever."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts into equal-length vectors."""


class SentenceTransformerEmbeddingBackend:
    """Compatibility placeholder; retrieval now defaults to hash embeddings."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ImportError("external sentence-transformer embeddings are not used by this MVP")


def retrieve_episodes(
    question: str,
    episodes: list[dict[str, Any]] | None = None,
    path: str | Path = DEFAULT_MEMORY_PATH,
    top_k: int = 5,
    embedding_backend: EmbeddingBackend | None = None,
    min_score: float = 0.0,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    meeting_id: str | None = None,
    speaker: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[dict[str, Any]]:
    """Return top-k episodes using custom BLAKE2 character n-gram embeddings."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k <= 0:
        return []
    if min_score < 0.0:
        raise ValueError("min_score must be non-negative")
    if min_relevance_score < 0.0:
        raise ValueError("min_relevance_score must be non-negative")

    records = list(episodes if episodes is not None else read_episodes(path))
    records = _filter_episodes(records, meeting_id, speaker, start_time, end_time)
    if not records:
        return []

    query_text = _expand_query(question)
    documents = [_index_text(episode) for episode in records]
    keyword_scores = _normalized_keyword_scores(query_text, documents)
    backend = embedding_backend or _default_embedding_backend()
    vectors = backend.encode([query_text, *documents])
    if len(vectors) != len(documents) + 1:
        raise ValueError("embedding backend returned an unexpected number of vectors")
    embedding_scores = [max(0.0, _cosine_similarity(vectors[0], vector)) for vector in vectors[1:]]

    ranked: list[tuple[float, dict[str, Any]]] = []
    for index, episode in enumerate(records):
        final_score = EMBEDDING_WEIGHT * embedding_scores[index] + KEYWORD_WEIGHT * keyword_scores[index]
        if keyword_scores[index] == 0.0 and embedding_scores[index] < 0.35:
            continue
        result = dict(episode)
        result["retrieval"] = {
            "final_score": round(final_score, 6),
            "relevance_score": round(final_score, 6),
            "embedding_similarity": round(embedding_scores[index], 6),
            "keyword_score": round(keyword_scores[index], 6),
            "embedding_backend": type(backend).__name__,
        }
        if final_score >= min_score and final_score >= min_relevance_score:
            ranked.append((final_score, result))

    ranked.sort(key=lambda item: (item[0], item[1]["retrieval"]["keyword_score"]), reverse=True)
    return [episode for _, episode in ranked[:top_k]]


def _filter_episodes(
    episodes: list[dict[str, Any]],
    meeting_id: str | None,
    speaker: str | None,
    start_time: float | None,
    end_time: float | None,
) -> list[dict[str, Any]]:
    if start_time is not None and end_time is not None and end_time < start_time:
        raise ValueError("end_time filter must be greater than or equal to start_time")
    filtered: list[dict[str, Any]] = []
    for episode in episodes:
        if meeting_id and str(episode.get("meeting_id", "")) != meeting_id:
            continue
        if speaker and speaker not in {str(value) for value in episode.get("speakers", [])}:
            continue
        episode_start = float(episode.get("start_time", 0.0))
        episode_end = float(episode.get("end_time", episode_start))
        if start_time is not None and episode_end < start_time:
            continue
        if end_time is not None and episode_start > end_time:
            continue
        filtered.append(episode)
    return filtered


def _default_embedding_backend() -> HashingEmbeddingBackend:
    """Return the project's dependency-free BLAKE2 hash embedding backend."""
    return HashingEmbeddingBackend()


def _index_text(episode: dict[str, Any]) -> str:
    values: list[str] = []
    for field in INDEX_FIELDS:
        value = episode.get(field, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values)


def _normalized_keyword_scores(query: str, documents: list[str]) -> list[float]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0] * len(documents)
    scores: list[float] = []
    for document in documents:
        counts = Counter(_tokenize(document))
        scores.append(sum(counts[token] for token in query_tokens))
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


def _expand_query(question: str) -> str:
    terms = [question]
    lowered = question.lower()
    for trigger, aliases in QUERY_ALIASES.items():
        if trigger in lowered:
            terms.extend(aliases)
    return " ".join(terms)


def _tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("_", " ")
    latin_tokens = re.findall(r"[a-z0-9]+(?:[+.-][a-z0-9]+)*", normalized)
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk_tokens: list[str] = []
    for sequence in cjk_sequences:
        cjk_tokens.extend(sequence)
        cjk_tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return latin_tokens + cjk_tokens


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same length")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


__all__ = [
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "SentenceTransformerEmbeddingBackend",
    "retrieve_episodes",
]
