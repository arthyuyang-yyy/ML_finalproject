"""BM25 and embedding hybrid retrieval over episodic memory."""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .episodic_store import DEFAULT_MEMORY_PATH, read_episodes

EMBEDDING_WEIGHT = 0.45
KEYWORD_WEIGHT = 0.25
IMPORTANCE_WEIGHT = 0.15
RECENCY_WEIGHT = 0.10
OVERLAP_PENALTY_WEIGHT = 0.20

INDEX_FIELDS = ("content", "topic", "event_type", "speakers", "evidence_text")

QUERY_EXPANSIONS = {
    "谁负责": ["action_item", "action item", "owner", "负责", "任务"],
    "负责": ["action_item", "action item", "owner", "任务"],
    "行动项": ["action_item", "action item", "task", "owner"],
    "不确定": ["uncertainty", "overlap", "uncertain", "重叠"],
    "重叠": ["uncertainty", "overlap", "不确定"],
    "决定": ["decision", "decided"],
    "决策": ["decision", "decided"],
    "上次": ["decision", "recent", "latest"],
    "question": ["open_question", "open question"],
    "uncertain": ["uncertainty", "overlap"],
    "decision": ["decision", "decided"],
    "responsible": ["action_item", "owner", "task"],
    "who": ["speaker", "owner"],
}


class EmbeddingBackend(Protocol):
    """Minimal embedding interface accepted by the hybrid retriever."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts into equal-length vectors."""


class HashingEmbeddingBackend:
    """Dependency-free multilingual character n-gram embedding baseline."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        features = _embedding_features(text)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        return _normalize_vector(vector)


class SentenceTransformerEmbeddingBackend:
    """Optional semantic embedding backend loaded only when requested."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to use the semantic embedding backend."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def retrieve_episodes(
    question: str,
    episodes: list[dict[str, Any]] | None = None,
    path: str | Path = DEFAULT_MEMORY_PATH,
    top_k: int = 5,
    embedding_backend: EmbeddingBackend | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Retrieve episodes with BM25 + embedding hybrid scoring.

    Each result includes a ``retrieval`` object containing the final score and
    all score components for evaluation and debugging. The persisted memory is
    never mutated.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k <= 0:
        return []
    if min_score < 0.0:
        raise ValueError("min_score must be non-negative")

    records = list(episodes if episodes is not None else _read_episodes(path))
    if not records:
        return []

    query_text = _expand_query(question)
    documents = [_index_text(episode) for episode in records]
    keyword_scores = _normalized_bm25_scores(query_text, documents)

    backend = embedding_backend or HashingEmbeddingBackend()
    vectors = backend.encode([query_text, *documents])
    if len(vectors) != len(documents) + 1:
        raise ValueError("embedding backend returned an unexpected number of vectors")
    query_vector = vectors[0]
    embedding_scores = [max(0.0, _cosine_similarity(query_vector, vector)) for vector in vectors[1:]]
    recency_scores = _recency_scores(records)

    ranked: list[tuple[float, dict[str, Any]]] = []
    for index, episode in enumerate(records):
        importance = _unit_score(episode.get("importance", 0.5), "importance")
        overlap_penalty = _overlap_penalty(episode)
        final_score = (
            EMBEDDING_WEIGHT * embedding_scores[index]
            + KEYWORD_WEIGHT * keyword_scores[index]
            + IMPORTANCE_WEIGHT * importance
            + RECENCY_WEIGHT * recency_scores[index]
            - OVERLAP_PENALTY_WEIGHT * overlap_penalty
        )
        result = dict(episode)
        result["retrieval"] = {
            "final_score": round(final_score, 6),
            "embedding_similarity": round(embedding_scores[index], 6),
            "keyword_score": round(keyword_scores[index], 6),
            "importance": round(importance, 6),
            "recency": round(recency_scores[index], 6),
            "overlap_penalty": round(overlap_penalty, 6),
            "embedding_backend": type(backend).__name__,
        }
        if final_score >= min_score:
            ranked.append((final_score, result))

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1]["retrieval"]["keyword_score"],
            item[1].get("importance", 0.0),
            item[1].get("memory_timestamp", ""),
        ),
        reverse=True,
    )
    return [episode for _, episode in ranked[:top_k]]


def _index_text(episode: dict[str, Any]) -> str:
    values: list[str] = []
    for field in INDEX_FIELDS:
        value = episode.get(field, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values)


def _normalized_bm25_scores(query: str, documents: list[str]) -> list[float]:
    tokenized_documents = [_tokenize(document) for document in documents]
    query_tokens = _tokenize(query)
    if not query_tokens or not tokenized_documents:
        return [0.0] * len(documents)

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))
    average_length = sum(len(tokens) for tokens in tokenized_documents) / len(tokenized_documents)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in tokenized_documents:
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * len(tokens) / max(average_length, 1.0)
            )
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum > 0.0 else 0.0 for score in scores]


def _tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("_", " ")
    latin_tokens = re.findall(r"[a-z0-9]+(?:[+.-][a-z0-9]+)*", normalized)
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk_tokens: list[str] = []
    for sequence in cjk_sequences:
        cjk_tokens.extend(sequence)
        cjk_tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return latin_tokens + cjk_tokens


def _embedding_features(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    compact = re.sub(r"\s+", "", normalized)
    features = _tokenize(normalized)
    for size in (2, 3):
        features.extend(compact[index : index + size] for index in range(max(0, len(compact) - size + 1)))
    return features


def _expand_query(question: str) -> str:
    expanded = [question]
    lowered = question.lower()
    for trigger, terms in QUERY_EXPANSIONS.items():
        if trigger in lowered:
            expanded.extend(terms)
    return " ".join(expanded)


def _recency_scores(episodes: list[dict[str, Any]]) -> list[float]:
    timestamps = [_parse_timestamp(episode.get("memory_timestamp")) for episode in episodes]
    valid = [value for value in timestamps if value is not None]
    if len(valid) >= 2:
        oldest = min(valid)
        newest = max(valid)
        span = max((newest - oldest).total_seconds(), 1.0)
        return [
            (value - oldest).total_seconds() / span if value is not None else 0.0
            for value in timestamps
        ]
    if len(episodes) == 1:
        return [1.0]
    denominator = max(len(episodes) - 1, 1)
    return [index / denominator for index in range(len(episodes))]


def _overlap_penalty(episode: dict[str, Any]) -> float:
    overlap_score = _unit_score(episode.get("overlap_score", 0.0), "overlap_score")
    if overlap_score <= 0.6 or episode.get("event_type") == "uncertainty":
        return 0.0
    return min(1.0, (overlap_score - 0.6) / 0.4)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _unit_score(value: Any, name: str) -> float:
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"episode.{name} must be between 0.0 and 1.0")
    return score


def _read_episodes(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        return read_episodes(path)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


__all__ = [
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "SentenceTransformerEmbeddingBackend",
    "retrieve_episodes",
]
