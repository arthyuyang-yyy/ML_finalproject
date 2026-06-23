"""Dependency-free multilingual BLAKE2 character n-gram embeddings."""

import hashlib
import math
import re
from collections.abc import Sequence


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


def _embedding_features(text: str) -> list[str]:
    """Extract character n-gram and CJK token features."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    compact = re.sub(r"\s+", "", normalized)
    features = _tokenize(normalized)
    for size in (2, 3):
        features.extend(compact[index : index + size] for index in range(max(0, len(compact) - size + 1)))
    return features


def _tokenize(text: str) -> list[str]:
    """Simple Latin + CJK tokenizer for embedding features."""
    normalized = text.lower().replace("_", " ")
    latin_tokens = re.findall(r"[a-z0-9]+(?:[+.-][a-z0-9]+)*", normalized)
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk_tokens: list[str] = []
    for sequence in cjk_sequences:
        cjk_tokens.extend(sequence)
        cjk_tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return latin_tokens + cjk_tokens


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


__all__ = ["HashingEmbeddingBackend"]
