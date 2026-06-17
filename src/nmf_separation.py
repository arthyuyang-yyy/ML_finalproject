"""Dependency-free single-channel NMF speech-separation baseline (numpy).

When two speakers talk over each other, the high-overlap path keeps multiple
transcript candidates instead of one. Separating the mixture into per-speaker
streams *before* candidate generation gives each hypothesis a cleaner signal to
work from. This module provides that separation as a **dependency-free baseline**
implemented from scratch in numpy, consistent with the project's other
from-scratch learning components (e.g. the agglomerative clustering in
``diarization/embedding_cluster``).

The baseline factorizes the magnitude spectrogram ``V ≈ W @ H`` with
multiplicative updates, groups the ``n_components`` bases into ``num_sources``
clusters (k-means), builds a soft Wiener-style mask per source, and reconstructs
each source with the mixture phase. It is deterministic for a fixed ``seed``.

This is a baseline for comparison, not a production separator: single-channel
NMF separates sources by their differing *temporal activations* and cannot fully
un-mix stationary or spectrally identical speech, so downstream consumers must
keep treating high-overlap output as uncertain.
"""

from __future__ import annotations

import numpy as np

DEFAULT_NUM_SOURCES = 2
DEFAULT_N_FFT = 512
DEFAULT_HOP = 128
DEFAULT_N_ITER = 200
EPSILON = 1e-10


class NmfSeparationBackend:
    """Single-channel NMF separation baseline (numpy only).

    Factorizes the magnitude spectrogram ``V ≈ W @ H`` with multiplicative
    updates, groups the ``n_components`` bases into ``num_sources`` clusters, and
    reconstructs each source through a soft Wiener mask applied to the original
    complex spectrogram. Deterministic for a fixed ``seed``.
    """

    def __init__(
        self,
        n_components: int | None = None,
        n_iter: int = DEFAULT_N_ITER,
        n_fft: int = DEFAULT_N_FFT,
        hop: int = DEFAULT_HOP,
        seed: int = 0,
    ) -> None:
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_fft = n_fft
        self.hop = hop
        self.seed = seed

    def separate(
        self, mixture: np.ndarray, sample_rate: int, num_sources: int
    ) -> list[np.ndarray]:
        if num_sources <= 0:
            raise ValueError("num_sources must be positive")
        mono = np.asarray(mixture, dtype=np.float64).reshape(-1)
        # Too short to frame, or a single source requested: nothing to un-mix.
        if num_sources == 1 or mono.size < self.n_fft:
            return [mono.astype(np.float64).copy() for _ in range(num_sources)]

        spectrogram = _stft(mono, self.n_fft, self.hop)
        magnitude = np.abs(spectrogram)
        n_components = max(self.n_components or num_sources, num_sources)
        basis, activation = _nmf(magnitude, n_components, self.n_iter, self.seed)

        groups = _group_components(basis, num_sources)
        model = basis @ activation + EPSILON
        sources: list[np.ndarray] = []
        for group in groups:
            source_magnitude = basis[:, group] @ activation[group, :]
            mask = source_magnitude / model
            source = _istft(mask * spectrogram, self.n_fft, self.hop, mono.size)
            sources.append(source)
        return sources


# --------------------------------------------------------------------------- #
# NMF and STFT helpers (numpy)
# --------------------------------------------------------------------------- #
def _nmf(
    matrix: np.ndarray, n_components: int, n_iter: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize ``matrix ≈ W @ H`` with non-negative multiplicative updates."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = matrix.shape
    basis = rng.random((n_rows, n_components)) + EPSILON
    activation = rng.random((n_components, n_cols)) + EPSILON
    for _ in range(n_iter):
        model = basis @ activation + EPSILON
        activation *= (basis.T @ matrix) / (basis.T @ model + EPSILON)
        model = basis @ activation + EPSILON
        basis *= (matrix @ activation.T) / (model @ activation.T + EPSILON)
    return basis, activation


def _group_components(basis: np.ndarray, num_sources: int) -> list[list[int]]:
    """Cluster NMF basis columns into ``num_sources`` non-empty source groups."""
    n_components = basis.shape[1]
    if num_sources >= n_components:
        return [[index] for index in range(n_components)]

    features = (basis / (np.linalg.norm(basis, axis=0, keepdims=True) + EPSILON)).T
    labels = _kmeans(features, num_sources, seed=0)
    groups = [list(np.flatnonzero(labels == source)) for source in range(num_sources)]
    # Guarantee every source gets at least one basis (round-robin any empties).
    if any(not group for group in groups):
        groups = [[] for _ in range(num_sources)]
        for index in range(n_components):
            groups[index % num_sources].append(index)
    return groups


def _kmeans(points: np.ndarray, k: int, *, seed: int, n_iter: int = 50) -> np.ndarray:
    """Deterministic small k-means returning a cluster label per row."""
    rng = np.random.default_rng(seed)
    n_points = points.shape[0]
    centers = points[rng.permutation(n_points)[:k]].copy()
    labels = np.zeros(n_points, dtype=np.int64)
    for _ in range(n_iter):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = points[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
    return labels


def _stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Center-padded STFT returning a complex ``[freq, frame]`` spectrogram."""
    window = np.hanning(n_fft)
    pad = n_fft // 2
    padded = np.pad(signal, pad, mode="reflect")
    n_frames = 1 + (padded.size - n_fft) // hop
    frames = np.stack([padded[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def _istft(spectrogram: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    """Inverse of :func:`_stft` via windowed overlap-add, cropped to ``length``."""
    window = np.hanning(n_fft)
    frames = np.fft.irfft(spectrogram.T, n=n_fft, axis=1)
    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop + n_fft
    output = np.zeros(out_len, dtype=np.float64)
    weight = np.zeros(out_len, dtype=np.float64)
    for index in range(n_frames):
        start = index * hop
        output[start : start + n_fft] += frames[index] * window
        weight[start : start + n_fft] += window**2
    output /= np.maximum(weight, EPSILON)
    pad = n_fft // 2
    return output[pad : pad + length]


__all__ = [
    "DEFAULT_NUM_SOURCES",
    "NmfSeparationBackend",
]
