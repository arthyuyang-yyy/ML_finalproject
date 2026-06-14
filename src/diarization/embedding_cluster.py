"""Speaker-embedding clustering for the low-overlap path.

This module is the project's genuine *unsupervised machine-learning* component.
It reproduces the senior thesis's "voiceprint clustering" idea (extract a
speaker embedding per segment, then cluster segments that belong to the same
speaker). Beyond a hard label, each segment also carries a
``cluster_similarity_distribution`` (a relative, uncalibrated similarity signal)
and an honest confidence based on the top-1/top-2 margin, with explicit UNKNOWN
handling for degenerate inputs — useful for downstream LLM joint decoding and
cross-meeting re-ID.

Design notes
------------
* Dependency-free by default. Embeddings come from a numpy log-mel acoustic
  feature extractor (``AcousticEmbeddingBackend``); the agglomerative
  hierarchical clustering (AHC) is implemented here from scratch rather than
  pulled from scikit-learn, so the learning component is the team's own code.
* A learned d-vector backend (``ResemblyzerEmbeddingBackend``) is available as
  an optional, lazily-imported upgrade when the heavier dependency is present.
* Speaker count is inferred automatically via a cosine distance threshold
  (standard for diarization), or fixed when ``num_speakers`` is known.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from ..audio.preprocess import TARGET_SAMPLE_RATE

# Clusters are merged while the closest pair sits within this cosine distance.
# Calibrated for the acoustic backend, where same-speaker pairs sit near 0 and
# different-speaker pairs sit around 0.4; learned d-vectors may want retuning.
DEFAULT_DISTANCE_THRESHOLD = 0.30
# Softmax temperature converting centroid similarities into a relative
# cluster-similarity distribution (NOT a calibrated speaker posterior).
DEFAULT_TEMPERATURE = 0.1
# Confidence is the *discrimination margin* between the top-1 and top-2 cluster
# similarities. Below this margin a segment cannot be confidently attributed and
# is labelled UNKNOWN, honouring Step 6's "no clear match" rule.
DEFAULT_MIN_CONFIDENCE = 0.3
# Embeddings with an L2 norm at or below this carry no usable voiceprint (e.g. a
# zero vector from an empty/silent clip, or a missing precomputed embedding) and
# are never given a confident speaker label.
MIN_VOICEPRINT_NORM = 1e-6
# Raw waveform clips with an RMS at or below this are treated as silence by the
# acoustic backend, which then emits a zero embedding so the guard above fires.
MIN_VOICEPRINT_RMS = 1e-4
DEFAULT_SPEAKER_PREFIX = "SPEAKER_"
UNKNOWN_SPEAKER = "UNKNOWN"
EPSILON = 1e-10


class SpeakerEmbeddingBackend(Protocol):
    """Maps each segment to a fixed-length speaker embedding vector."""

    def encode_segments(
        self,
        samples: np.ndarray,
        segments: Sequence[dict[str, Any]],
        sample_rate: int,
    ) -> np.ndarray:
        """Return an ``[N, D]`` array of one embedding per input segment."""
        ...


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #
class AcousticEmbeddingBackend:
    """Dependency-free log-mel speaker embedding (numpy only).

    Each segment is framed, transformed to a log-mel spectrogram, and pooled
    over time with mean and standard deviation. The pooled vector captures the
    spectral envelope (a proxy for vocal-tract characteristics), which makes
    same-speaker segments cluster together. It is deterministic and needs no
    model download, so it powers the unit tests and the lightweight demo path.
    """

    def __init__(
        self,
        n_mels: int = 40,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        fmin: float = 20.0,
        fmax: float | None = None,
        min_rms: float = MIN_VOICEPRINT_RMS,
    ) -> None:
        if n_mels <= 0:
            raise ValueError("n_mels must be positive")
        self.n_mels = n_mels
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.fmin = fmin
        self.fmax = fmax
        self.min_rms = min_rms

    def encode_segments(
        self,
        samples: np.ndarray,
        segments: Sequence[dict[str, Any]],
        sample_rate: int,
    ) -> np.ndarray:
        vectors = [
            self._encode_one(_slice_segment(samples, segment, sample_rate), sample_rate)
            for segment in segments
        ]
        return np.asarray(vectors, dtype=np.float64)

    def _encode_one(self, clip: np.ndarray, sample_rate: int) -> np.ndarray:
        dimension = 2 * self.n_mels
        # Silence has a constant log-mel that would otherwise normalize to a unit
        # vector and masquerade as a real voiceprint. Emit a zero embedding so the
        # downstream norm guard labels it UNKNOWN instead.
        if clip.size == 0 or float(np.sqrt(np.mean(np.square(clip)))) <= self.min_rms:
            return np.zeros(dimension, dtype=np.float64)
        log_mel = self._log_mel(clip, sample_rate)
        if log_mel.size == 0:
            return np.zeros(dimension, dtype=np.float64)
        pooled = np.concatenate([log_mel.mean(axis=0), log_mel.std(axis=0)])
        return _l2_normalize(pooled)

    def _log_mel(self, clip: np.ndarray, sample_rate: int) -> np.ndarray:
        frame_len = max(1, int(sample_rate * self.frame_ms / 1000.0))
        hop = max(1, int(sample_rate * self.hop_ms / 1000.0))
        frames = _frame_signal(clip, frame_len, hop)
        if frames.shape[0] == 0:
            return np.empty((0, self.n_mels), dtype=np.float64)
        n_fft = _next_power_of_two(frame_len)
        window = np.hamming(frame_len)
        power = np.abs(np.fft.rfft(frames * window, n=n_fft)) ** 2
        filterbank = _mel_filterbank(self.n_mels, n_fft, sample_rate, self.fmin, self.fmax)
        mel_energy = power @ filterbank.T
        return np.log(mel_energy + EPSILON)


class ResemblyzerEmbeddingBackend:
    """Learned d-vector speaker embeddings via the optional ``resemblyzer`` package.

    Lazily imported so the dependency stays optional. Produces 256-dim d-vectors
    from a model pretrained on speaker verification — a drop-in, higher-quality
    replacement for the acoustic baseline when the package is installed.
    """

    def __init__(self) -> None:
        self._encoder: Any | None = None

    def _load(self) -> Any:
        if self._encoder is None:
            from resemblyzer import VoiceEncoder  # type: ignore[import-not-found]

            self._encoder = VoiceEncoder()
        return self._encoder

    def encode_segments(
        self,
        samples: np.ndarray,
        segments: Sequence[dict[str, Any]],
        sample_rate: int,
    ) -> np.ndarray:
        from resemblyzer import preprocess_wav  # type: ignore[import-not-found]

        encoder = self._load()
        vectors = []
        for segment in segments:
            clip = _slice_segment(samples, segment, sample_rate)
            if clip.size == 0:
                vectors.append(np.zeros(256, dtype=np.float64))
                continue
            wav = preprocess_wav(clip.astype(np.float32), source_sr=sample_rate)
            vectors.append(_l2_normalize(np.asarray(encoder.embed_utterance(wav), dtype=np.float64)))
        return np.asarray(vectors, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Agglomerative hierarchical clustering (implemented here, not scikit-learn)
# --------------------------------------------------------------------------- #
def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine distance matrix for L2-normalized rows."""
    normalized = _l2_normalize_rows(embeddings)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return 1.0 - similarity


def agglomerative_cluster(
    embeddings: np.ndarray,
    *,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    num_speakers: int | None = None,
    max_speakers: int = 10,
) -> np.ndarray:
    """Cluster embeddings with average-linkage AHC over cosine distance.

    Merging stops when either the target ``num_speakers`` is reached, or the
    closest remaining cluster pair exceeds ``distance_threshold``. Returns an
    integer label per row, relabelled to a contiguous ``0..K-1`` range ordered
    by first appearance.
    """
    n = len(embeddings)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    distances = cosine_distance_matrix(embeddings)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    target = _clamp_target(num_speakers, n, max_speakers)

    while len(members) > 1:
        left, right, linkage = _closest_pair(members, distances)
        if target is not None:
            if len(members) <= target:
                break
        # Auto mode: the threshold stops merging, but ``max_speakers`` is a hard
        # cap that keeps merging the closest pair even past the threshold.
        elif linkage > distance_threshold and len(members) <= max_speakers:
            break
        members[left].extend(members[right])
        del members[right]

    return _relabel(members, n)


def _closest_pair(
    members: dict[int, list[int]],
    distances: np.ndarray,
) -> tuple[int, int, float]:
    """Return the two cluster ids with the smallest average-linkage distance."""
    keys = list(members)
    best: tuple[int, int, float] = (keys[0], keys[1], float("inf"))
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            linkage = float(distances[np.ix_(members[left], members[right])].mean())
            if linkage < best[2]:
                best = (left, right, linkage)
    return best


def _relabel(members: dict[int, list[int]], n: int) -> np.ndarray:
    """Map cluster members to contiguous labels ordered by first appearance."""
    labels = np.empty(n, dtype=np.int64)
    order = sorted(members.values(), key=min)
    for new_label, indices in enumerate(order):
        for index in indices:
            labels[index] = new_label
    return labels


def _clamp_target(num_speakers: int | None, n: int, max_speakers: int) -> int | None:
    if num_speakers is None:
        return None
    if num_speakers <= 0:
        raise ValueError("num_speakers must be positive")
    return max(1, min(num_speakers, n, max_speakers))


# --------------------------------------------------------------------------- #
# Cluster-similarity distribution (a relative signal, not a calibrated posterior)
# --------------------------------------------------------------------------- #
def cluster_similarity_distributions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each segment's relative similarity to every cluster centroid.

    Each cluster centroid is the mean of its (normalized) member embeddings.
    Segment-to-centroid cosine similarities are scaled by ``temperature`` and
    softmaxed, yielding an ``[N, K]`` distribution matrix plus the ``[K, D]``
    centroids.

    This is deliberately **not** called a speaker posterior: segments take part
    in their own centroid, so the values are optimistic, a single cluster
    trivially yields 1.0, and the softmax sharpness is governed by
    ``temperature``. It is a relative, display-only signal. Confidence and
    UNKNOWN decisions in :func:`cluster_segments` instead use the raw,
    temperature-free cosine-similarity margin from :func:`_centroid_similarities`
    plus explicit degeneracy checks.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(embeddings) == 0:
        return np.empty((0, 0)), np.empty((0, 0))

    similarity, centroids = _centroid_similarities(embeddings, labels)
    distributions = _softmax(similarity / temperature, axis=1)
    return distributions, centroids


def _centroid_similarities(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``[N, K]`` cosine similarities to each centroid and the centroids.

    These cosine similarities are temperature-free, so any confidence derived
    from them is a property of the embeddings rather than of a softmax knob.
    """
    normalized = _l2_normalize_rows(embeddings)
    num_clusters = int(labels.max()) + 1
    centroids = np.stack(
        [_l2_normalize(normalized[labels == k].mean(axis=0)) for k in range(num_clusters)]
    )
    similarity = np.clip(normalized @ centroids.T, -1.0, 1.0)
    return similarity, centroids


# --------------------------------------------------------------------------- #
# Public orchestration
# --------------------------------------------------------------------------- #
def cluster_segments(
    samples: np.ndarray,
    segments: Sequence[dict[str, Any]],
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    backend: SpeakerEmbeddingBackend | None = None,
    num_speakers: int | None = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    speaker_prefix: str = DEFAULT_SPEAKER_PREFIX,
    attach_embeddings: bool = False,
) -> list[dict[str, Any]]:
    """Assign speakers to segments by embedding + clustering, honestly.

    Returns each input segment enriched with:

    * ``speaker`` — the assigned label, e.g. ``"SPEAKER_00"``, or ``"UNKNOWN"``.
    * ``speaker_confidence`` — the **discrimination margin** between the top-1
      and top-2 cluster similarities, in ``[0, 1]``. It is ``0.0`` whenever the
      attribution carries no real evidence (see below), so degenerate inputs are
      never dressed up as certain.
    * ``cluster_similarity_distribution`` — the relative, uncalibrated
      ``{label: value}`` similarity distribution, kept for inspection.
    * ``embedding`` — the raw vector (only when ``attach_embeddings`` is set;
      useful for cross-meeting global speaker re-identification / memory).

    Voiceprint-free segments (silence / zero vectors) are excluded from the
    clustering itself, so they never occupy a speaker slot or split real speakers
    apart under a fixed ``num_speakers``; they are backfilled afterwards as
    ``UNKNOWN`` with an empty distribution.

    A segment is labelled ``UNKNOWN`` (and given confidence ``0.0``) when:

    * its embedding has no usable voiceprint (e.g. silence / a zero vector); or
    * clustering produced a single group, so no speaker discrimination was
      actually performed (this includes the single-segment case); or
    * the top-1/top-2 similarity margin is below ``min_confidence``.

    Segments that already carry a precomputed ``"embedding"`` are reused as-is,
    so embeddings stored across meetings can be re-clustered without audio.
    """
    if not segments:
        return []

    embeddings = _resolve_embeddings(samples, segments, sample_rate, backend)
    norms = np.linalg.norm(embeddings, axis=1)

    # Voiceprint-free segments (silence / zero vectors) must not take part in
    # clustering: a zero vector would otherwise occupy a speaker slot and, under a
    # fixed ``num_speakers``, force real speakers to be merged. Cluster only the
    # valid embeddings, then backfill the invalid ones as UNKNOWN with an empty
    # distribution.
    valid_index = np.flatnonzero(norms > MIN_VOICEPRINT_NORM)
    valid_position = {int(global_idx): pos for pos, global_idx in enumerate(valid_index)}

    num_clusters = 0
    valid_similarities = np.empty((0, 0))
    valid_distributions = np.empty((0, 0))
    if valid_index.size:
        valid_embeddings = embeddings[valid_index]
        valid_labels = agglomerative_cluster(
            valid_embeddings, distance_threshold=distance_threshold, num_speakers=num_speakers
        )
        valid_similarities, _ = _centroid_similarities(valid_embeddings, valid_labels)
        valid_distributions, _ = cluster_similarity_distributions(
            valid_embeddings, valid_labels, temperature=temperature
        )
        num_clusters = valid_similarities.shape[1] if valid_similarities.size else 0
    discriminative = num_clusters >= 2

    enriched: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        pos = valid_position.get(index)
        if pos is None:
            speaker, confidence, distribution_map = UNKNOWN_SPEAKER, 0.0, {}
        else:
            distribution_map = {
                f"{speaker_prefix}{k:02d}": round(float(valid_distributions[pos][k]), 4)
                for k in range(num_clusters)
            }
            speaker, confidence = _attribute_speaker(
                valid_similarities[pos],
                float(norms[index]),
                discriminative,
                min_confidence,
                speaker_prefix,
            )
        attributed = {
            **segment,
            "speaker": speaker,
            "speaker_confidence": round(confidence, 3),
            "cluster_similarity_distribution": distribution_map,
        }
        if attach_embeddings:
            attributed["embedding"] = embeddings[index].tolist()
        enriched.append(attributed)
    return enriched


def _attribute_speaker(
    similarities: np.ndarray,
    norm: float,
    discriminative: bool,
    min_confidence: float,
    speaker_prefix: str,
) -> tuple[str, float]:
    """Pick a speaker label and an honest confidence for one segment.

    Confidence is the top-1/top-2 margin of the **raw cosine similarities** to
    the cluster centroids — a temperature-free property of the embeddings, not
    of the softmax distribution. It is forced to ``0.0`` for inputs with no
    discriminative evidence (silence / no voiceprint, or a single cluster), so a
    degenerate value is never sold as certainty.
    """
    if norm <= MIN_VOICEPRINT_NORM or not discriminative:
        return UNKNOWN_SPEAKER, 0.0
    ordered = np.sort(similarities)[::-1]
    margin = float(np.clip(ordered[0] - ordered[1], 0.0, 1.0))
    if margin < min_confidence:
        return UNKNOWN_SPEAKER, margin
    winner = int(np.argmax(similarities))
    return f"{speaker_prefix}{winner:02d}", margin


def _resolve_embeddings(
    samples: np.ndarray,
    segments: Sequence[dict[str, Any]],
    sample_rate: int,
    backend: SpeakerEmbeddingBackend | None,
) -> np.ndarray:
    """Use precomputed segment embeddings when present, else encode audio."""
    precomputed = [segment.get("embedding") for segment in segments]
    if all(vector is not None for vector in precomputed):
        return _l2_normalize_rows(np.asarray(precomputed, dtype=np.float64))
    backend = backend or AcousticEmbeddingBackend()
    return backend.encode_segments(samples, segments, sample_rate)


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _slice_segment(samples: np.ndarray, segment: dict[str, Any], sample_rate: int) -> np.ndarray:
    """Return the waveform slice for a segment's time span."""
    if samples is None or len(samples) == 0:
        return np.zeros(0, dtype=np.float64)
    start = int(round(float(segment.get("start_time", 0.0)) * sample_rate))
    end = int(round(float(segment.get("end_time", 0.0)) * sample_rate))
    start = max(0, min(start, len(samples)))
    end = max(start, min(end, len(samples)))
    return np.asarray(samples[start:end], dtype=np.float64)


def _frame_signal(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split a 1-D signal into overlapping frames ``[num_frames, frame_len]``."""
    if signal.size < frame_len:
        if signal.size == 0:
            return np.empty((0, frame_len), dtype=np.float64)
        padded = np.zeros(frame_len, dtype=np.float64)
        padded[: signal.size] = signal
        return padded[np.newaxis, :]
    num_frames = 1 + (signal.size - frame_len) // hop
    indices = np.arange(frame_len)[np.newaxis, :] + hop * np.arange(num_frames)[:, np.newaxis]
    return signal[indices]


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float | None,
) -> np.ndarray:
    """Build a triangular mel filterbank ``[n_mels, n_fft // 2 + 1]``."""
    fmax = fmax if fmax is not None else sample_rate / 2.0
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            if center > left:
                filterbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                filterbank[m - 1, k] = (right - k) / (right - center)
    return filterbank


def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else int(2 ** np.ceil(np.log2(value)))


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > EPSILON else vector


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > EPSILON)


def _softmax(scores: np.ndarray, axis: int) -> np.ndarray:
    shifted = scores - scores.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


__all__ = [
    "DEFAULT_DISTANCE_THRESHOLD",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_TEMPERATURE",
    "MIN_VOICEPRINT_NORM",
    "MIN_VOICEPRINT_RMS",
    "UNKNOWN_SPEAKER",
    "AcousticEmbeddingBackend",
    "ResemblyzerEmbeddingBackend",
    "SpeakerEmbeddingBackend",
    "agglomerative_cluster",
    "cluster_segments",
    "cluster_similarity_distributions",
    "cosine_distance_matrix",
]
