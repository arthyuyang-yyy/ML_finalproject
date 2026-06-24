"""Tests for speaker-embedding clustering (the unsupervised ML component).

Synthetic "speakers" are built as harmonic tones at distinct fundamental
frequencies, which the log-mel acoustic backend separates into spectral
envelopes. This keeps the tests dependency-free (numpy only) and GPU-free.
Confidence is the top-1/top-2 cluster-similarity margin, and degenerate inputs
(silence, a single cluster, a single segment) are deliberately UNKNOWN.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.diarization import diarize_audio
from src.diarization.embedding_cluster import (
    AcousticEmbeddingBackend,
    agglomerative_cluster,
    cluster_segments,
    cluster_similarity_distributions,
    cosine_distance_matrix,
)

SAMPLE_RATE = 16000


def _voice(f0: float, duration: float, sample_rate: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """A simple harmonic 'voice': fundamental plus decaying harmonics and noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(times)
    for harmonic in range(1, 6):
        signal += (1.0 / harmonic) * np.sin(2.0 * np.pi * f0 * harmonic * times)
    signal += 0.01 * rng.standard_normal(times.shape)
    peak = np.max(np.abs(signal))
    return (signal / peak).astype(np.float32) if peak else signal.astype(np.float32)


def _two_speaker_meeting(
    pattern: str = "ABABAB",
    seg_duration: float = 1.0,
    f0_a: float = 110.0,
    f0_b: float = 220.0,
):
    """Build a waveform and segment list for an alternating two-speaker meeting."""
    samples: list[np.ndarray] = []
    segments: list[dict] = []
    cursor = 0.0
    for index, who in enumerate(pattern):
        f0 = f0_a if who == "A" else f0_b
        clip = _voice(f0, seg_duration, seed=index)
        samples.append(clip)
        segments.append(
            {
                "segment_id": f"s{index}",
                "start_time": round(cursor, 3),
                "end_time": round(cursor + seg_duration, 3),
                "truth": who,
            }
        )
        cursor += seg_duration
    return np.concatenate(samples), segments


def _ambiguous_meeting() -> list[dict]:
    """Two confident speakers plus one segment sitting midway between them."""
    speaker_a = [[1.0, 0.0, 0.0]] * 3
    speaker_b = [[0.0, 1.0, 0.0]] * 3
    ambiguous = [0.7071, 0.7071, 0.0]
    embeddings = speaker_a + speaker_b + [ambiguous]
    return [
        {"segment_id": f"s{i}", "start_time": float(i), "end_time": float(i + 1), "embedding": vec}
        for i, vec in enumerate(embeddings)
    ]


class AcousticEmbeddingBackendTests(unittest.TestCase):
    def test_same_speaker_closer_than_different_speaker(self) -> None:
        samples, segments = _two_speaker_meeting("ABA", seg_duration=1.0)
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        distances = cosine_distance_matrix(embeddings)
        self.assertLess(distances[0, 2], distances[0, 1])
        self.assertLess(distances[0, 2], distances[1, 2])

    def test_embedding_dimension_is_two_times_n_mels(self) -> None:
        samples, segments = _two_speaker_meeting("A", seg_duration=1.0)
        embeddings = AcousticEmbeddingBackend(n_mels=32).encode_segments(samples, segments, SAMPLE_RATE)
        self.assertEqual(embeddings.shape, (1, 64))

    def test_silent_clip_yields_zero_vector(self) -> None:
        # Silence must NOT be normalized into a unit vector that masquerades as a
        # real voiceprint; it has to come back as a zero embedding.
        segments = [{"segment_id": "s0", "start_time": 0.0, "end_time": 1.0}]
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        embeddings = AcousticEmbeddingBackend().encode_segments(silence, segments, SAMPLE_RATE)
        self.assertTrue(np.isfinite(embeddings).all())
        self.assertAlmostEqual(float(np.linalg.norm(embeddings[0])), 0.0)


class AgglomerativeClusterTests(unittest.TestCase):
    def test_separates_two_speakers_automatically(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        self.assertEqual(len(set(labels)), 2)
        a_labels = {labels[i] for i, seg in enumerate(segments) if seg["truth"] == "A"}
        b_labels = {labels[i] for i, seg in enumerate(segments) if seg["truth"] == "B"}
        self.assertEqual(len(a_labels), 1)
        self.assertEqual(len(b_labels), 1)
        self.assertFalse(a_labels & b_labels)

    def test_num_speakers_forces_cluster_count(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings, num_speakers=3)
        self.assertEqual(len(set(labels)), 3)

    def test_max_speakers_caps_count_in_auto_mode(self) -> None:
        # Twelve mutually orthogonal embeddings would otherwise stay 12 clusters;
        # max_speakers is a hard cap even without an explicit num_speakers.
        labels = agglomerative_cluster(np.eye(12), max_speakers=3)
        self.assertEqual(len(set(labels)), 3)

    def test_max_speakers_does_not_over_merge_separated_speakers(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings, max_speakers=10)
        self.assertEqual(len(set(labels)), 2)

    def test_single_segment_returns_single_cluster(self) -> None:
        self.assertEqual(list(agglomerative_cluster(np.array([[1.0, 0.0, 0.0]]))), [0])

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(len(agglomerative_cluster(np.empty((0, 4)))), 0)

    def test_high_threshold_collapses_to_one_cluster(self) -> None:
        samples, segments = _two_speaker_meeting("ABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings, distance_threshold=2.0)
        self.assertEqual(len(set(labels)), 1)

    def test_labels_are_contiguous_from_zero(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        self.assertEqual(sorted(set(labels)), list(range(len(set(labels)))))


class SimilarityDistributionTests(unittest.TestCase):
    def test_distributions_are_valid_and_sum_to_one(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        distributions, centroids = cluster_similarity_distributions(embeddings, labels)
        self.assertEqual(distributions.shape, (len(segments), len(set(labels))))
        np.testing.assert_allclose(distributions.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue((distributions >= 0).all())
        self.assertEqual(centroids.shape[0], len(set(labels)))

    def test_argmax_matches_hard_labels(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        distributions, _ = cluster_similarity_distributions(embeddings, labels)
        np.testing.assert_array_equal(distributions.argmax(axis=1), labels)


class ClusterSegmentsTests(unittest.TestCase):
    def test_enriches_segments_with_speaker_fields(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        self.assertEqual(len(result), len(segments))
        for item in result:
            self.assertIn("cluster_similarity_distribution", item)
            self.assertAlmostEqual(sum(item["cluster_similarity_distribution"].values()), 1.0, places=3)
            self.assertGreaterEqual(item["speaker_confidence"], 0.0)
            self.assertLessEqual(item["speaker_confidence"], 1.0)

    def test_confidence_is_temperature_invariant(self) -> None:
        # Confidence comes from the raw cosine margin, not the softmax, so the
        # temperature knob must not move labels or confidence values.
        embeddings = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]]
        segments = [
            {"segment_id": f"s{i}", "start_time": float(i), "end_time": float(i + 1), "embedding": vec}
            for i, vec in enumerate(embeddings)
        ]
        baseline = cluster_segments(np.zeros(0, dtype=np.float32), segments, num_speakers=2, temperature=0.1)
        for temperature in (1.0, 5.0):
            other = cluster_segments(
                np.zeros(0, dtype=np.float32), segments, num_speakers=2, temperature=temperature
            )
            self.assertEqual(
                [s["speaker"] for s in other], [s["speaker"] for s in baseline]
            )
            self.assertEqual(
                [s["speaker_confidence"] for s in other],
                [s["speaker_confidence"] for s in baseline],
            )

    def test_preserves_original_segment_fields(self) -> None:
        samples, segments = _two_speaker_meeting("AB")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        self.assertEqual(result[0]["segment_id"], "s0")
        self.assertEqual(result[0]["start_time"], segments[0]["start_time"])

    def test_two_speakers_get_two_distinct_labels(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        speakers_by_truth = {"A": set(), "B": set()}
        for item, seg in zip(result, segments):
            speakers_by_truth[seg["truth"]].add(item["speaker"])
        self.assertEqual(len(speakers_by_truth["A"]), 1)
        self.assertEqual(len(speakers_by_truth["B"]), 1)
        self.assertFalse(speakers_by_truth["A"] & speakers_by_truth["B"])
        self.assertNotIn("UNKNOWN", speakers_by_truth["A"] | speakers_by_truth["B"])

    def test_attach_embeddings_includes_vector(self) -> None:
        samples, segments = _two_speaker_meeting("AB")
        result = cluster_segments(samples, segments, SAMPLE_RATE, attach_embeddings=True)
        self.assertIsInstance(result[0]["embedding"], list)
        self.assertGreater(len(result[0]["embedding"]), 0)

    def test_empty_segments_returns_empty(self) -> None:
        self.assertEqual(cluster_segments(np.zeros(10, dtype=np.float32), []), [])

    def test_precomputed_embeddings_are_reused_without_audio(self) -> None:
        # Cross-meeting re-ID path: cluster stored embeddings with no waveform.
        segments = [
            {"segment_id": "m1", "start_time": 0.0, "end_time": 1.0, "embedding": [1.0, 0.0, 0.0]},
            {"segment_id": "m2", "start_time": 1.0, "end_time": 2.0, "embedding": [0.97, 0.05, 0.0]},
            {"segment_id": "m3", "start_time": 2.0, "end_time": 3.0, "embedding": [0.0, 0.0, 1.0]},
        ]
        result = cluster_segments(np.zeros(0, dtype=np.float32), segments)
        self.assertEqual(result[0]["speaker"], result[1]["speaker"])
        self.assertNotEqual(result[0]["speaker"], result[2]["speaker"])


class DegeneracyTests(unittest.TestCase):
    """Inputs without discriminative evidence must never look confident."""

    def test_single_cluster_is_unknown_not_confident(self) -> None:
        # One speaker repeated -> a single cluster -> no discrimination performed.
        samples, segments = _two_speaker_meeting("AAAA")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        for item in result:
            self.assertEqual(item["speaker"], "UNKNOWN")
            self.assertEqual(item["speaker_confidence"], 0.0)

    def test_single_segment_is_unknown(self) -> None:
        samples, segments = _two_speaker_meeting("A")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        self.assertEqual(result[0]["speaker"], "UNKNOWN")
        self.assertEqual(result[0]["speaker_confidence"], 0.0)

    def test_silence_is_unknown(self) -> None:
        segments = [
            {"segment_id": "s0", "start_time": 0.0, "end_time": 1.0},
            {"segment_id": "s1", "start_time": 1.0, "end_time": 2.0},
        ]
        silence = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
        result = cluster_segments(silence, segments, SAMPLE_RATE)
        for item in result:
            self.assertEqual(item["speaker"], "UNKNOWN")
            self.assertEqual(item["speaker_confidence"], 0.0)

    def test_silence_among_real_speakers_is_unknown(self) -> None:
        # The reviewer's scenario: silence + speaker A + speaker B in one meeting.
        # Real speakers stay labelled; silence must not borrow a confident label.
        silence = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
        samples = np.concatenate([silence, _voice(110.0, 1.0, seed=1), _voice(240.0, 1.0, seed=2)])
        segments = [
            {"segment_id": "sil", "start_time": 0.0, "end_time": 1.0},
            {"segment_id": "a", "start_time": 1.0, "end_time": 2.0},
            {"segment_id": "b", "start_time": 2.0, "end_time": 3.0},
        ]
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        by_id = {item["segment_id"]: item for item in result}
        self.assertEqual(by_id["sil"]["speaker"], "UNKNOWN")
        self.assertEqual(by_id["sil"]["speaker_confidence"], 0.0)
        self.assertTrue(by_id["a"]["speaker"].startswith("SPEAKER_"))
        self.assertTrue(by_id["b"]["speaker"].startswith("SPEAKER_"))
        self.assertNotEqual(by_id["a"]["speaker"], by_id["b"]["speaker"])

    def test_silence_does_not_consume_a_speaker_slot(self) -> None:
        # The reviewer's follow-up: with num_speakers=2 fixed, the silence zero
        # vector must not occupy a cluster and force A and B to merge.
        silence = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
        samples = np.concatenate([silence, _voice(110.0, 1.0, seed=1), _voice(240.0, 1.0, seed=2)])
        segments = [
            {"segment_id": "sil", "start_time": 0.0, "end_time": 1.0},
            {"segment_id": "a", "start_time": 1.0, "end_time": 2.0},
            {"segment_id": "b", "start_time": 2.0, "end_time": 3.0},
        ]
        result = cluster_segments(samples, segments, SAMPLE_RATE, num_speakers=2)
        by_id = {item["segment_id"]: item for item in result}
        self.assertEqual(by_id["sil"]["speaker"], "UNKNOWN")
        # Silence carries no phantom SPEAKER_xx in its distribution.
        self.assertEqual(by_id["sil"]["cluster_similarity_distribution"], {})
        # A and B keep distinct labels instead of being merged into one.
        self.assertNotEqual(by_id["a"]["speaker"], by_id["b"]["speaker"])
        self.assertNotIn("UNKNOWN", {by_id["a"]["speaker"], by_id["b"]["speaker"]})
        # The phantom SPEAKER_00 from silence never appears anywhere.
        all_labels = {item["speaker"] for item in result if item["speaker"] != "UNKNOWN"}
        self.assertEqual(len(all_labels), 2)

    def test_zero_vector_embedding_is_unknown(self) -> None:
        segments = [
            {"segment_id": "s0", "start_time": 0.0, "end_time": 1.0, "embedding": [0.0, 0.0, 0.0]},
            {"segment_id": "s1", "start_time": 1.0, "end_time": 2.0, "embedding": [1.0, 0.0, 0.0]},
            {"segment_id": "s2", "start_time": 2.0, "end_time": 3.0, "embedding": [0.0, 1.0, 0.0]},
        ]
        result = cluster_segments(np.zeros(0, dtype=np.float32), segments)
        self.assertEqual(result[0]["speaker"], "UNKNOWN")
        self.assertEqual(result[0]["speaker_confidence"], 0.0)


class UnknownMarginTests(unittest.TestCase):
    """Step 6 'no clear match -> UNKNOWN' via the discrimination margin."""

    def test_ambiguous_segment_becomes_unknown(self) -> None:
        result = cluster_segments(
            np.zeros(0, dtype=np.float32),
            _ambiguous_meeting(),
            num_speakers=2,
            temperature=1.0,
        )
        for item in result[:6]:
            self.assertTrue(item["speaker"].startswith("SPEAKER_"))
        self.assertEqual(result[-1]["speaker"], "UNKNOWN")
        self.assertLess(result[-1]["speaker_confidence"], 0.3)

    def test_confident_segments_are_never_unknown(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        self.assertTrue(all(item["speaker"] != "UNKNOWN" for item in result))

    def test_min_confidence_zero_disables_margin_unknown(self) -> None:
        # With no margin floor every discriminative segment keeps a label,
        # though degenerate single clusters would still be UNKNOWN.
        result = cluster_segments(
            np.zeros(0, dtype=np.float32),
            _ambiguous_meeting(),
            num_speakers=2,
            temperature=1.0,
            min_confidence=0.0,
        )
        self.assertTrue(all(item["speaker"] != "UNKNOWN" for item in result))


class DiarizeAudioIntegrationTests(unittest.TestCase):
    """Step 5 public interface must run the real clustering, not a placeholder."""

    def test_diarize_audio_runs_clustering_fallback(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        # Two distinct voices separated by silence so VAD yields two segments.
        gap = np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32)
        waveform = np.concatenate([_voice(110.0, 1.2, seed=1), gap, _voice(240.0, 1.2, seed=2)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two_speakers.wav"
            sf.write(path, waveform, SAMPLE_RATE)
            # silero VAD is a learned speech detector and does not reliably fire
            # on synthetic harmonic tones, so stub two segments aligned with the
            # two voices; the test verifies the clustering fallback, not the VAD.
            vad_segments = [
                {"meeting_id": "meeting", "segment_id": "meeting_seg_001", "start_time": 0.0, "end_time": 1.2},
                {"meeting_id": "meeting", "segment_id": "meeting_seg_002", "start_time": 2.0, "end_time": 3.2},
            ]
            with patch("src.audio.preprocess.segment_waveform", return_value=vad_segments):
                result = diarize_audio(str(path))

        self.assertTrue(result)
        # The new clustering path attaches this field; the old placeholder did not.
        for turn in result:
            self.assertIn("cluster_similarity_distribution", turn)
            self.assertIn("speaker", turn)


if __name__ == "__main__":
    unittest.main()
