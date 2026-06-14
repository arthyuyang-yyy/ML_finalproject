"""Tests for speaker-embedding clustering (the unsupervised ML component).

Synthetic "speakers" are built as harmonic tones at distinct fundamental
frequencies, which the log-mel acoustic backend separates into spectral
envelopes. This keeps the tests dependency-free (numpy only) and GPU-free.
"""

import unittest

import numpy as np

from src.diarization.embedding_cluster import (
    AcousticEmbeddingBackend,
    agglomerative_cluster,
    cluster_segments,
    cosine_distance_matrix,
    soft_speaker_posteriors,
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


class AcousticEmbeddingBackendTests(unittest.TestCase):
    def test_same_speaker_closer_than_different_speaker(self) -> None:
        samples, segments = _two_speaker_meeting("ABA", seg_duration=1.0)
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        distances = cosine_distance_matrix(embeddings)
        # segments 0 and 2 are speaker A; segment 1 is speaker B.
        self.assertLess(distances[0, 2], distances[0, 1])
        self.assertLess(distances[0, 2], distances[1, 2])

    def test_embedding_dimension_is_two_times_n_mels(self) -> None:
        samples, segments = _two_speaker_meeting("A", seg_duration=1.0)
        backend = AcousticEmbeddingBackend(n_mels=32)
        embeddings = backend.encode_segments(samples, segments, SAMPLE_RATE)
        self.assertEqual(embeddings.shape, (1, 64))

    def test_silent_clip_yields_finite_zero_vector(self) -> None:
        segments = [{"segment_id": "s0", "start_time": 0.0, "end_time": 1.0}]
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        embeddings = AcousticEmbeddingBackend().encode_segments(silence, segments, SAMPLE_RATE)
        self.assertTrue(np.isfinite(embeddings).all())


class AgglomerativeClusterTests(unittest.TestCase):
    def test_separates_two_speakers_automatically(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        self.assertEqual(len(set(labels)), 2)
        # Every A segment shares a label distinct from every B segment.
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

    def test_single_segment_returns_single_cluster(self) -> None:
        labels = agglomerative_cluster(np.array([[1.0, 0.0, 0.0]]))
        self.assertEqual(list(labels), [0])

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


class SoftPosteriorTests(unittest.TestCase):
    def test_posteriors_are_valid_distributions(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        posteriors, centroids = soft_speaker_posteriors(embeddings, labels)
        self.assertEqual(posteriors.shape, (len(segments), len(set(labels))))
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue((posteriors >= 0).all())
        self.assertEqual(centroids.shape[0], len(set(labels)))

    def test_confident_assignment_for_well_separated_speakers(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        posteriors, _ = soft_speaker_posteriors(embeddings, labels)
        self.assertTrue((posteriors.max(axis=1) > 0.6).all())

    def test_argmax_matches_hard_labels(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        embeddings = AcousticEmbeddingBackend().encode_segments(samples, segments, SAMPLE_RATE)
        labels = agglomerative_cluster(embeddings)
        posteriors, _ = soft_speaker_posteriors(embeddings, labels)
        np.testing.assert_array_equal(posteriors.argmax(axis=1), labels)


class ClusterSegmentsTests(unittest.TestCase):
    def test_enriches_segments_with_speaker_fields(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        self.assertEqual(len(result), len(segments))
        for item in result:
            self.assertTrue(item["speaker"].startswith("SPEAKER_"))
            self.assertIn("speaker_posterior", item)
            self.assertAlmostEqual(sum(item["speaker_posterior"].values()), 1.0, places=3)
            # confidence is the winning posterior (rounded to fewer places).
            self.assertAlmostEqual(
                item["speaker_posterior"][item["speaker"]], item["speaker_confidence"], places=2
            )
            self.assertEqual(item["speaker"], max(item["speaker_posterior"], key=item["speaker_posterior"].get))

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


class UnknownEscapeHatchTests(unittest.TestCase):
    """Step 6 'no clear match -> UNKNOWN' rule for the clustering path."""

    def _meeting_with_ambiguous_segment(self) -> list[dict]:
        # Two confident speakers plus one segment sitting midway between them.
        speaker_a = [[1.0, 0.0, 0.0]] * 3
        speaker_b = [[0.0, 1.0, 0.0]] * 3
        ambiguous = [0.7071, 0.7071, 0.0]
        embeddings = speaker_a + speaker_b + [ambiguous]
        return [
            {"segment_id": f"s{i}", "start_time": float(i), "end_time": float(i + 1), "embedding": vec}
            for i, vec in enumerate(embeddings)
        ]

    def test_ambiguous_segment_becomes_unknown(self) -> None:
        segments = self._meeting_with_ambiguous_segment()
        result = cluster_segments(
            np.zeros(0, dtype=np.float32),
            segments,
            num_speakers=2,
            temperature=1.0,
            min_confidence=0.6,
        )
        # The six confident segments keep real speaker labels.
        for item in result[:6]:
            self.assertTrue(item["speaker"].startswith("SPEAKER_"))
        # The midway segment cannot be confidently attributed.
        self.assertEqual(result[-1]["speaker"], "UNKNOWN")
        # Its low confidence is preserved and the full distribution stays.
        self.assertLess(result[-1]["speaker_confidence"], 0.6)
        self.assertAlmostEqual(sum(result[-1]["speaker_posterior"].values()), 1.0, places=3)

    def test_confident_segments_are_never_unknown(self) -> None:
        samples, segments = _two_speaker_meeting("ABABAB")
        result = cluster_segments(samples, segments, SAMPLE_RATE)
        self.assertTrue(all(item["speaker"] != "UNKNOWN" for item in result))

    def test_min_confidence_zero_disables_unknown(self) -> None:
        segments = self._meeting_with_ambiguous_segment()
        result = cluster_segments(
            np.zeros(0, dtype=np.float32),
            segments,
            num_speakers=2,
            temperature=1.0,
            min_confidence=0.0,
        )
        self.assertTrue(all(item["speaker"] != "UNKNOWN" for item in result))


if __name__ == "__main__":
    unittest.main()
