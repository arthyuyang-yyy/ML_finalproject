"""Tests for the single-channel speech-separation baseline (Step 11).

Run with::

    python -m unittest tests.test_speech_separation
"""

import unittest

import numpy as np

from src.speech_separation import (
    NmfSeparationBackend,
    separate_waveform,
)

SAMPLE_RATE = 16000


def _tone(freq: float, duration_s: float = 1.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A deterministic single-frequency source."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


def _source(
    freq: float, mod_freq: float, duration_s: float = 1.5, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """A tone with a distinct temporal envelope.

    NMF separates sources by their differing *temporal activations*, so a usable
    test source needs both a distinct carrier (spectral shape) and a distinct
    amplitude modulation (activation) — stationary tones are ill-posed for
    single-channel NMF and are not a meaningful test of separation.
    """
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    return (np.sin(2 * np.pi * freq * t) * envelope).astype(np.float64)


def _best_correlation(estimate: np.ndarray, target: np.ndarray) -> float:
    """Absolute Pearson correlation between two equal-length signals."""
    a = estimate - estimate.mean()
    b = target - target.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(a @ b) / denom) if denom else 0.0


class SeparateWaveformTests(unittest.TestCase):
    def test_returns_one_stream_per_source(self) -> None:
        mixture = _source(220.0, 2.0) + _source(660.0, 7.0)
        sources = separate_waveform(mixture, SAMPLE_RATE, num_sources=2)
        self.assertEqual(len(sources), 2)
        for source in sources:
            self.assertEqual(len(source), len(mixture))

    def test_separates_two_distinct_sources(self) -> None:
        # The two NMF bases lock onto the two carriers and the activations capture
        # the two envelopes, so each output stream correlates with one true source
        # far better than with the other.
        source_a = _source(220.0, 2.0)
        source_b = _source(880.0, 9.0)
        estimates = separate_waveform(source_a + source_b, SAMPLE_RATE, num_sources=2)

        # Permutation-invariant: match each estimate to its best-fitting source.
        corr = np.array(
            [[_best_correlation(est, src) for src in (source_a, source_b)] for est in estimates]
        )
        for row in corr:
            self.assertGreater(row.max(), 0.8)
            self.assertGreater(row.max() - row.min(), 0.3)
        # The two estimates explain different sources (one each).
        self.assertNotEqual(int(corr[0].argmax()), int(corr[1].argmax()))

    def test_is_deterministic_for_fixed_seed(self) -> None:
        mixture = _source(300.0, 3.0) + _source(900.0, 8.0)
        first = separate_waveform(mixture, SAMPLE_RATE, num_sources=2)
        second = separate_waveform(mixture, SAMPLE_RATE, num_sources=2)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)

    def test_single_source_returns_copy(self) -> None:
        mixture = _tone(440.0)
        sources = separate_waveform(mixture, SAMPLE_RATE, num_sources=1)
        self.assertEqual(len(sources), 1)
        np.testing.assert_allclose(sources[0], mixture)

    def test_short_clip_does_not_crash(self) -> None:
        tiny = np.ones(16, dtype=np.float64)
        sources = separate_waveform(tiny, SAMPLE_RATE, num_sources=2)
        self.assertEqual(len(sources), 2)
        for source in sources:
            self.assertEqual(len(source), len(tiny))

    def test_more_components_than_sources_groups_into_sources(self) -> None:
        mixture = _source(200.0, 2.0) + _source(700.0, 6.0)
        backend = NmfSeparationBackend(n_components=4, seed=1)
        sources = separate_waveform(mixture, SAMPLE_RATE, num_sources=2, backend=backend)
        self.assertEqual(len(sources), 2)

    def test_reconstruction_is_approximately_additive(self) -> None:
        # A soft Wiener mask partitions the mixture, so summing the estimated
        # sources should approximately rebuild the original mixture.
        mixture = _source(250.0, 3.0) + _source(750.0, 8.0)
        sources = separate_waveform(mixture, SAMPLE_RATE, num_sources=2)
        reconstruction = np.sum(sources, axis=0)
        self.assertGreater(_best_correlation(reconstruction, mixture), 0.95)

    def test_rejects_invalid_num_sources(self) -> None:
        with self.assertRaises(ValueError):
            separate_waveform(_tone(440.0), SAMPLE_RATE, num_sources=0)


if __name__ == "__main__":
    unittest.main()
