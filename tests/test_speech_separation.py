"""Tests for optional high-overlap speech separation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.errors import BackendExecutionError
from src.speech_separation import (
    DisabledSpeechSeparationAdapter,
    MockSpeechSeparationAdapter,
    NmfSeparationAdapter,
    SepFormerAdapter,
    get_separation_adapter,
    separate_speakers,
    separate_waveform,
)

SAMPLE_RATE = 16000


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
    return (np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)


def _best_correlation(estimate: np.ndarray, target: np.ndarray) -> float:
    """Absolute Pearson correlation between two equal-length signals."""
    a = estimate - estimate.mean()
    b = target - target.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(a @ b) / denom) if denom else 0.0


class SpeechSeparationFactoryTests(unittest.TestCase):
    def test_default_adapter_is_disabled(self) -> None:
        adapter = get_separation_adapter()
        self.assertIsInstance(adapter, DisabledSpeechSeparationAdapter)
        self.assertEqual(adapter.separate_array(np.ones(10, dtype=np.float32), SAMPLE_RATE), [])

    def test_builds_sepformer_without_loading_model(self) -> None:
        adapter = get_separation_adapter("sepformer", model_source="test/model")
        self.assertIsInstance(adapter, SepFormerAdapter)
        self.assertEqual(adapter.model_source, "test/model")

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown speech-separation backend"):
            get_separation_adapter("unknown")


class SpeechSeparationRuntimeTests(unittest.TestCase):
    def test_mock_adapter_returns_float32_sources(self) -> None:
        sources = separate_waveform(
            np.ones(8, dtype=np.float32),
            SAMPLE_RATE,
            MockSpeechSeparationAdapter([
                np.ones(8, dtype=np.float64),
                np.zeros(8, dtype=np.float64),
            ]),
        )
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(source.dtype == np.float32 for source in sources))

    def test_backend_failure_falls_back_to_no_sources(self) -> None:
        adapter = MagicMock()
        adapter.separate_array.side_effect = BackendExecutionError("failed")
        self.assertEqual(separate_waveform(np.ones(8), SAMPLE_RATE, adapter), [])

    def test_backend_failure_can_be_surfaced(self) -> None:
        adapter = MagicMock()
        adapter.separate_array.side_effect = BackendExecutionError("failed")
        with self.assertRaisesRegex(BackendExecutionError, "failed"):
            separate_waveform(np.ones(8), SAMPLE_RATE, adapter, fallback_on_error=False)

    @patch("src.speech_separation._to_model_batch", return_value="batch")
    @patch("src.speech_separation._load_sepformer")
    def test_sepformer_converts_time_source_output(self, mocked_load, mocked_batch) -> None:
        separator = MagicMock()
        separator.separate_batch.return_value = np.stack([
            np.ones(12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        ], axis=-1)[np.newaxis, :, :]
        mocked_load.return_value = separator

        sources = SepFormerAdapter(model_source="test/model").separate_array(
            np.ones(12, dtype=np.float32),
            SAMPLE_RATE,
        )

        self.assertEqual(len(sources), 2)
        self.assertEqual([source.size for source in sources], [12, 12])
        separator.separate_batch.assert_called_once_with("batch")

    def test_file_facade_writes_each_source(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            sf.write(input_path, np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1, SAMPLE_RATE)
            paths = separate_speakers(
                str(input_path),
                root / "separated",
                MockSpeechSeparationAdapter(),
            )
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(path).is_file() for path in paths))


class NmfSeparationAdapterTests(unittest.TestCase):
    """Real, dependency-free separation — actually exercised in CI (no mocks)."""

    def test_factory_builds_nmf_adapter(self) -> None:
        adapter = get_separation_adapter("nmf")
        self.assertIsInstance(adapter, NmfSeparationAdapter)
        self.assertEqual(adapter.name, "nmf")

    def test_returns_one_float32_stream_per_source(self) -> None:
        mixture = _source(220.0, 2.0) + _source(660.0, 7.0)
        sources = separate_waveform(mixture, SAMPLE_RATE, NmfSeparationAdapter())
        self.assertEqual(len(sources), 2)
        for source in sources:
            self.assertEqual(source.dtype, np.float32)
            self.assertEqual(len(source), len(mixture))

    def test_separates_two_distinct_sources(self) -> None:
        # The two NMF bases lock onto the two carriers and the activations capture
        # the two envelopes, so each output stream correlates with one true source
        # far better than with the other.
        source_a = _source(220.0, 2.0)
        source_b = _source(880.0, 9.0)
        estimates = separate_waveform(
            source_a + source_b, SAMPLE_RATE, NmfSeparationAdapter()
        )
        # Permutation-invariant: match each estimate to its best-fitting source.
        corr = np.array(
            [[_best_correlation(est, src) for src in (source_a, source_b)] for est in estimates]
        )
        for row in corr:
            self.assertGreater(row.max(), 0.8)
            self.assertGreater(row.max() - row.min(), 0.3)
        self.assertNotEqual(int(corr[0].argmax()), int(corr[1].argmax()))

    def test_is_deterministic_for_fixed_seed(self) -> None:
        mixture = _source(300.0, 3.0) + _source(900.0, 8.0)
        first = separate_waveform(mixture, SAMPLE_RATE, NmfSeparationAdapter())
        second = separate_waveform(mixture, SAMPLE_RATE, NmfSeparationAdapter())
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)

    def test_reconstruction_is_approximately_additive(self) -> None:
        # A soft Wiener mask partitions the mixture, so summing the estimated
        # sources should approximately rebuild the original mixture.
        mixture = _source(250.0, 3.0) + _source(750.0, 8.0)
        sources = separate_waveform(mixture, SAMPLE_RATE, NmfSeparationAdapter())
        reconstruction = np.sum(sources, axis=0)
        self.assertGreater(_best_correlation(reconstruction, mixture), 0.95)

    def test_empty_input_returns_no_sources(self) -> None:
        self.assertEqual(
            separate_waveform(np.array([], dtype=np.float32), SAMPLE_RATE, NmfSeparationAdapter()),
            [],
        )

    def test_rejects_invalid_num_sources(self) -> None:
        with self.assertRaises(ValueError):
            NmfSeparationAdapter(num_sources=0)


if __name__ == "__main__":
    unittest.main()
