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
    SepFormerAdapter,
    get_separation_adapter,
    separate_speakers,
    separate_waveform,
)

SAMPLE_RATE = 16000


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


if __name__ == "__main__":
    unittest.main()
