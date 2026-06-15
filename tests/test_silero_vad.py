"""Tests for the silero VAD backend and segment_waveform method dispatch."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from src.audio.preprocess import segment_waveform, silero_vad

_HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None


class SegmentWaveformDispatchTests(unittest.TestCase):
    def test_unknown_method_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown VAD method"):
            segment_waveform(np.zeros(16000, dtype=np.float32), 16000, method="bogus")

    def test_energy_is_default_and_routes_to_energy_vad(self) -> None:
        with patch("src.audio.preprocess.energy_vad", return_value=[(0.0, 1.0)]) as energy:
            segs = segment_waveform(np.zeros(16000, dtype=np.float32), 16000, meeting_id="m")
        energy.assert_called_once()
        self.assertEqual(segs[0]["segment_id"], "m_seg_001")
        self.assertEqual((segs[0]["start_time"], segs[0]["end_time"]), (0.0, 1.0))

    def test_silero_method_routes_to_silero_vad(self) -> None:
        with patch("src.audio.preprocess.silero_vad", return_value=[(1.0, 3.0)]) as silero:
            segs = segment_waveform(np.zeros(16000, dtype=np.float32), 16000, method="silero")
        silero.assert_called_once()
        self.assertEqual((segs[0]["start_time"], segs[0]["end_time"]), (1.0, 3.0))


@unittest.skipUnless(_HAS_FASTER_WHISPER, "faster-whisper not installed")
class SileroVadTests(unittest.TestCase):
    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(silero_vad(np.zeros(0, dtype=np.float32), 16000), [])

    def test_converts_sample_timestamps_to_seconds(self) -> None:
        # Patch the detector so the test is deterministic and needs no real speech.
        with patch("faster_whisper.vad.get_speech_timestamps",
                   return_value=[{"start": 16000, "end": 48000}]):
            regions = silero_vad(np.ones(64000, dtype=np.float32), 16000)
        self.assertEqual(regions, [(1.0, 3.0)])

    def test_stereo_input_is_mixed_to_mono(self) -> None:
        stereo = np.zeros((32000, 2), dtype=np.float32)
        with patch("faster_whisper.vad.get_speech_timestamps", return_value=[]) as detector:
            silero_vad(stereo, 16000)
        passed = detector.call_args[0][0]
        self.assertEqual(passed.ndim, 1)


if __name__ == "__main__":
    unittest.main()
