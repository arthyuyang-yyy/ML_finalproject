"""Tests for audio preprocessing and the energy-based VAD baseline.

Run with::

    python -m unittest tests.test_preprocessing
"""

import unittest
from inspect import signature

import numpy as np

from src.audio.preprocess import (
    energy_vad,
    frame_rms,
    load_audio,
    peak_normalize,
    preprocess_audio,
    resample,
    resample_linear,
    segment_audio,
    segment_waveform,
    to_mono,
)

SAMPLE_RATE = 16000


def _tone(duration_s: float, freq: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


def _two_burst_signal() -> np.ndarray:
    # 0.5s silence | 1.0s speech | 1.0s silence | 1.0s speech | 0.5s silence
    return np.concatenate([
        _silence(0.5),
        _tone(1.0),
        _silence(1.0),
        _tone(1.0),
        _silence(0.5),
    ])


def _short_gap_signal() -> np.ndarray:
    # 0.5s silence | 0.4s speech | 0.4s silence | 0.4s speech | 0.5s silence
    return np.concatenate([
        _silence(0.5),
        _tone(0.4),
        _silence(0.4),
        _tone(0.4),
        _silence(0.5),
    ])


class HelperTests(unittest.TestCase):
    def test_to_mono_averages_channels(self) -> None:
        stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(to_mono(stereo), [2.0, 3.0])

    def test_peak_normalize_scales_to_target(self) -> None:
        out = peak_normalize(np.array([0.1, -0.2, 0.05], dtype=np.float32), target_peak=0.8)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.8, places=5)

    def test_peak_normalize_handles_silence(self) -> None:
        out = peak_normalize(np.zeros(10, dtype=np.float32))
        self.assertTrue(np.all(out == 0.0))

    def test_resample_changes_length_proportionally(self) -> None:
        signal = _tone(1.0)
        out = resample_linear(signal, SAMPLE_RATE, SAMPLE_RATE // 2)
        self.assertEqual(out.size, SAMPLE_RATE // 2)

    def test_resample_prefers_shared_public_entrypoint(self) -> None:
        signal = _tone(1.0)
        out = resample(signal, SAMPLE_RATE, SAMPLE_RATE // 2)
        self.assertEqual(out.size, SAMPLE_RATE // 2)

    def test_resample_rejects_negative_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            resample_linear(_tone(0.1), -1, 16000)

    def test_frame_rms_positive_length_required(self) -> None:
        with self.assertRaises(ValueError):
            frame_rms(_tone(0.1), 0, 160)

    def test_frame_rms_short_signal_returns_empty(self) -> None:
        rms, starts = frame_rms(_tone(0.01), 400, 160)
        self.assertEqual(rms.size, 0)
        self.assertEqual(starts.size, 0)

    def test_frame_rms_computes_expected_values(self) -> None:
        samples = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        rms, starts = frame_rms(samples, frame_length=3, hop_length=1)
        self.assertEqual(starts.size, 3)
        self.assertGreater(rms[0], 0.0)

    def test_segment_audio_integration(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, _tone(2.0, amplitude=0.99), SAMPLE_RATE, subtype="FLOAT")
            segments = segment_audio(tmp.name, meeting_id="test")
            self.assertGreaterEqual(len(segments), 1)
            self.assertEqual(segments[0]["meeting_id"], "test")

    def test_audio_package_wrapper_uses_shared_preprocess(self) -> None:
        self.assertIn("target_sample_rate", signature(preprocess_audio).parameters)
        self.assertIn("target_sr", signature(preprocess_audio).parameters)

    def test_load_audio_respects_normalize_flag(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            raw = np.ones(16000, dtype=np.float32) * 0.5
            sf.write(tmp.name, raw, SAMPLE_RATE, subtype="FLOAT")
            loaded, _ = load_audio(tmp.name, normalize=False)
            np.testing.assert_array_almost_equal(loaded, raw)


class EnergyVadTests(unittest.TestCase):
    def test_detects_two_speech_regions(self) -> None:
        regions = energy_vad(_two_burst_signal(), SAMPLE_RATE)
        self.assertEqual(len(regions), 2)

    def test_regions_align_with_bursts(self) -> None:
        regions = energy_vad(_two_burst_signal(), SAMPLE_RATE)
        (s1, e1), (s2, e2) = regions
        # First burst spans ~0.5-1.5s, second ~2.5-3.5s (with small padding).
        self.assertAlmostEqual(s1, 0.5, delta=0.1)
        self.assertAlmostEqual(e1, 1.5, delta=0.1)
        self.assertAlmostEqual(s2, 2.5, delta=0.1)
        self.assertAlmostEqual(e2, 3.5, delta=0.1)

    def test_silence_yields_no_regions(self) -> None:
        self.assertEqual(energy_vad(_silence(2.0), SAMPLE_RATE), [])

    def test_empty_signal_yields_no_regions(self) -> None:
        self.assertEqual(energy_vad(np.zeros(0, dtype=np.float32), SAMPLE_RATE), [])

    def test_regions_are_ordered_and_non_overlapping(self) -> None:
        regions = energy_vad(_two_burst_signal(), SAMPLE_RATE)
        for (s, e) in regions:
            self.assertLess(s, e)
        for earlier, later in zip(regions, regions[1:]):
            self.assertLessEqual(earlier[1], later[0])

    def test_short_neighboring_regions_are_merged(self) -> None:
        regions = energy_vad(_short_gap_signal(), SAMPLE_RATE)
        self.assertEqual(len(regions), 1)
        self.assertGreaterEqual(regions[0][1] - regions[0][0], 1.0)

    def test_long_regions_are_split(self) -> None:
        regions = energy_vad(_tone(31.0), SAMPLE_RATE, max_segment_s=20.0, target_segment_s=12.0)
        self.assertGreaterEqual(len(regions), 2)
        for start, end in regions:
            self.assertLessEqual(end - start, 20.0)


class SegmentWaveformTests(unittest.TestCase):
    def test_returns_schema_compatible_segments(self) -> None:
        segments = segment_waveform(_two_burst_signal(), SAMPLE_RATE, meeting_id="demo")
        self.assertEqual(len(segments), 2)
        first = segments[0]
        self.assertEqual(set(first), {"meeting_id", "segment_id", "start_time", "end_time"})
        self.assertEqual(first["meeting_id"], "demo")
        self.assertEqual(first["segment_id"], "demo_seg_001")
        self.assertLess(first["start_time"], first["end_time"])


if __name__ == "__main__":
    unittest.main()
