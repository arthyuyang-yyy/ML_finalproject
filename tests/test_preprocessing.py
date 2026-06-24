"""Tests for audio preprocessing and the silero VAD backend.

Run with::

    python -m unittest tests.test_preprocessing
"""

import unittest
from inspect import signature
from unittest.mock import patch

import numpy as np

from src.audio.preprocess import (
    decode_audio_with_pyav,
    load_audio,
    peak_normalize,
    preprocess_audio,
    reduce_stationary_noise,
    resample,
    resample_linear,
    segment_audio,
    segment_waveform,
    silero_vad,
    to_mono,
)

SAMPLE_RATE = 16000


def _tone(duration_s: float, freq: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_aac_container(path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE, container_format=None) -> None:
    import av

    output = av.open(str(path), "w", format=container_format)
    stream = output.add_stream("aac", rate=sample_rate)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="fltp", layout="mono")
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        output.mux(packet)
    for packet in stream.encode(None):
        output.mux(packet)
    output.close()


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

    def test_segment_audio_integration(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test.wav"
            sf.write(wav_path, _tone(2.0, amplitude=0.99), SAMPLE_RATE, subtype="FLOAT")
            # silero VAD is a learned model and does not fire on a pure sine tone,
            # so stub the detector to exercise the segment_audio load+dispatch path.
            with patch("src.audio.preprocess.silero_vad", return_value=[(0.0, 1.0)]):
                segments = segment_audio(str(wav_path), meeting_id="test")
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]["meeting_id"], "test")

    def test_audio_package_wrapper_uses_shared_preprocess(self) -> None:
        self.assertIn("target_sample_rate", signature(preprocess_audio).parameters)
        self.assertIn("target_sr", signature(preprocess_audio).parameters)
        self.assertIn("denoise", signature(preprocess_audio).parameters)

    def test_load_audio_respects_normalize_flag(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test.wav"
            raw = np.ones(16000, dtype=np.float32) * 0.5
            sf.write(wav_path, raw, SAMPLE_RATE, subtype="FLOAT")
            loaded, _ = load_audio(str(wav_path), normalize=False)
            np.testing.assert_array_almost_equal(loaded, raw)

    def test_load_audio_uses_soundfile_without_pyav_for_wav(self) -> None:
        import soundfile as sf
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test.wav"
            sf.write(path, _tone(1.0), SAMPLE_RATE)
            with patch("src.audio.preprocess.decode_audio_with_pyav") as mocked_decode:
                loaded, sample_rate = load_audio(str(path))
            mocked_decode.assert_not_called()
            self.assertEqual(sample_rate, SAMPLE_RATE)
            self.assertEqual(loaded.ndim, 1)


class AudioDecodeTests(unittest.TestCase):
    def test_pyav_decodes_m4a_to_native_rate_pcm(self) -> None:
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("PyAV is not installed")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "meeting.m4a"
            _write_aac_container(path, _tone(2.0))
            samples, sample_rate = decode_audio_with_pyav(path)
            self.assertEqual(sample_rate, SAMPLE_RATE)
            self.assertEqual(samples.ndim, 2)
            self.assertEqual(samples.shape[1], 1)
            self.assertGreater(samples.size, 0)

    def test_load_audio_decodes_mp4_and_standardizes_for_asr(self) -> None:
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("PyAV is not installed")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "meeting.mp4"
            _write_aac_container(path, _tone(2.0), container_format="mp4")
            samples, sample_rate = load_audio(str(path))
            self.assertEqual(sample_rate, SAMPLE_RATE)
            self.assertEqual(samples.ndim, 1)
            self.assertEqual(samples.dtype, np.float32)
            self.assertAlmostEqual(float(np.max(np.abs(samples))), 0.97, places=4)

    def test_load_audio_resamples_only_after_native_rate_decode(self) -> None:
        decoded = np.ones((8000, 1), dtype=np.float32) * 0.2
        with patch("src.audio.preprocess.Path.is_file", return_value=True):
            with patch("src.audio.preprocess._decode_audio", return_value=(decoded, 8000)):
                with patch("src.audio.preprocess.resample", wraps=resample) as mocked_resample:
                    samples, sample_rate = load_audio("ignored.wav", target_sample_rate=16000)
        mocked_resample.assert_called_once()
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.size, 16000)

    def test_load_audio_reports_invalid_container_clearly(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "broken.m4a"
            path.write_text("not audio", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unable to decode audio file"):
                load_audio(str(path))

    def test_load_audio_rejects_missing_path(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_audio("/definitely/missing/audio.m4a")

    def test_optional_denoise_is_disabled_by_default(self) -> None:
        decoded = np.ones((16000, 1), dtype=np.float32) * 0.2
        with patch("src.audio.preprocess.Path.is_file", return_value=True):
            with patch("src.audio.preprocess._decode_audio", return_value=(decoded, SAMPLE_RATE)):
                with patch("src.audio.preprocess.reduce_stationary_noise") as mocked_denoise:
                    load_audio("ignored.wav")
        mocked_denoise.assert_not_called()

    def test_optional_denoise_runs_when_enabled(self) -> None:
        decoded = np.ones((16000, 1), dtype=np.float32) * 0.2
        with patch("src.audio.preprocess.Path.is_file", return_value=True):
            with patch("src.audio.preprocess._decode_audio", return_value=(decoded, SAMPLE_RATE)):
                with patch(
                    "src.audio.preprocess.reduce_stationary_noise",
                    return_value=np.ones(16000, dtype=np.float32) * 0.1,
                ) as mocked_denoise:
                    load_audio("ignored.wav", denoise=True, denoise_strength=0.4)
        mocked_denoise.assert_called_once()

    def test_denoise_rejects_invalid_strength_before_loading_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "strength"):
            reduce_stationary_noise(_tone(1.0), SAMPLE_RATE, strength=1.5)


class SileroVadTests(unittest.TestCase):
    def test_empty_signal_yields_no_regions(self) -> None:
        self.assertEqual(silero_vad(np.zeros(0, dtype=np.float32), SAMPLE_RATE), [])

    def test_forwards_threshold_and_padding_to_detector(self) -> None:
        with patch("faster_whisper.vad.get_speech_timestamps",
                   return_value=[{"start": 0, "end": 16000}]) as detector:
            silero_vad(_tone(1.0), SAMPLE_RATE, threshold=0.4, min_silence_ms=300, speech_pad_ms=100)
        options = detector.call_args[0][1]
        self.assertEqual(options.threshold, 0.4)
        self.assertEqual(options.min_silence_duration_ms, 300)
        self.assertEqual(options.speech_pad_ms, 100)


class SegmentWaveformTests(unittest.TestCase):
    def test_runs_silero_vad_and_maps_schema(self) -> None:
        with patch("src.audio.preprocess.silero_vad", return_value=[(0.5, 1.5), (2.5, 3.5)]) as silero:
            segments = segment_waveform(np.zeros(16000, dtype=np.float32), SAMPLE_RATE, meeting_id="demo")
        silero.assert_called_once()
        self.assertEqual(len(segments), 2)
        first = segments[0]
        self.assertEqual(set(first), {"meeting_id", "segment_id", "start_time", "end_time"})
        self.assertEqual(first["meeting_id"], "demo")
        self.assertEqual(first["segment_id"], "demo_seg_001")
        self.assertEqual((first["start_time"], first["end_time"]), (0.5, 1.5))


if __name__ == "__main__":
    unittest.main()
