"""Tests for audio clip export helpers."""

import tempfile
import unittest

from pathlib import Path

import numpy as np

from src.audio.clipper import write_segment_clips


class ClipExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_writes_one_clip_per_segment(self) -> None:
        try:
            import soundfile as sf  # noqa: F401
        except ImportError:
            self.skipTest("soundfile is not installed")

        samples = np.ones(16000 * 3, dtype=np.float32) * 0.3
        segments = [
            {"segment_id": "s1", "start_time": 0.0, "end_time": 1.0},
            {"segment_id": "s2", "start_time": 1.0, "end_time": 2.0},
        ]
        updated = write_segment_clips(samples, 16000, segments, self.temp_dir.name)

        self.assertEqual(len(updated), 2)
        self.assertTrue((Path(self.temp_dir.name) / "s1.wav").exists())
        self.assertTrue((Path(self.temp_dir.name) / "s2.wav").exists())
        self.assertEqual(updated[0]["audio_clip_path"], str(Path(self.temp_dir.name) / "s1.wav"))

    def test_uses_evidence_id_as_filename(self) -> None:
        try:
            import soundfile as sf  # noqa: F401
        except ImportError:
            self.skipTest("soundfile is not installed")

        samples = np.ones(16000, dtype=np.float32) * 0.1
        segments = [{"segment_id": "seg1", "evidence_id": "ev_abc", "start_time": 0.0, "end_time": 1.0}]
        updated = write_segment_clips(samples, 16000, segments, self.temp_dir.name)
        self.assertTrue((Path(self.temp_dir.name) / "ev_abc.wav").exists())
        self.assertEqual(updated[0]["audio_clip_path"], str(Path(self.temp_dir.name) / "ev_abc.wav"))

    def test_clip_boundaries_clamped(self) -> None:
        try:
            import soundfile as sf  # noqa: F401
        except ImportError:
            self.skipTest("soundfile is not installed")

        samples = np.ones(16000 * 2, dtype=np.float32) * 0.1
        # start before signal, end after signal
        segments = [{"segment_id": "s1", "start_time": -1.0, "end_time": 10.0}]
        updated = write_segment_clips(samples, 16000, segments, self.temp_dir.name)
        self.assertIn("audio_clip_path", updated[0])

    def test_empty_segments_returns_empty(self) -> None:
        try:
            import soundfile as sf  # noqa: F401
        except ImportError:
            self.skipTest("soundfile is not installed")

        samples = np.ones(16000, dtype=np.float32)
        updated = write_segment_clips(samples, 16000, [], self.temp_dir.name)
        self.assertEqual(updated, [])


if __name__ == "__main__":
    unittest.main()
