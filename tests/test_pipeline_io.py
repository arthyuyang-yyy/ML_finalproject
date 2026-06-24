"""Tests for pipeline I/O helpers and configuration."""

import tempfile
import unittest

from pathlib import Path

from src.pipeline.config import PipelineConfig
from src.pipeline.io import ensure_meeting_dirs, read_json, write_json


class PipelineIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ensure_meeting_dirs_creates_directories(self) -> None:
        base = Path(self.temp_dir.name) / "meeting_test"
        paths = ensure_meeting_dirs(base)
        self.assertTrue(paths["base"].exists())
        self.assertTrue(paths["clips"].exists())
        self.assertIn("preprocessed", paths)
        self.assertIn("vad_segments", paths)
        self.assertIn("evidence_segments", paths)

    def test_write_and_read_json_roundtrip(self) -> None:
        path = Path(self.temp_dir.name) / "test.json"
        data = {"key": [1, 2, 3], "nested": {"a": True}}
        write_json(path, data)
        read_back = read_json(path)
        self.assertEqual(read_back, data)

    def test_write_json_creates_parent_directories(self) -> None:
        path = Path(self.temp_dir.name) / "deep" / "nested" / "data.json"
        write_json(path, [1, 2, 3])
        self.assertTrue(path.exists())

    def test_read_json_handles_non_ascii(self) -> None:
        path = Path(self.temp_dir.name) / "unicode.json"
        data = {"text": "你好世界", "notes": "emoji: 🚀"}
        write_json(path, data)
        read_back = read_json(path)
        self.assertEqual(read_back, data)


class ConfigTests(unittest.TestCase):
    def test_default_values(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.target_sample_rate, 16000)
        self.assertEqual(cfg.language, "und")
        self.assertEqual(cfg.low_overlap_asr_model, "mock")
        self.assertEqual(cfg.vad_max_segment_s, 30.0)
        self.assertEqual(cfg.vad_speech_pad_ms, 400)
        self.assertEqual(cfg.vad_min_silence_ms, 500)
        self.assertEqual(cfg.asr_context_padding_s, 0.2)
        self.assertEqual(cfg.high_overlap_min_segment_s, 2.0)
        self.assertEqual(cfg.high_overlap_decode_context_s, 2.0)
        self.assertEqual(cfg.suspected_overlap_threshold, 0.3)
        self.assertEqual(cfg.speech_separation_backend, "none")
        self.assertGreaterEqual(cfg.overlap_threshold, 0.0)
        self.assertLessEqual(cfg.overlap_threshold, 1.0)

    def test_meeting_dir_resolves_correctly(self) -> None:
        cfg = PipelineConfig(outputs_root=Path("/tmp/outputs"))
        self.assertEqual(cfg.meeting_dir("m1"), Path("/tmp/outputs/m1"))

    def test_config_is_immutable(self) -> None:
        cfg = PipelineConfig()
        with self.assertRaises(Exception):
            cfg.target_sample_rate = 8000  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
