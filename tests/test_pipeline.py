"""Tests for the lightweight end-to-end meeting pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pipeline.config import PipelineConfig
from src.pipeline.io import read_json
from src.pipeline.run_pipeline import run_meeting_pipeline


class PipelineTests(unittest.TestCase):
    def test_run_meeting_pipeline_writes_per_meeting_artifacts(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            input_path = root / "input.wav"
            sf.write(input_path, audio, sample_rate)

            result = run_meeting_pipeline(
                str(input_path),
                "meeting_test",
                PipelineConfig(outputs_root=root / "outputs", language="en"),
            )

            output_dir = Path(result["output_dir"])
            self.assertEqual(output_dir.name, "meeting_test")
            self.assertTrue((output_dir / "preprocessed.wav").exists())
            self.assertTrue((output_dir / "vad_segments.json").exists())
            self.assertTrue((output_dir / "evidence_segments.json").exists())
            self.assertTrue((output_dir / "clips").exists())

            evidence = read_json(output_dir / "evidence_segments.json")
            self.assertGreaterEqual(len(evidence), 1)
            first = evidence[0]
            self.assertIn("evidence_id", first)
            self.assertIn("audio_clip_path", first)
            self.assertIn("route_reason", first)
            self.assertTrue(Path(first["audio_clip_path"]).exists())


if __name__ == "__main__":
    unittest.main()
