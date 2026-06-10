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
            self.assertTrue((output_dir / "low_overlap_segments.json").exists())
            self.assertTrue((output_dir / "evidence_segments.json").exists())
            self.assertTrue((output_dir / "clips").exists())

            evidence = read_json(output_dir / "evidence_segments.json")
            self.assertGreaterEqual(len(evidence), 1)
            first = evidence[0]
            self.assertIn("evidence_id", first)
            self.assertIn("audio_clip_path", first)
            self.assertIn("route_reason", first)
            self.assertTrue(Path(first["audio_clip_path"]).exists())

            low_overlap = read_json(output_dir / "low_overlap_segments.json")
            self.assertGreaterEqual(len(low_overlap), 1)
            low = low_overlap[0]
            self.assertEqual(low["processing_path"], "low_overlap_cluster")
            self.assertIn("text", low)
            self.assertIn("speaker", low)
            self.assertIn("asr_confidence", low)
            self.assertIn("speaker_confidence", low)
            self.assertEqual(low["candidates"], [])

    def test_high_overlap_pipeline_output_preserves_candidates(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate * 2) / sample_rate
            audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            input_path = root / "input.wav"
            sf.write(input_path, audio, sample_rate)

            result = run_meeting_pipeline(
                str(input_path),
                "meeting_high",
                PipelineConfig(outputs_root=root / "outputs", language="en", overlap_threshold=0.0),
            )

            high_overlap = read_json(Path(result["output_dir"]) / "high_overlap_candidates.json")
            self.assertGreaterEqual(len(high_overlap), 1)
            segment = high_overlap[0]
            self.assertEqual(segment["speaker"], "MIXED")
            self.assertEqual(segment["text"], "")
            self.assertEqual(segment["processing_path"], "high_overlap_candidate")
            self.assertGreaterEqual(len(segment["candidates"]), 1)
            self.assertIn("candidate_id", segment["candidates"][0])
            self.assertIn("decode_config", segment["candidates"][0])


if __name__ == "__main__":
    unittest.main()
