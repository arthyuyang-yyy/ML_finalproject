"""Tests for the lightweight end-to-end meeting pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.pipeline.config import PipelineConfig
from src.pipeline.io import read_json
from src.pipeline.run_pipeline import run_meeting_pipeline
from src.audio.preprocess import load_audio
from src.llm.gemma_client import GemmaClient


class PipelineTests(unittest.TestCase):
    def test_pipeline_accepts_structured_gemma_client(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            client = GemmaClient(generator=lambda prompt: {
                "meeting_id": "meeting_gemma",
                "meeting_summary": "The meeting selected the mock baseline.",
                "events": [{
                    "event_id": "ev_001",
                    "event_type": "decision",
                    "content": "Use the mock baseline for pipeline testing.",
                    "speakers": ["SPEAKER_00"],
                    "evidence_ids": ["meeting_gemma_seg_001"],
                    "confidence": "high",
                }],
            })
            result = run_meeting_pipeline(
                str(input_path),
                "meeting_gemma",
                PipelineConfig(outputs_root=root / "outputs"),
                llm_client=client,
            )

            self.assertEqual(result["meeting_events"]["events"][0]["event_type"], "decision")
            stored = read_json(Path(result["output_dir"]) / "meeting_events.json")
            self.assertEqual(stored["meeting_summary"], "The meeting selected the mock baseline.")
            memory_path = Path(result["artifacts"]["long_term_episodic_memory"])
            self.assertTrue(memory_path.exists())
            memory = read_json(memory_path)
            self.assertEqual(memory[0]["event_type"], "decision")
            self.assertEqual(memory[0]["evidence_ids"], ["meeting_gemma_seg_001"])

            repeated = run_meeting_pipeline(
                str(input_path),
                "meeting_gemma",
                PipelineConfig(outputs_root=root / "outputs"),
                llm_client=client,
            )
            self.assertEqual(repeated["long_term_memory_size"], 1)
            self.assertEqual(len(read_json(memory_path)), 1)

    def test_pipeline_calls_load_audio_for_preprocessing_and_diarization(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            with patch("src.audio.preprocess.load_audio", wraps=load_audio) as mocked_load:
                run_meeting_pipeline(
                    str(input_path),
                    "meeting_single_read",
                    PipelineConfig(outputs_root=root / "outputs"),
                )

            self.assertEqual(mocked_load.call_count, 2)

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
            self.assertTrue((output_dir / "diarization.json").exists())
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

            meeting_events = read_json(output_dir / "meeting_events.json")
            self.assertEqual(meeting_events["meeting_id"], "meeting_test")
            self.assertIn("meeting_summary", meeting_events)
            self.assertIsInstance(meeting_events["events"], list)

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
