"""Tests for high-overlap candidate generation."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.candidates.generator import generate_high_overlap_candidates
from src.high_overlap import HIGH_OVERLAP_PATH, HIGH_OVERLAP_SPEAKER, process_high_overlap_segments
from src.speech_separation import MockSpeechSeparationAdapter


class CandidateGeneratorTests(unittest.TestCase):
    def test_fallback_candidates_preserve_decode_configs(self) -> None:
        candidates = generate_high_overlap_candidates(
            {"segment_id": "m1_seg_009", "text": ""},
            samples=np.ones(16000, dtype=np.float32),
            sample_rate=16000,
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["candidate_id"], "m1_seg_009_c1")
        self.assertEqual(candidates[0]["speaker"], "UNKNOWN")
        self.assertIn("decode_config", candidates[0])
        self.assertIn("High-overlap segment", candidates[0]["uncertainty_note"])


class HighOverlapPathTests(unittest.TestCase):
    def test_high_overlap_record_keeps_main_transcript_empty(self) -> None:
        samples = np.ones(16000 * 6, dtype=np.float32) * 0.1
        segments = [
            {
                "meeting_id": "meeting_001",
                "segment_id": "m1_seg_009",
                "start_time": 0.0,
                "end_time": 6.0,
                "overlap_score": 0.78,
                "processing_path": HIGH_OVERLAP_PATH,
            }
        ]

        processed = process_high_overlap_segments(samples, segments, sample_rate=16000)

        self.assertEqual(len(processed), 1)
        segment = processed[0]
        self.assertEqual(segment["speaker"], HIGH_OVERLAP_SPEAKER)
        self.assertEqual(segment["text"], "")
        self.assertEqual(segment["processing_path"], HIGH_OVERLAP_PATH)
        self.assertEqual(segment["speaker_confidence"], 0.35)
        self.assertGreaterEqual(len(segment["candidates"]), 2)
        self.assertIn("speaker attribution is uncertain", segment["uncertainty_note"])

    @patch("src.candidates.generator._load_faster_whisper_model")
    def test_separated_sources_become_distinct_asr_candidates(self, mocked_load) -> None:
        model = MagicMock()
        model.transcribe.side_effect = [
            (
                iter([SimpleNamespace(text=" source one", avg_logprob=-0.1, no_speech_prob=0.0)]),
                SimpleNamespace(language="en"),
            ),
            (
                iter([SimpleNamespace(text=" source two", avg_logprob=-0.2, no_speech_prob=0.0)]),
                SimpleNamespace(language="en"),
            ),
        ]
        mocked_load.return_value = model
        samples = np.ones(16000 * 2, dtype=np.float32) * 0.1
        segments = [{
            "meeting_id": "meeting_001",
            "segment_id": "m1_seg_009",
            "start_time": 0.0,
            "end_time": 2.0,
            "overlap_score": 0.78,
            "processing_path": HIGH_OVERLAP_PATH,
        }]

        processed = process_high_overlap_segments(
            samples,
            segments,
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start_time": 0.0, "end_time": 2.0},
                {"speaker": "SPEAKER_01", "start_time": 0.0, "end_time": 2.0},
            ],
            separation_adapter=MockSpeechSeparationAdapter([
                samples * 0.6,
                samples * 0.4,
            ]),
        )

        candidates = processed[0]["candidates"]
        self.assertEqual([candidate["text"] for candidate in candidates], ["source one", "source two"])
        self.assertEqual(
            [candidate["speaker"] for candidate in candidates],
            ["SEPARATED_SOURCE_01", "SEPARATED_SOURCE_02"],
        )
        self.assertTrue(all(candidate["decode_config"]["backend"] == "mock" for candidate in candidates))

    @patch(
        "src.high_overlap.generate_high_overlap_candidates",
        return_value=[{"candidate_id": "m1_seg_009_c1", "confidence": 0.5}],
    )
    @patch("src.high_overlap.generate_separated_source_candidates", return_value=[])
    def test_empty_separation_candidates_keep_existing_fallback(
        self,
        mocked_separated,
        mocked_fallback,
    ) -> None:
        samples = np.ones(16000, dtype=np.float32) * 0.1
        segments = [{
            "meeting_id": "meeting_001",
            "segment_id": "m1_seg_009",
            "start_time": 0.0,
            "end_time": 1.0,
            "overlap_score": 0.78,
            "processing_path": HIGH_OVERLAP_PATH,
        }]

        processed = process_high_overlap_segments(
            samples,
            segments,
            separation_adapter=MockSpeechSeparationAdapter(),
        )

        self.assertEqual(processed[0]["candidates"], mocked_fallback.return_value)
        mocked_fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
