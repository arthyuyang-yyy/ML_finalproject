"""Tests for high-overlap candidate generation."""

import unittest

import numpy as np

from src.candidates.generator import generate_high_overlap_candidates
from src.high_overlap import HIGH_OVERLAP_PATH, HIGH_OVERLAP_SPEAKER, process_high_overlap_segments


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
        self.assertFalse(segment["separation_applied"])
        self.assertIn("speaker attribution is uncertain", segment["uncertainty_note"])

    def test_separation_opt_in_generates_candidates_per_stream(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(16000 * 3).astype(np.float32) * 0.1
        segments = [
            {
                "meeting_id": "meeting_001",
                "segment_id": "m1_seg_010",
                "start_time": 0.0,
                "end_time": 3.0,
                "overlap_score": 0.8,
                "processing_path": HIGH_OVERLAP_PATH,
            }
        ]
        diarization_turns = [
            {"speaker": "SPEAKER_00", "start_time": 0.0, "end_time": 3.0},
            {"speaker": "SPEAKER_01", "start_time": 0.0, "end_time": 3.0},
        ]

        processed = process_high_overlap_segments(
            samples,
            segments,
            sample_rate=16000,
            diarization_turns=diarization_turns,
            separate=True,
        )

        segment = processed[0]
        self.assertTrue(segment["separation_applied"])
        candidates = segment["candidates"]
        self.assertGreaterEqual(len(candidates), 2)
        # Merged per-stream candidates keep unique, contiguous IDs.
        ids = [candidate["candidate_id"] for candidate in candidates]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "m1_seg_010_c1")
        # Streams are tagged with the diarization speaker hypotheses.
        speakers = {candidate["speaker"] for candidate in candidates}
        self.assertTrue(speakers & {"SPEAKER_00", "SPEAKER_01"})


if __name__ == "__main__":
    unittest.main()
