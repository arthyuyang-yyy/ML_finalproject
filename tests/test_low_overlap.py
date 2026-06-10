"""Tests for the low-overlap ASR + speaker-attribution path."""

import unittest

import numpy as np

from src.asr import MockASRAdapter
from src.low_overlap import LOW_OVERLAP_PATH, process_low_overlap_segments


class LowOverlapPathTests(unittest.TestCase):
    def test_processes_low_overlap_segments_with_asr_and_speaker(self) -> None:
        samples = np.ones(16000 * 3, dtype=np.float32) * 0.1
        segments = [
            {
                "meeting_id": "meeting_001",
                "segment_id": "meeting_001_seg_001",
                "start_time": 0.0,
                "end_time": 3.0,
                "overlap_score": 0.12,
            }
        ]
        diarization_turns = [
            {"speaker": "SPEAKER_00", "start_time": 0.0, "end_time": 3.0},
        ]

        processed = process_low_overlap_segments(
            samples,
            segments,
            asr_adapter=MockASRAdapter(confidence=0.91),
            diarization_turns=diarization_turns,
        )

        self.assertEqual(len(processed), 1)
        segment = processed[0]
        self.assertEqual(segment["processing_path"], LOW_OVERLAP_PATH)
        self.assertEqual(segment["speaker"], "SPEAKER_00")
        self.assertEqual(segment["speaker_confidence"], 1.0)
        self.assertEqual(segment["asr_confidence"], 0.91)
        self.assertIn("text", segment)
        self.assertEqual(segment["candidates"], [])
        self.assertEqual(segment["uncertainty_note"], "")


if __name__ == "__main__":
    unittest.main()
