"""Tests for speaker diarization and clustering."""

import unittest
from unittest.mock import patch

from src.diarization import (
    DEFAULT_SPEAKER_CONFIDENCE,
    assign_speakers_to_segments,
    diarize_with_pyannote,
    _best_speaker_for_segment,
)
from src.fallbacks.diarization import cluster_speakers


class ClusterSpeakersTests(unittest.TestCase):
    def test_missing_pyannote_token_logs_fallback_reason(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertLogs("src.diarization.core", level="INFO") as captured:
                result = diarize_with_pyannote("missing.wav")
        self.assertIsNone(result)
        self.assertIn("no Hugging Face token", " ".join(captured.output))

    def test_marks_speakers_unknown_without_diarization(self) -> None:
        segments = [
            {"segment_id": "s1", "start_time": 0.0, "end_time": 1.0},
            {"segment_id": "s2", "start_time": 1.0, "end_time": 2.0},
            {"segment_id": "s3", "start_time": 2.0, "end_time": 3.0},
        ]
        clustered = cluster_speakers(segments)
        self.assertEqual(len(clustered), 3)
        self.assertTrue(all(segment["speaker"] == "UNKNOWN" for segment in clustered))

    def test_assignment_uses_coverage_threshold_and_mixed_label(self) -> None:
        segments = [
            {"segment_id": "dominant", "start_time": 0.0, "end_time": 10.0},
            {"segment_id": "mixed", "start_time": 10.0, "end_time": 20.0},
            {"segment_id": "unknown", "start_time": 20.0, "end_time": 30.0},
        ]
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 8.0},
            {"speaker": "A", "start_time": 10.0, "end_time": 15.0},
            {"speaker": "B", "start_time": 15.0, "end_time": 20.0},
        ]
        assigned = assign_speakers_to_segments(segments, turns)
        self.assertEqual([item["speaker"] for item in assigned], ["A", "MIXED", "UNKNOWN"])

    def test_preserves_existing_speaker_when_present(self) -> None:
        segments = [{"segment_id": "s1", "speaker": "ALICE", "start_time": 0.0, "end_time": 1.0}]
        clustered = cluster_speakers(segments)
        self.assertEqual(clustered[0]["speaker"], "ALICE")

    def test_empty_segments_returns_empty(self) -> None:
        self.assertEqual(cluster_speakers([]), [])

    def test_assigns_default_confidence(self) -> None:
        segments = [{"segment_id": "s1", "start_time": 0.0, "end_time": 1.0}]
        clustered = cluster_speakers(segments)
        self.assertEqual(clustered[0]["speaker_confidence"], DEFAULT_SPEAKER_CONFIDENCE)


class AssignSpeakersToSegmentsTests(unittest.TestCase):
    def test_delegates_to_cluster_when_no_diarization_turns(self) -> None:
        segments = [{"segment_id": "s1", "start_time": 0.0, "end_time": 1.0}]
        assigned = assign_speakers_to_segments(segments)
        self.assertEqual(len(assigned), 1)

    def test_empty_segments_returns_empty(self) -> None:
        self.assertEqual(assign_speakers_to_segments([]), [])
        self.assertEqual(assign_speakers_to_segments([], diarization_turns=[{"speaker": "A", "start_time": 0, "end_time": 1}]), [])

    def test_best_speaker_for_segment_finds_max_coverage(self) -> None:
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 3.0},
            {"speaker": "B", "start_time": 2.0, "end_time": 4.0},
        ]
        speaker, coverage = _best_speaker_for_segment(
            {"start_time": 1.0, "end_time": 4.0}, turns
        )
        # A covers 1.0-3.0 (2s), B covers 2.0-4.0 (2s) -> tie, first wins
        self.assertIn(speaker, {"A", "B"})
        self.assertAlmostEqual(coverage, 2.0 / 3.0, places=4)

    def test_best_speaker_zero_duration_segment(self) -> None:
        speaker, coverage = _best_speaker_for_segment(
            {"start_time": 1.0, "end_time": 1.0},
            [{"speaker": "A", "start_time": 0.0, "end_time": 2.0}],
        )
        self.assertIsNone(speaker)
        self.assertEqual(coverage, 0.0)

    def test_best_speaker_no_overlap_returns_none(self) -> None:
        speaker, coverage = _best_speaker_for_segment(
            {"start_time": 10.0, "end_time": 12.0},
            [{"speaker": "A", "start_time": 0.0, "end_time": 1.0}],
        )
        self.assertIsNone(speaker)


if __name__ == "__main__":
    unittest.main()
