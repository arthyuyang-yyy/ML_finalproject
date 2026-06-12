"""Tests for overlap scoring and dual-path routing."""

import unittest
from unittest.mock import patch

import numpy as np

from src.overlap.detector import (
    DEFAULT_OVERLAP_THRESHOLD,
    _merge_regions,
    _overlap_fraction,
    detect_pyannote_overlap_regions,
    estimate_segment_overlap_scores,
)
from src.fallbacks.overlap import energy_overlap_proxy
from src.errors import BackendExecutionError, BackendUnavailableError
from src.overlap.router import route_segment


class OverlapScoringTests(unittest.TestCase):
    def test_scores_segments_by_overlap_coverage(self) -> None:
        samples = np.zeros(16000 * 20, dtype=np.float32)
        segments = [
            {"segment_id": "m1_seg_001", "start_time": 0.0, "end_time": 10.0},
            {"segment_id": "m1_seg_002", "start_time": 10.0, "end_time": 20.0},
        ]

        scored = estimate_segment_overlap_scores(
            samples,
            segments,
            overlap_regions=[(2.0, 7.0), (12.0, 14.0)],
        )

        self.assertEqual(scored[0]["overlap_score"], 0.5)
        self.assertEqual(scored[1]["overlap_score"], 0.2)
        self.assertEqual(scored[0]["overlap_detector"], "provided_regions")

    def test_energy_fallback_stays_below_routing_threshold(self) -> None:
        samples = np.ones(16000 * 3, dtype=np.float32) * 0.1
        scored = estimate_segment_overlap_scores(
            samples,
            [{"segment_id": "m1_seg_001", "start_time": 0.0, "end_time": 3.0}],
            sample_rate=16000,
        )

        self.assertLess(scored[0]["overlap_score"], DEFAULT_OVERLAP_THRESHOLD)
        self.assertEqual(scored[0]["overlap_detector"], "energy_fallback")

    def test_empty_segments_returns_empty(self) -> None:
        scored = estimate_segment_overlap_scores(np.zeros(16000, dtype=np.float32), [])
        self.assertEqual(scored, [])

    def test_overlap_fraction_zero_duration(self) -> None:
        segment = {"start_time": 1.0, "end_time": 1.0}
        self.assertEqual(_overlap_fraction(segment, [(0.0, 2.0)]), 0.0)

    def test_merge_regions_overlapping(self) -> None:
        merged = _merge_regions([(0.0, 2.0), (1.0, 3.0), (4.0, 5.0)])
        self.assertEqual(merged, [(0.0, 3.0), (4.0, 5.0)])

    def test_merge_regions_empty(self) -> None:
        self.assertEqual(_merge_regions([]), [])

    def test_energy_proxy_silence_returns_zero(self) -> None:
        self.assertEqual(energy_overlap_proxy(np.zeros(16000, dtype=np.float32), 16000), 0.0)

    def test_fuses_diarization_overlap_and_speaker_change(self) -> None:
        segments = [{"segment_id": "s1", "start_time": 0.0, "end_time": 10.0}]
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 8.0},
            {"speaker": "B", "start_time": 2.0, "end_time": 10.0},
        ]
        scored = estimate_segment_overlap_scores(
            np.zeros(160000, dtype=np.float32),
            segments,
            overlap_regions=[],
            diarization_turns=turns,
        )
        self.assertGreater(scored[0]["overlap_score"], 0.0)
        self.assertEqual(scored[0]["overlap_components"]["diarization_overlap"], 0.6)
        self.assertGreater(scored[0]["overlap_components"]["speaker_change"], 0.0)

    @patch("src.overlap.detector.load_pyannote_pipeline", side_effect=RuntimeError("model denied"))
    def test_configured_pyannote_model_failure_is_reported(self, _mocked_load) -> None:
        with self.assertRaisesRegex(BackendExecutionError, "OSD failed"):
            detect_pyannote_overlap_regions("missing.wav", auth_token="token")

    @patch("src.overlap.detector.load_pyannote_pipeline", side_effect=ImportError("missing"))
    def test_configured_pyannote_missing_dependency_is_reported(self, _mocked_load) -> None:
        with self.assertRaisesRegex(BackendUnavailableError, "pyannote.audio is unavailable"):
            detect_pyannote_overlap_regions("missing.wav", auth_token="token")


class RoutingThresholdTests(unittest.TestCase):
    def test_default_threshold_is_point_four(self) -> None:
        self.assertEqual(DEFAULT_OVERLAP_THRESHOLD, 0.4)
        self.assertEqual(route_segment(0.399), "low_overlap_cluster")
        self.assertEqual(route_segment(0.4), "high_overlap_candidate")

    def test_route_segment_rejects_out_of_range_score(self) -> None:
        with self.assertRaises(ValueError):
            route_segment(1.5)
        with self.assertRaises(ValueError):
            route_segment(-0.1)

    def test_route_segment_rejects_out_of_range_threshold(self) -> None:
        with self.assertRaises(ValueError):
            route_segment(0.5, threshold=1.5)

    def test_detect_overlap_segments_uses_point_four_threshold(self) -> None:
        samples = np.zeros(16000 * 10, dtype=np.float32)
        segments = [{"segment_id": "m1_seg_001", "start_time": 0.0, "end_time": 10.0}]
        scored = estimate_segment_overlap_scores(samples, segments, overlap_regions=[(0.0, 4.0)])
        high = [segment for segment in scored if route_segment(segment["overlap_score"]) == "high_overlap_candidate"]

        self.assertEqual(len(high), 1)


if __name__ == "__main__":
    unittest.main()
