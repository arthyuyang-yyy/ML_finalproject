"""Tests for the overlap threshold-calibration pure functions (no audio / pyannote).

The labeling, meeting-level split, threshold sweep, and evaluation logic are
exercised on synthetic scored segments so the calibration is validated without
the dataset or the optional backends.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.core import HIGH_OVERLAP, LOW_OVERLAP
from experiments.overlap_threshold.run_calibration import (
    calibrate_threshold,
    default_thresholds,
    evaluate_at_threshold,
    label_segment,
    segment_overlap_fraction,
    split_meetings,
)


class OverlapFractionTests(unittest.TestCase):
    def test_fraction_of_segment_covered(self) -> None:
        segment = {"start_time": 0.0, "end_time": 10.0}
        regions = [(2.0, 4.0), (6.0, 7.0)]  # 2.0 + 1.0 covered
        self.assertAlmostEqual(segment_overlap_fraction(segment, regions), 0.3)

    def test_no_regions_is_zero(self) -> None:
        self.assertEqual(segment_overlap_fraction({"start_time": 0.0, "end_time": 5.0}, []), 0.0)

    def test_zero_duration_is_zero(self) -> None:
        self.assertEqual(segment_overlap_fraction({"start_time": 3.0, "end_time": 3.0}, [(0.0, 9.0)]), 0.0)

    def test_coverage_clamped_to_one(self) -> None:
        segment = {"start_time": 0.0, "end_time": 2.0}
        self.assertEqual(segment_overlap_fraction(segment, [(-1.0, 5.0)]), 1.0)


class LabelSegmentTests(unittest.TestCase):
    def test_threshold_is_inclusive(self) -> None:
        segment = {"start_time": 0.0, "end_time": 10.0}
        regions = [(0.0, 3.0)]  # fraction 0.3
        self.assertEqual(label_segment(segment, regions, gt_fraction=0.3), HIGH_OVERLAP)
        self.assertEqual(label_segment(segment, regions, gt_fraction=0.31), LOW_OVERLAP)


class SplitMeetingsTests(unittest.TestCase):
    def test_split_is_disjoint_and_covers_all(self) -> None:
        ids = [f"m{i}" for i in range(10)]
        train, test = split_meetings(ids, test_ratio=0.3, seed=0)
        self.assertEqual(len(test), 3)
        self.assertEqual(len(train), 7)
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(set(train) | set(test), set(ids))

    def test_deterministic_with_seed(self) -> None:
        ids = [f"m{i}" for i in range(8)]
        self.assertEqual(split_meetings(ids, seed=42), split_meetings(ids, seed=42))

    def test_single_meeting_has_empty_test(self) -> None:
        self.assertEqual(split_meetings(["only"], test_ratio=0.3), (["only"], []))

    def test_keeps_at_least_one_train_meeting(self) -> None:
        train, test = split_meetings(["a", "b"], test_ratio=0.9)
        self.assertTrue(train)
        self.assertTrue(test)


class ThresholdSweepTests(unittest.TestCase):
    def test_default_thresholds_span_open_interval(self) -> None:
        thresholds = default_thresholds(step=0.05)
        self.assertEqual(thresholds[0], 0.05)
        self.assertEqual(thresholds[-1], 0.95)
        self.assertNotIn(0.0, thresholds)
        self.assertNotIn(1.0, thresholds)

    def test_evaluate_at_threshold_matches_labels(self) -> None:
        labeled = [
            {"overlap_score": 0.8, "gt_label": HIGH_OVERLAP},
            {"overlap_score": 0.1, "gt_label": LOW_OVERLAP},
        ]
        metrics = evaluate_at_threshold(labeled, 0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_calibrate_picks_separating_threshold(self) -> None:
        labeled = (
            [{"meeting_id": "m1", "overlap_score": 0.8, "gt_label": HIGH_OVERLAP}] * 3
            + [{"meeting_id": "m2", "overlap_score": 0.1, "gt_label": LOW_OVERLAP}] * 3
        )
        result = calibrate_threshold(labeled)
        self.assertEqual(result["best_metrics"]["f1"], 1.0)
        self.assertGreater(result["best_threshold"], 0.1)
        self.assertLessEqual(result["best_threshold"], 0.8)

    def test_calibrate_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_threshold([])


if __name__ == "__main__":
    unittest.main()
