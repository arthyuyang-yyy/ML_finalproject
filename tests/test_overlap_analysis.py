"""Tests for the offline overlap-score analysis pure functions (no audio/pyannote)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.core import HIGH_OVERLAP, LOW_OVERLAP
from experiments.overlap_threshold.analyze_scores import (
    flagged_rate_by_bucket,
    meeting_overlap_ratios,
    relabel,
    stratified_split,
)


def _win(meeting, score, frac):
    return {"meeting_id": meeting, "overlap_score": score, "gt_fraction": frac}


class RelabelTests(unittest.TestCase):
    def test_label_threshold_inclusive(self) -> None:
        scored = [_win("m", 0.2, 0.5), _win("m", 0.2, 0.49)]
        labeled = relabel(scored, gt_fraction=0.5)
        self.assertEqual(labeled[0]["gt_label"], HIGH_OVERLAP)
        self.assertEqual(labeled[1]["gt_label"], LOW_OVERLAP)


class MeetingRatioTests(unittest.TestCase):
    def test_mean_fraction_per_meeting(self) -> None:
        scored = [_win("a", 0, 0.2), _win("a", 0, 0.4), _win("b", 0, 0.6)]
        ratios = meeting_overlap_ratios(scored)
        self.assertAlmostEqual(ratios["a"], 0.3)
        self.assertAlmostEqual(ratios["b"], 0.6)


class StratifiedSplitTests(unittest.TestCase):
    def test_test_set_spans_overlap_range(self) -> None:
        # 4 meetings from low to high overlap; test gets one low-half + one high-half.
        ratios = {"low": 0.05, "mid": 0.2, "high": 0.45, "huge": 0.6}
        train, test = stratified_split(ratios, test_ratio=0.5, seed=0)
        self.assertEqual(len(test), 2)
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(set(train) | set(test), set(ratios))
        # one test meeting from the lower-overlap half, one from the upper half.
        lower_half = {"low", "mid"}
        self.assertTrue(any(t in lower_half for t in test))
        self.assertTrue(any(t not in lower_half for t in test))

    def test_deterministic(self) -> None:
        ratios = {m: i / 10 for i, m in enumerate(["a", "b", "c", "d", "e", "f"])}
        self.assertEqual(stratified_split(ratios, seed=3), stratified_split(ratios, seed=3))

    def test_single_meeting_empty_test(self) -> None:
        self.assertEqual(stratified_split({"only": 0.3}), (["only"], []))


class BucketTests(unittest.TestCase):
    def test_flagged_rate_per_bucket(self) -> None:
        scored = [
            _win("m", 0.4, 0.7),   # heavy, flagged at thr 0.1
            _win("m", 0.05, 0.7),  # heavy, not flagged at thr 0.1
            _win("m", 0.4, 0.0),   # none, flagged -> false alarm
        ]
        rows = {r["bucket"]: r for r in flagged_rate_by_bucket(scored, threshold=0.1)}
        self.assertEqual(rows["heavy (>=50%)"]["n"], 2)
        self.assertAlmostEqual(rows["heavy (>=50%)"]["flagged_high_rate"], 0.5)
        self.assertEqual(rows["none (0%)"]["n"], 1)
        self.assertAlmostEqual(rows["none (0%)"]["flagged_high_rate"], 1.0)
        self.assertEqual(rows["moderate (30-50%)"]["flagged_high_rate"], None)


if __name__ == "__main__":
    unittest.main()
