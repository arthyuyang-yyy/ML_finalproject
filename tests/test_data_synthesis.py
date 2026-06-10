"""Tests for synthetic overlapping-speech data generation.

Run with::

    python -m unittest tests.test_data_synthesis
"""

import unittest

import numpy as np

from src.data_synthesis import (
    ANNOTATION_COLUMNS,
    mix_two_speakers,
    overlap_intervals,
    overlap_ratio,
    scale_to_snr,
    signal_power,
    to_annotation_rows,
    total_overlap_duration,
)

SAMPLE_RATE = 16000


def _tone(duration_s: float, freq: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class SnrScalingTests(unittest.TestCase):
    def test_equal_power_at_zero_db(self) -> None:
        a = _tone(1.0, amplitude=0.5)
        b = _tone(1.0, freq=330.0, amplitude=0.1)
        scaled = scale_to_snr(a, b, snr_db=0.0)
        self.assertAlmostEqual(signal_power(scaled), signal_power(a), places=4)

    def test_positive_db_makes_signal_quieter(self) -> None:
        a = _tone(1.0, amplitude=0.5)
        b = _tone(1.0, freq=330.0, amplitude=0.5)
        scaled = scale_to_snr(a, b, snr_db=10.0)  # reference 10 dB louder
        self.assertAlmostEqual(signal_power(a) / signal_power(scaled), 10.0, places=2)

    def test_silent_signal_unchanged(self) -> None:
        a = _tone(1.0)
        b = np.zeros(SAMPLE_RATE, dtype=np.float32)
        np.testing.assert_array_equal(scale_to_snr(a, b, 0.0), b)

    def test_signal_power_empty_array(self) -> None:
        self.assertEqual(signal_power(np.array([], dtype=np.float32)), 0.0)


class MixTwoSpeakersTests(unittest.TestCase):
    def test_mixture_length_accounts_for_overlap(self) -> None:
        a = _tone(2.0)
        b = _tone(2.0, freq=330.0)
        mixture, _ = mix_two_speakers(a, b, overlap_s=0.5, sample_rate=SAMPLE_RATE)
        expected = a.size + b.size - int(round(0.5 * SAMPLE_RATE))
        self.assertEqual(mixture.size, expected)

    def test_zero_overlap_is_back_to_back(self) -> None:
        a = _tone(1.0)
        b = _tone(1.0, freq=330.0)
        mixture, ann = mix_two_speakers(a, b, overlap_s=0.0, sample_rate=SAMPLE_RATE)
        self.assertEqual(mixture.size, a.size + b.size)
        self.assertEqual(ann["overlap_duration"], 0.0)
        self.assertEqual(ann["overlap_regions"], [])

    def test_annotation_reports_overlap(self) -> None:
        a = _tone(2.0)
        b = _tone(2.0, freq=330.0)
        _, ann = mix_two_speakers(a, b, overlap_s=0.7, sample_rate=SAMPLE_RATE)
        self.assertAlmostEqual(ann["overlap_duration"], 0.7, delta=0.01)
        self.assertEqual(len(ann["segments"]), 2)
        self.assertTrue(any(s["is_overlap"] for s in ann["segments"]))
        self.assertEqual(len(ann["overlap_regions"]), 1)

    def test_negative_overlap_raises(self) -> None:
        a, b = _tone(1.0), _tone(1.0)
        with self.assertRaises(ValueError):
            mix_two_speakers(a, b, overlap_s=-0.1)


class OverlapGeometryTests(unittest.TestCase):
    def test_overlap_intervals_detects_shared_region(self) -> None:
        segments = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 1.5, "end": 3.0},
        ]
        self.assertEqual(overlap_intervals(segments), [(1.5, 2.0)])

    def test_no_overlap_when_disjoint(self) -> None:
        segments = [
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "B", "start": 1.0, "end": 2.0},
        ]
        self.assertEqual(overlap_intervals(segments), [])
        self.assertEqual(total_overlap_duration(segments), 0.0)

    def test_overlap_ratio(self) -> None:
        segments = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 1.0, "end": 3.0},
        ]
        self.assertAlmostEqual(overlap_ratio(segments, duration_s=3.0), 1.0 / 3.0, places=3)

    def test_overlap_ratio_zero_duration(self) -> None:
        segments = [{"speaker": "A", "start": 0.0, "end": 1.0}]
        self.assertEqual(overlap_ratio(segments, duration_s=0.0), 0.0)


class AnnotationRowsTests(unittest.TestCase):
    def test_rows_match_template_columns(self) -> None:
        a = _tone(1.0)
        b = _tone(1.0, freq=330.0)
        _, ann = mix_two_speakers(a, b, overlap_s=0.3, sample_rate=SAMPLE_RATE)
        rows = to_annotation_rows(ann)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(set(row), set(ANNOTATION_COLUMNS))

    def test_to_annotation_rows_empty_segments(self) -> None:
        from src.data_synthesis import build_annotation
        ann = build_annotation([], SAMPLE_RATE, 0, "meeting")
        rows = to_annotation_rows(ann)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
