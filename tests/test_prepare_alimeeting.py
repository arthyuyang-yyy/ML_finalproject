"""Tests for the AliMeeting annotation converter (no audio / no real corpus).

Synthetic Praat TextGrids exercise the pure functions so the parser and the
ground-truth overlap-region logic are validated without downloading the dataset.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_alimeeting import (
    convert_meeting,
    overlap_regions_from_turns,
    parse_textgrid,
    prepare_root,
    textgrid_to_turns,
    validate_record,
)


def _record(turns, overlap_seconds=0.0):
    return {
        "meeting_id": "M",
        "num_speakers": len({t["speaker"] for t in turns}),
        "turns": turns,
        "overlap_regions": [],
        "overlap_seconds": overlap_seconds,
    }


def make_textgrid(tiers: dict[str, list[tuple[float, float, str]]], xmax: float) -> str:
    """Render a minimal standard Praat TextGrid from ``{speaker: [(xmin, xmax, text)]}``."""
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {xmax}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for i, (name, intervals) in enumerate(tiers.items(), start=1):
        lines += [
            f"    item [{i}]:",
            '        class = "IntervalTier"',
            f'        name = "{name}"',
            "        xmin = 0",
            f"        xmax = {xmax}",
            f"        intervals: size = {len(intervals)}",
        ]
        for j, (a, b, text) in enumerate(intervals, start=1):
            lines += [
                f"        intervals [{j}]:",
                f"            xmin = {a}",
                f"            xmax = {b}",
                f'            text = "{text}"',
            ]
    return "\n".join(lines)


class ParseTextGridTests(unittest.TestCase):
    def test_keeps_nonempty_intervals_per_speaker(self) -> None:
        text = make_textgrid(
            {
                "A": [(0.0, 2.0, "hello"), (2.0, 3.0, ""), (3.0, 4.0, "world")],
                "B": [(1.0, 1.5, "hi"), (3.5, 5.0, "bye")],
            },
            xmax=5.0,
        )
        tiers = parse_textgrid(text)
        self.assertEqual(set(tiers), {"A", "B"})
        self.assertEqual(tiers["A"], [(0.0, 2.0), (3.0, 4.0)])  # empty interval dropped
        self.assertEqual(tiers["B"], [(1.0, 1.5), (3.5, 5.0)])

    def test_tier_header_not_mistaken_for_interval(self) -> None:
        text = make_textgrid({"A": [(0.0, 1.0, "x")]}, xmax=9.0)
        tiers = parse_textgrid(text)
        # Only the real interval, never the tier's own xmin/xmax (0..9).
        self.assertEqual(tiers["A"], [(0.0, 1.0)])

    def test_ignores_non_interval_tiers(self) -> None:
        text = make_textgrid({"spk1": [(0.0, 1.0, "a")]}, xmax=1.0)
        text = text.replace('class = "IntervalTier"', 'class = "TextTier"', 1)
        self.assertEqual(parse_textgrid(text), {})


class TurnConversionTests(unittest.TestCase):
    def test_turns_are_sorted_and_formatted(self) -> None:
        text = make_textgrid(
            {"A": [(3.0, 4.0, "late")], "B": [(0.0, 1.0, "early")]},
            xmax=4.0,
        )
        turns = textgrid_to_turns(text, "meeting_001")
        self.assertEqual([t["speaker"] for t in turns], ["B", "A"])
        self.assertEqual(turns[0], {
            "meeting_id": "meeting_001",
            "speaker": "B",
            "start_time": 0.0,
            "end_time": 1.0,
        })


class OverlapRegionTests(unittest.TestCase):
    def test_two_speaker_overlap(self) -> None:
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 2.0},
            {"speaker": "B", "start_time": 1.0, "end_time": 1.5},
            {"speaker": "A", "start_time": 3.0, "end_time": 4.0},
            {"speaker": "B", "start_time": 3.5, "end_time": 5.0},
        ]
        self.assertEqual(overlap_regions_from_turns(turns), [(1.0, 1.5), (3.5, 4.0)])

    def test_touching_turns_are_not_overlap(self) -> None:
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 1.0},
            {"speaker": "B", "start_time": 1.0, "end_time": 2.0},
        ]
        self.assertEqual(overlap_regions_from_turns(turns), [])

    def test_nested_three_speaker_overlap_merges(self) -> None:
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 5.0},
            {"speaker": "B", "start_time": 1.0, "end_time": 4.0},
            {"speaker": "C", "start_time": 2.0, "end_time": 3.0},
        ]
        # Overlap (>=2 active) spans the whole [1, 4] window after merging.
        self.assertEqual(overlap_regions_from_turns(turns), [(1.0, 4.0)])

    def test_single_speaker_has_no_overlap(self) -> None:
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 1.0},
            {"speaker": "A", "start_time": 2.0, "end_time": 3.0},
        ]
        self.assertEqual(overlap_regions_from_turns(turns), [])

    def test_same_speaker_overlapping_turns_are_not_overlap(self) -> None:
        # Two concurrent turns from the SAME speaker must not count as multi-speaker overlap.
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 2.0},
            {"speaker": "A", "start_time": 1.0, "end_time": 3.0},
        ]
        self.assertEqual(overlap_regions_from_turns(turns), [])


class ConvertMeetingTests(unittest.TestCase):
    def test_record_summary_fields(self) -> None:
        text = make_textgrid(
            {
                "A": [(0.0, 2.0, "x")],
                "B": [(1.0, 1.5, "y")],
            },
            xmax=2.0,
        )
        record = convert_meeting(text, "meeting_007")
        self.assertEqual(record["meeting_id"], "meeting_007")
        self.assertEqual(record["num_speakers"], 2)
        self.assertEqual(len(record["turns"]), 2)
        self.assertEqual(record["overlap_regions"], [{"start_time": 1.0, "end_time": 1.5}])
        self.assertEqual(record["overlap_seconds"], 0.5)


class PrepareRootTests(unittest.TestCase):
    def test_duplicate_meeting_id_raises(self) -> None:
        # The same meeting id under two subdirs (e.g. far/ and near/) must error
        # instead of one silently overwriting the other.
        text = make_textgrid({"A": [(0.0, 1.0, "x")]}, xmax=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("far", "near"):
                (root / sub).mkdir()
                (root / sub / "R0001_M0001.TextGrid").write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_root(root, root / "out")


class ValidateRecordTests(unittest.TestCase):
    def test_well_formed_record_passes(self) -> None:
        turns = [
            {"speaker": "A", "start_time": 0.0, "end_time": 2.0},
            {"speaker": "B", "start_time": 1.0, "end_time": 3.0},
        ]
        validate_record(_record(turns, overlap_seconds=1.0))  # does not raise

    def test_empty_turns_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no speaker turns"):
            validate_record(_record([]))

    def test_non_positive_turn_duration_raises(self) -> None:
        turns = [{"speaker": "A", "start_time": 2.0, "end_time": 2.0}]
        with self.assertRaisesRegex(ValueError, "non-positive turn duration"):
            validate_record(_record(turns))

    def test_overlap_exceeding_total_speech_raises(self) -> None:
        turns = [{"speaker": "A", "start_time": 0.0, "end_time": 2.0}]
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_record(_record(turns, overlap_seconds=5.0))

    def test_convert_meeting_output_is_valid(self) -> None:
        text = make_textgrid({"A": [(0.0, 2.0, "x")], "B": [(1.0, 1.5, "y")]}, xmax=2.0)
        validate_record(convert_meeting(text, "M0"))  # does not raise


if __name__ == "__main__":
    unittest.main()
