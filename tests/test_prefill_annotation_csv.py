"""Tests for the annotation CSV pre-fill (口径 B clustering + overlap_type).

Synthetic TextGrids and utterance lists exercise the pure functions so the
clustering, overlap typing, and text-preserving parser are validated without the
AliMeeting corpus.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prefill_annotation_csv import (
    cluster_segments,
    meeting_id_of,
    overlap_type_of,
    parse_utterances,
    segments_to_csv_rows,
    speaker_token_of,
)


def _utt(speaker, start, end, text="x"):
    return {"meeting_id": "M", "speaker": speaker, "start_time": start, "end_time": end, "text": text}


class MeetingIdTests(unittest.TestCase):
    def test_strips_near_and_far_suffixes(self) -> None:
        self.assertEqual(meeting_id_of("R8001_M8004_N_SPK8013"), "R8001_M8004")
        self.assertEqual(meeting_id_of("R8001_M8004_MS801"), "R8001_M8004")
        self.assertEqual(meeting_id_of("R8001_M8004"), "R8001_M8004")

    def test_non_matching_stem_is_its_own_meeting(self) -> None:
        self.assertEqual(meeting_id_of("custom_meeting"), "custom_meeting")

    def test_speaker_token(self) -> None:
        self.assertEqual(speaker_token_of("R8001_M8004_N_SPK8013", "R8001_M8004"), "N_SPK8013")
        self.assertEqual(speaker_token_of("R8001_M8004", "R8001_M8004"), "")


class ParseUtterancesTests(unittest.TestCase):
    def _grid(self, tiers):
        # Minimal Praat blocks; parse_utterances only needs class/name/intervals.
        out = []
        for name, intervals in tiers.items():
            out.append(f'class = "IntervalTier"\n name = "{name}"')
            for a, b, t in intervals:
                out.append(f' xmin = {a} xmax = {b} text = "{t}"')
        return "\n".join(out)

    def test_keeps_text(self) -> None:
        grid = self._grid({"A": [(0.0, 1.0, "hello"), (1.0, 2.0, "")]})
        utts = parse_utterances(grid, "M", file_token="")
        self.assertEqual(len(utts), 1)  # empty interval dropped
        self.assertEqual(utts[0]["text"], "hello")
        self.assertEqual(utts[0]["speaker"], "A")

    def test_single_tier_uses_file_token_as_speaker(self) -> None:
        grid = self._grid({"speaker": [(0.0, 1.0, "hi")]})
        utts = parse_utterances(grid, "R1_M1", file_token="N_SPK42")
        self.assertEqual(utts[0]["speaker"], "N_SPK42")


class OverlapTypeTests(unittest.TestCase):
    def test_buckets(self) -> None:
        self.assertEqual(overlap_type_of(0.0, 0.5), "none")
        self.assertEqual(overlap_type_of(0.2, 0.5), "partial")
        self.assertEqual(overlap_type_of(0.5, 0.5), "full")
        self.assertEqual(overlap_type_of(0.9, 0.5), "full")


class ClusterSegmentsTests(unittest.TestCase):
    def test_non_overlapping_become_low_overlap_singles(self) -> None:
        segs = cluster_segments([_utt("A", 0.0, 1.0), _utt("B", 2.0, 3.0)])
        self.assertEqual(len(segs), 2)
        self.assertTrue(all(not s["is_overlap"] and s["overlap_type"] == "none" for s in segs))
        self.assertTrue(all(len(s["rows"]) == 1 for s in segs))

    def test_cross_speaker_overlap_merges_into_one_multi_candidate_segment(self) -> None:
        segs = cluster_segments([_utt("A", 0.0, 10.0, "aa"), _utt("B", 0.0, 10.0, "bb")])
        self.assertEqual(len(segs), 1)
        seg = segs[0]
        self.assertTrue(seg["is_overlap"])
        self.assertEqual(seg["overlap_type"], "full")  # 100% covered
        self.assertEqual({r["speaker"] for r in seg["rows"]}, {"A", "B"})

    def test_partial_coverage_is_partial(self) -> None:
        # A spans [0,10]; B overlaps only [8,10] -> 2/10 = 0.2 of the segment.
        segs = cluster_segments([_utt("A", 0.0, 10.0), _utt("B", 8.0, 10.0)], full_threshold=0.5)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["overlap_type"], "partial")

    def test_same_speaker_overlap_is_not_high_overlap(self) -> None:
        segs = cluster_segments([_utt("A", 0.0, 2.0), _utt("A", 1.0, 3.0)])
        self.assertTrue(all(not s["is_overlap"] for s in segs))

    def test_short_overlap_ignored_at_default_min_overlap(self) -> None:
        # Ratified口径 (issue #63): min_overlap_seconds defaults to 1.0, so a 0.5s
        # cross-speaker overlap is ignored and the segments stay low-overlap.
        utts = [_utt("A", 0.0, 2.0), _utt("B", 1.5, 2.0)]  # overlap [1.5,2.0] = 0.5s
        self.assertTrue(all(not s["is_overlap"] for s in cluster_segments(utts)))
        # but with the threshold lowered to 0 it would count as high-overlap
        self.assertTrue(any(s["is_overlap"] for s in cluster_segments(utts, min_overlap_seconds=0.0)))


class CsvRowTests(unittest.TestCase):
    def test_high_overlap_rows_share_segment_id_and_blank_semantics(self) -> None:
        segs = cluster_segments([_utt("A", 0.0, 10.0, "aa"), _utt("B", 0.0, 10.0, "bb")])
        rows = segments_to_csv_rows("R1_M1", segs)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["segment_id"], rows[1]["segment_id"])
        self.assertEqual(rows[0]["segment_id"], "R1_M1_seg0001")
        self.assertEqual(rows[0]["is_overlap"], "True")
        for row in rows:
            self.assertEqual(row["event_type"], "")  # semantic columns left for annotators
            self.assertEqual(row["content"], "")


if __name__ == "__main__":
    unittest.main()
