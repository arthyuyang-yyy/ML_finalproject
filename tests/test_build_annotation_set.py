"""Tests for the annotation-set builder (CSV -> validated annotations.json)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_annotation_set as bas  # noqa: E402

HEADER = (
    "meeting_id,segment_id,start_time,end_time,speaker,text,is_overlap,"
    "overlap_type,topic,decision,action_item,event_type,content,owner,deadline,uncertainty_note"
)

# A well-formed mini meeting: one decision, one action item, one high-overlap
# segment carrying two speaker candidates on the same segment_id.
GOOD_CSV = "\n".join(
    [
        HEADER,
        "# comment line that must be skipped",
        "m1,m1_s1,0,4,SPEAKER_00,经过讨论决定采用方案A,False,none,,,,decision,采用方案A,,,",
        "m1,m1_s2,4,9,SPEAKER_01,李明负责明天前完成接口测试,False,none,,,,action_item,完成接口测试,SPEAKER_01,明天,",
        "m1,m1_s3,9,13,SPEAKER_00,我同意,True,full,,,,uncertainty,高重叠不确定,,,多人同时说话",
        "m1,m1_s3,9,13,SPEAKER_02,这个不行,True,full,,,,uncertainty,高重叠不确定,,,多人同时说话",
        "",
    ]
)


def _write(text: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "labels.csv"
    tmp.write_text(text, encoding="utf-8")
    return tmp


class BuildAnnotationSetTests(unittest.TestCase):
    def test_read_rows_skips_comments_and_blanks(self):
        rows = bas.read_rows(_write(GOOD_CSV))
        # 4 data rows (two of them share segment m1_s3); comments/blanks dropped.
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["meeting_id"] for r in rows}, {"m1"})

    def test_builds_validated_document(self):
        document = bas.build_meeting("m1", bas.read_rows(_write(GOOD_CSV)))
        # Three distinct segments (the two m1_s3 rows merge into one).
        self.assertEqual(len(document["evidence_segments"]), 3)
        by_id = {s["segment_id"]: s for s in document["evidence_segments"]}

        # Low-overlap segment keeps its transcript and speaker.
        self.assertEqual(by_id["m1_s1"]["processing_path"], bas.LOW_OVERLAP)
        self.assertEqual(by_id["m1_s1"]["speaker"], "SPEAKER_00")
        self.assertEqual(by_id["m1_s1"]["candidates"], [])

        # High-overlap segment is MIXED, empty text, two candidates.
        high = by_id["m1_s3"]
        self.assertEqual(high["processing_path"], bas.HIGH_OVERLAP)
        self.assertEqual(high["speaker"], "MIXED")
        self.assertEqual(high["text"], "")
        self.assertEqual(len(high["candidates"]), 2)

        # Gold events: decision, action_item (with owner+deadline), uncertainty.
        events = {e["event_type"]: e for e in document["gold_events"]}
        self.assertEqual(set(events), {"decision", "action_item", "uncertainty"})
        self.assertEqual(events["action_item"]["owner"], "SPEAKER_01")
        self.assertEqual(events["action_item"]["deadline"], "明天")
        self.assertEqual(events["uncertainty"]["evidence_ids"], ["m1_s3"])

    def test_high_overlap_label_forced_to_uncertainty(self):
        # Annotator wrongly labels a high-overlap row as a decision; builder forces uncertainty.
        csv_text = "\n".join(
            [
                HEADER,
                "m1,m1_s1,0,4,SPEAKER_00,经过讨论,False,none,,,,decision,采用方案A,,,",
                "m1,m1_s2,4,8,SPEAKER_00,我同意,True,full,,,,decision,不该是decision,,,",
            ]
        )
        document = bas.build_meeting("m1", bas.read_rows(_write(csv_text)))
        types = {e["event_type"] for e in document["gold_events"]}
        self.assertIn("uncertainty", types)
        self.assertNotIn("decision", {
            e["event_type"] for e in document["gold_events"] if e["evidence_ids"] == ["m1_s2"]
        })

    def test_action_item_unsupported_owner_raises(self):
        # owner is a speaker that does not appear in the cited evidence -> rejected by validator.
        csv_text = "\n".join(
            [
                HEADER,
                "m1,m1_s1,0,4,SPEAKER_00,安排任务,False,none,,,,action_item,做点事,SPEAKER_99,,",
            ]
        )
        with self.assertRaises(ValueError):
            bas.build_meeting("m1", bas.read_rows(_write(csv_text)))

    def test_build_all_separates_meetings(self):
        csv_text = "\n".join(
            [
                HEADER,
                "m1,m1_s1,0,4,SPEAKER_00,决定A,False,none,,,,decision,A,,,",
                "m2,m2_s1,0,4,SPEAKER_00,决定B,False,none,,,,decision,B,,,",
            ]
        )
        documents = bas.build_all(bas.read_rows(_write(csv_text)))
        self.assertEqual(set(documents), {"m1", "m2"})
        self.assertEqual(documents["m2"]["meeting_id"], "m2")


if __name__ == "__main__":
    unittest.main()
