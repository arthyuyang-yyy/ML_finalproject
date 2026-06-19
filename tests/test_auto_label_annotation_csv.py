"""Tests for semantic auto-labeling of pre-filled annotation CSV rows."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import auto_label_annotation_csv as labeler  # noqa: E402
import build_annotation_set as bas  # noqa: E402


class AutoLabelAnnotationCsvTests(unittest.TestCase):
    def test_high_overlap_forces_uncertainty(self) -> None:
        row = {
            "speaker": "N_SPK1",
            "text": "我同意",
            "is_overlap": "True",
            "overlap_type": "partial",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "uncertainty")
        self.assertEqual(labeled["content"], labeler.UNCERTAINTY_CONTENT)
        self.assertEqual(labeled["owner"], "")
        self.assertTrue(labeled["uncertainty_note"])

    def test_first_person_action_uses_current_speaker_as_owner(self) -> None:
        row = {
            "speaker": "N_SPK2",
            "text": "我来负责明天提交测试报告",
            "is_overlap": "False",
            "overlap_type": "none",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "action_item")
        self.assertEqual(labeled["owner"], "N_SPK2")
        self.assertEqual(labeled["deadline"], "明天")

    def test_named_assignee_without_mapping_is_uncertain(self) -> None:
        row = {
            "speaker": "N_SPK3",
            "text": "李明负责下周五提交测试报告",
            "is_overlap": "False",
            "overlap_type": "none",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "action_item")
        self.assertEqual(labeled["owner"], "uncertain")

    def test_speaker_id_prefix_does_not_assign_owner(self) -> None:
        row = {
            "speaker": "N_SPK1",
            "text": "请 N_SPK10 明天提交报告",
            "is_overlap": "False",
            "overlap_type": "none",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "action_item")
        self.assertEqual(labeled["owner"], "uncertain")
        self.assertEqual(labeled["deadline"], "明天")

    def test_exact_current_speaker_id_can_assign_owner(self) -> None:
        row = {
            "speaker": "N_SPK1",
            "text": "N_SPK1 负责明天提交报告",
            "is_overlap": "False",
            "overlap_type": "none",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "action_item")
        self.assertEqual(labeled["owner"], "N_SPK1")

    def test_action_without_clear_owner_is_uncertain(self) -> None:
        row = {
            "speaker": "N_SPK4",
            "text": "需要明天提交测试报告",
            "is_overlap": "False",
            "overlap_type": "none",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "action_item")
        self.assertEqual(labeled["owner"], "uncertain")

    def test_first_person_action_with_empty_speaker_is_uncertain(self) -> None:
        row = {
            "speaker": "",
            "text": "我来负责明天提交测试报告",
            "is_overlap": "False",
            "overlap_type": "none",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "action_item")
        self.assertEqual(labeled["owner"], "uncertain")
        self.assertEqual(labeled["deadline"], "明天")

    def test_deadline_patterns_cover_zh_and_en_and_take_first(self) -> None:
        self.assertEqual(labeler.extract_deadline("下个月底前提交"), "下个月底")
        self.assertEqual(labeler.extract_deadline("please deliver by next monday"), "by next monday")
        self.assertEqual(labeler.extract_deadline("明天或者下周五提交"), "明天")

    def test_empty_csv_roundtrip_writes_header_only(self) -> None:
        header = (
            "meeting_id,segment_id,start_time,end_time,speaker,text,is_overlap,"
            "overlap_type,topic,decision,action_item,event_type,content,owner,deadline,uncertainty_note\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "empty.csv"
            dst = Path(tmp) / "out.csv"
            src.write_text(header, encoding="utf-8-sig")

            total, counts = labeler.label_csv(src, dst)

            self.assertEqual(total, 0)
            self.assertEqual(counts, {})
            with dst.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_pure_acknowledgement_is_not_decision(self) -> None:
        row = {
            "speaker": "N_SPK5",
            "text": "同意。",
            "is_overlap": "False",
            "overlap_type": "none",
        }

        labeled = labeler.label_row(row)

        self.assertEqual(labeled["event_type"], "speaker_stance")
        self.assertEqual(labeled["decision"], "")

    def test_csv_roundtrip_writes_labeled_rows(self) -> None:
        header = (
            "meeting_id,segment_id,start_time,end_time,speaker,text,is_overlap,"
            "overlap_type,topic,decision,action_item,event_type,content,owner,deadline,uncertainty_note\n"
        )
        body = "m1,s1,0,1,N_SPK1,我们决定采用方案A,False,none,,,,,,,,\n"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.csv"
            dst = Path(tmp) / "out.csv"
            src.write_text(header + body, encoding="utf-8-sig")

            total, counts = labeler.label_csv(src, dst)

            self.assertEqual(total, 1)
            self.assertEqual(counts["decision"], 1)
            with dst.open(encoding="utf-8-sig", newline="") as handle:
                labeled = next(csv.DictReader(handle))
            self.assertEqual(labeled["event_type"], "decision")
            self.assertEqual(labeled["decision"], "")
            self.assertEqual(labeled["topic"], "")

    def test_labeled_csv_can_feed_annotation_set_builder(self) -> None:
        header = (
            "meeting_id,segment_id,start_time,end_time,speaker,text,is_overlap,"
            "overlap_type,topic,decision,action_item,event_type,content,owner,deadline,uncertainty_note\n"
        )
        body = "\n".join(
            [
                "m1,s1,0,1,SPEAKER_00,我们决定采用方案A,False,none,,,,,,,,",
                "m1,s2,1,2,SPEAKER_01,我来负责明天提交测试报告,False,none,,,,,,,,",
                "m1,s3,2,3,SPEAKER_00,我同意,True,full,,,,,,,,",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.csv"
            dst = Path(tmp) / "out.csv"
            src.write_text(header + body, encoding="utf-8-sig")

            labeler.label_csv(src, dst)
            document = bas.build_meeting("m1", bas.read_rows(dst))

            self.assertEqual(len(document["evidence_segments"]), 3)
            event_types = {event["event_type"] for event in document["gold_events"]}
            self.assertEqual(event_types, {"decision", "action_item", "uncertainty"})


if __name__ == "__main__":
    unittest.main()
