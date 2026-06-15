"""Tests for the rule-based event extraction baseline and its metric (Step 16a)."""

import unittest

from src.evaluation.core import evaluate_event_extraction
from src.events.rule_extractor import (
    classify_event_type,
    extract_deadline,
    extract_rule_based_events,
)


def _segment(segment_id, text, *, path="low_overlap_cluster", speaker="SPEAKER_00", **overrides):
    segment = {
        "meeting_id": "meeting_001",
        "segment_id": segment_id,
        "evidence_id": segment_id,
        "speaker": speaker,
        "start_time": 0.0,
        "end_time": 1.0,
        "text": text,
        "processing_path": path,
        "route_reason": "r",
        "overlap_score": 0.1,
        "asr_confidence": 0.9,
        "speaker_confidence": 0.9,
        "audio_clip_path": "",
        "source_audio_path": "",
        "language": "zh",
        "candidates": [],
        "uncertainty_note": "",
    }
    segment.update(overrides)
    return segment


class ClassifyEventTypeTests(unittest.TestCase):
    def test_decision_cues_zh_and_en(self) -> None:
        self.assertEqual(classify_event_type("我们决定采用方案A"), "decision")
        self.assertEqual(classify_event_type("We decided to ship it"), "decision")

    def test_action_item_cues(self) -> None:
        self.assertEqual(classify_event_type("请在周五前提交报告"), "action_item")
        self.assertEqual(classify_event_type("Please follow up with the client"), "action_item")

    def test_open_question_by_mark_or_cue(self) -> None:
        self.assertEqual(classify_event_type("下一阶段预算是多少？"), "open_question")
        self.assertEqual(classify_event_type("What is the timeline?"), "open_question")
        self.assertEqual(classify_event_type("这件事还有待确认"), "open_question")

    def test_default_is_speaker_stance(self) -> None:
        self.assertEqual(classify_event_type("我个人比较认同这个方向"), "speaker_stance")

    def test_decision_takes_priority_over_action(self) -> None:
        # Contains both a decision cue and an action cue; decision wins.
        self.assertEqual(classify_event_type("我们决定让李明负责"), "decision")


class ExtractDeadlineTests(unittest.TestCase):
    def test_zh_relative_and_absolute(self) -> None:
        self.assertEqual(extract_deadline("请在明天之前完成"), "明天")
        self.assertEqual(extract_deadline("下周五交付"), "下周五")
        self.assertEqual(extract_deadline("截止 2026-07-03"), "2026-07-03")

    def test_en_relative(self) -> None:
        self.assertEqual(extract_deadline("Finish it by Friday please").lower(), "by friday")
        self.assertEqual(extract_deadline("ship next week"), "next week")

    def test_none_when_absent(self) -> None:
        self.assertIsNone(extract_deadline("没有任何时间信息"))


class ExtractRuleBasedEventsTests(unittest.TestCase):
    def test_emits_one_event_per_usable_segment_and_validates(self) -> None:
        evidence = [
            _segment("m1_seg_001", "我们决定采用方案A", speaker="SPEAKER_00"),
            _segment("m1_seg_002", "李明负责在明天之前完成测试", speaker="SPEAKER_01"),
            _segment("m1_seg_003", "", path="high_overlap_candidate", speaker="MIXED",
                     overlap_score=0.82, asr_confidence=0.0, speaker_confidence=0.35,
                     uncertainty_note="High-overlap segment; speaker attribution is uncertain.",
                     candidates=[{"candidate_id": "m1_seg_003_c1", "speaker": "SPEAKER_00",
                                  "text": "我同意", "confidence": 0.4, "uncertainty_note": "x"}]),
        ]

        document = extract_rule_based_events(evidence)
        events = document["events"]
        self.assertEqual([e["event_type"] for e in events],
                         ["decision", "action_item", "uncertainty"])

        action = events[1]
        self.assertEqual(action["owner"], "SPEAKER_01")
        self.assertEqual(action["deadline"], "明天")
        self.assertEqual(action["evidence_ids"], ["m1_seg_002"])

        uncertainty = events[2]
        self.assertEqual(uncertainty["confidence"], "low")
        self.assertEqual(uncertainty["speakers"], [])

    def test_blank_low_overlap_segment_is_skipped(self) -> None:
        evidence = [_segment("m1_seg_001", "   ")]
        self.assertEqual(extract_rule_based_events(evidence)["events"], [])

    def test_empty_input_returns_empty_document(self) -> None:
        self.assertEqual(extract_rule_based_events([]),
                         {"meeting_id": "", "meeting_summary": "", "events": []})


class EvaluateEventExtractionTests(unittest.TestCase):
    def test_per_type_and_overall_scores(self) -> None:
        predicted = [
            {"event_type": "decision", "evidence_ids": ["a"]},
            {"event_type": "action_item", "evidence_ids": ["b"]},
            {"event_type": "speaker_stance", "evidence_ids": ["c"]},
        ]
        gold = [
            {"event_type": "decision", "evidence_ids": ["a"]},
            {"event_type": "action_item", "evidence_ids": ["b"]},
            {"event_type": "open_question", "evidence_ids": ["c"]},
        ]

        result = evaluate_event_extraction(predicted, gold)
        self.assertEqual(result["matched"], 2)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["per_type"]["decision"]["f1"], 1.0)
        self.assertEqual(result["per_type"]["speaker_stance"]["precision"], 0.0)
        self.assertEqual(result["per_type"]["open_question"]["recall"], 0.0)

    def test_same_type_without_shared_evidence_is_not_matched(self) -> None:
        predicted = [{"event_type": "decision", "evidence_ids": ["x"]}]
        gold = [{"event_type": "decision", "evidence_ids": ["y"]}]
        self.assertEqual(evaluate_event_extraction(predicted, gold)["matched"], 0)

    def test_empty_inputs(self) -> None:
        result = evaluate_event_extraction([], [])
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
