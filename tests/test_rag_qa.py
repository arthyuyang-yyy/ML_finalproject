"""Tests for evidence-constrained QA over retrieved episodic memory."""

import unittest

from src.llm.gemma_client import GemmaClient
from src.qa.answerer import answer_question
from src.qa.prompts import build_qa_prompt
from src.rag_qa import answer_question_with_evidence


def _episode(
    *,
    episode_id: str = "m1_ep_001",
    event_type: str = "action_item",
    content: str = "SPEAKER_01 will test WhisperX and pyannote alignment.",
    evidence_ids: list[str] | None = None,
    speakers: list[str] | None = None,
    start_time: float = 60.2,
    end_time: float = 68.4,
    confidence: str = "high",
    overlap_score: float = 0.08,
    uncertainty_note: str = "",
) -> dict:
    return {
        "episode_id": episode_id,
        "meeting_id": "meeting_001",
        "event_type": event_type,
        "topic": "ASR baseline",
        "content": content,
        "speakers": speakers if speakers is not None else ["SPEAKER_01"],
        "start_time": start_time,
        "end_time": end_time,
        "evidence_ids": evidence_ids if evidence_ids is not None else ["m1_seg_012"],
        "evidence_text": "我来测试 WhisperX 和 pyannote 的对齐。",
        "overlap_score": overlap_score,
        "confidence": confidence,
        "importance": 0.9,
        "audio_clip_paths": [],
        "uncertainty_note": uncertainty_note,
    }


def _valid_model_answer() -> dict:
    return {
        "answer": (
            "SPEAKER_01 负责测试 WhisperX 和 pyannote 的对齐。"
            "证据来自 m1_seg_012，时间范围是 60.2–68.4 秒。"
        ),
        "episode_ids": ["m1_ep_001"],
        "evidence_ids": ["m1_seg_012"],
        "citations": [{
            "episode_id": "m1_ep_001",
            "evidence_ids": ["m1_seg_012"],
            "start_time": 60.2,
            "end_time": 68.4,
        }],
        "speakers": ["SPEAKER_01"],
        "confidence": "high",
        "uncertainty_note": "",
        "insufficient_evidence": False,
    }


class AnswerWithEvidenceTests(unittest.TestCase):
    def test_empty_episodes_returns_explicit_insufficient_answer(self) -> None:
        result = answer_question_with_evidence("谁负责？", [])
        self.assertTrue(result["insufficient_evidence"])
        self.assertIn("无法确定", result["answer"])
        self.assertEqual(result["evidence_ids"], [])

    def test_deterministic_fallback_cites_evidence_and_timestamp(self) -> None:
        result = answer_question("谁负责测试 WhisperX？", [_episode()])
        self.assertIn("m1_seg_012", result["answer"])
        self.assertIn("60.200-68.400s", result["answer"])
        self.assertEqual(result["speaker"], "SPEAKER_01")
        self.assertEqual(result["confidence"], "high")

    def test_accepts_valid_gemma_json(self) -> None:
        client = GemmaClient(generator=lambda _: _valid_model_answer())
        result = answer_question("谁负责测试 WhisperX？", [_episode()], client=client)
        self.assertEqual(result["episode_ids"], ["m1_ep_001"])
        self.assertEqual(result["evidence_ids"], ["m1_seg_012"])
        self.assertFalse(result["insufficient_evidence"])

    def test_accepts_valid_gemma_json_string(self) -> None:
        client = GemmaClient(generator=lambda _: """```json
        {
          "answer": "SPEAKER_01 负责测试。证据 m1_seg_012，时间 60.2-68.4 秒。",
          "episode_ids": ["m1_ep_001"],
          "evidence_ids": ["m1_seg_012"],
          "citations": [{"episode_id": "m1_ep_001", "evidence_ids": ["m1_seg_012"], "start_time": 60.2, "end_time": 68.4}],
          "speakers": ["SPEAKER_01"],
          "confidence": "high",
          "uncertainty_note": "",
          "insufficient_evidence": false
        }
        ```""")
        result = answer_question("谁负责测试？", [_episode()], client=client)
        self.assertEqual(result["confidence"], "high")

    def test_invalid_evidence_id_triggers_safe_fallback(self) -> None:
        invalid = _valid_model_answer()
        invalid["evidence_ids"] = ["invented"]
        invalid["citations"][0]["evidence_ids"] = ["invented"]
        client = GemmaClient(generator=lambda _: invalid)
        result = answer_question("谁负责测试？", [_episode()], client=client)
        self.assertEqual(result["evidence_ids"], ["m1_seg_012"])
        self.assertNotIn("invented", result["answer"])

    def test_invalid_timestamp_triggers_safe_fallback(self) -> None:
        invalid = _valid_model_answer()
        invalid["citations"][0]["start_time"] = 0.0
        client = GemmaClient(generator=lambda _: invalid)
        result = answer_question("谁负责测试？", [_episode()], client=client)
        self.assertEqual(result["timestamp"], "60.200-68.400s")

    def test_backend_failure_triggers_safe_fallback(self) -> None:
        def fail(_: str) -> dict:
            raise ConnectionError("backend unavailable")

        result = answer_question("谁负责测试？", [_episode()], client=GemmaClient(generator=fail))
        self.assertEqual(result["evidence_ids"], ["m1_seg_012"])
        self.assertIn("60.200-68.400s", result["answer"])

    def test_invented_speaker_triggers_safe_fallback(self) -> None:
        invalid = _valid_model_answer()
        invalid["answer"] = invalid["answer"].replace("SPEAKER_01", "SPEAKER_99")
        invalid["speakers"] = ["SPEAKER_99"]
        client = GemmaClient(generator=lambda _: invalid)
        result = answer_question("谁负责测试？", [_episode()], client=client)
        self.assertEqual(result["speakers"], ["SPEAKER_01"])
        self.assertNotIn("SPEAKER_99", result["answer"])

    def test_high_overlap_fallback_explicitly_marks_uncertainty(self) -> None:
        episode = _episode(
            event_type="uncertainty",
            content="The segment may discuss Gemma usage.",
            evidence_ids=["m1_seg_013"],
            speakers=["MIXED"],
            confidence="low",
            overlap_score=0.82,
            uncertainty_note="High-overlap segment with conflicting candidates.",
        )
        result = answer_question("Gemma 的角色是什么？", [episode])
        self.assertEqual(result["confidence"], "low")
        self.assertIn("不确定", result["answer"])
        self.assertIn("m1_seg_013", result["answer"])

    def test_low_confidence_model_answer_requires_uncertainty(self) -> None:
        episode = _episode(confidence="low")
        invalid = _valid_model_answer()
        client = GemmaClient(generator=lambda _: invalid)
        result = answer_question("谁负责测试？", [episode], client=client)
        self.assertEqual(result["confidence"], "low")
        self.assertIn("不确定", result["answer"])

    def test_repair_attempt_can_recover_invalid_first_output(self) -> None:
        outputs = iter([{"answer": "unsupported"}, _valid_model_answer()])
        prompts: list[str] = []

        def generate(prompt: str) -> dict:
            prompts.append(prompt)
            return next(outputs)

        result = answer_question(
            "谁负责测试？",
            [_episode()],
            client=GemmaClient(generator=generate),
        )
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(len(prompts), 2)
        self.assertIn("failed validation", prompts[1])

    def test_prompt_contains_only_supplied_episode_and_rules(self) -> None:
        prompt = build_qa_prompt("谁负责测试？", [_episode()])
        self.assertIn("Use only the retrieved episodes", prompt)
        self.assertIn("m1_ep_001", prompt)
        self.assertIn("m1_seg_012", prompt)
        self.assertIn("Do not invent", prompt)

    def test_rejects_noncanonical_retrieved_episode(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing fields"):
            answer_question("test", [{"summary": "legacy shape"}])


if __name__ == "__main__":
    unittest.main()
