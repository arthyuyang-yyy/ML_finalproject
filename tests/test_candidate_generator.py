"""Tests for high-overlap candidate generation."""

import unittest

import numpy as np

from src.candidates.generator import _reject_hallucinated_candidate, generate_high_overlap_candidates


class FallbackCandidateTests(unittest.TestCase):
    def test_uses_supported_speaker_without_inventing_another(self) -> None:
        segment = {
            "segment_id": "m1_seg_003",
            "speaker": "SPEAKER_00",
            "text": "No, we have to cut it.",
            "asr_confidence": 0.61,
        }
        candidates = generate_high_overlap_candidates(segment)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["candidate_id"], "m1_seg_003_c1")
        self.assertEqual(candidates[1]["candidate_id"], "m1_seg_003_c2")
        self.assertEqual(candidates[0]["speaker"], "SPEAKER_00")
        self.assertEqual(candidates[1]["speaker"], "SPEAKER_00")
        self.assertIn("uncertainty_note", candidates[0])
        self.assertIn("decode_config", candidates[0])

    def test_uses_evidence_id_when_segment_id_is_missing(self) -> None:
        segment = {"evidence_id": "ev_001", "text": "hello", "asr_confidence": 0.5}
        candidates = generate_high_overlap_candidates(segment)
        self.assertEqual(candidates[0]["candidate_id"], "ev_001_c1")

    def test_fallback_text_when_text_is_empty(self) -> None:
        segment = {"segment_id": "s1", "text": ""}
        candidates = generate_high_overlap_candidates(segment)
        self.assertIn("pending ASR decode", candidates[0]["text"])

    def test_confidence_clamped_in_unit_range(self) -> None:
        segment = {"segment_id": "s1", "text": "test"}
        candidates = generate_high_overlap_candidates(segment)
        for c in candidates:
            self.assertGreaterEqual(c["confidence"], 0.0)
            self.assertLessEqual(c["confidence"], 1.0)

    def test_language_override_replaces_auto_language(self) -> None:
        segment = {"segment_id": "s1", "text": "test"}
        candidates = generate_high_overlap_candidates(segment, language="zh")
        self.assertEqual(candidates[0]["decode_config"]["language"], "zh")
        self.assertEqual(candidates[1]["decode_config"]["language"], "zh")

    def test_zh_high_overlap_rejects_short_english_hallucinations(self) -> None:
        self.assertTrue(_reject_hallucinated_candidate("Have a good one!", "zh"))
        self.assertTrue(_reject_hallucinated_candidate("Bye!", "zh"))
        self.assertFalse(_reject_hallucinated_candidate("这个目标客户群", "zh"))
        self.assertFalse(_reject_hallucinated_candidate("Have a good one!", "en"))

    def test_empty_samples_forces_fallback(self) -> None:
        segment = {"segment_id": "s1", "text": "test"}
        candidates = generate_high_overlap_candidates(segment, samples=np.array([], dtype=np.float32))
        self.assertEqual(len(candidates), 2)
        self.assertIn("fallback", candidates[0]["uncertainty_note"])

    def test_uses_diarization_speaker_hypotheses(self) -> None:
        candidates = generate_high_overlap_candidates(
            {"segment_id": "s1", "speaker": "MIXED", "text": ""},
            speaker_hypotheses=["ALICE", "BOB"],
        )
        self.assertEqual([candidate["speaker"] for candidate in candidates], ["ALICE", "BOB"])


if __name__ == "__main__":
    unittest.main()
