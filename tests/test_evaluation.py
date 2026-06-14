"""Tests for objective evaluation metrics.

Run with::

    python -m unittest tests.test_evaluation
"""

import unittest

from src.evaluation import (
    character_error_rate,
    edit_distance,
    evaluate_candidate_usefulness,
    evaluate_evidence_support,
    evaluate_overlap_routing,
    evaluate_uncertainty_preservation,
    speaker_attribution_accuracy,
    word_error_rate,
)


class EditDistanceTests(unittest.TestCase):
    def test_identical_sequences(self) -> None:
        result = edit_distance(list("abc"), list("abc"))
        self.assertEqual(result["distance"], 0)

    def test_both_empty_sequences(self) -> None:
        result = edit_distance([], [])
        self.assertEqual(result["distance"], 0)
        self.assertEqual(result["reference_length"], 0)

    def test_counts_substitution_deletion_insertion(self) -> None:
        # ref: a b c d ; hyp: a x c -> 1 sub (b->x), 1 del (d)
        result = edit_distance(["a", "b", "c", "d"], ["a", "x", "c"])
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["insertions"], 0)
        self.assertEqual(result["distance"], 2)


class ErrorRateTests(unittest.TestCase):
    def test_perfect_wer_is_zero(self) -> None:
        self.assertEqual(word_error_rate("the cat sat", "the cat sat")["error_rate"], 0.0)

    def test_wer_one_substitution(self) -> None:
        result = word_error_rate("the cat sat", "the dog sat")
        self.assertAlmostEqual(result["error_rate"], 1 / 3, places=4)
        self.assertEqual(result["substitutions"], 1)

    def test_cer_ignores_whitespace_and_counts_chars(self) -> None:
        result = character_error_rate("会 议 纪要", "会议要")  # ref 4 chars, drop 1 -> 1 del
        self.assertEqual(result["reference_length"], 4)
        self.assertEqual(result["deletions"], 1)
        self.assertAlmostEqual(result["error_rate"], 0.25, places=4)

    def test_empty_reference_with_insertions(self) -> None:
        result = word_error_rate("", "extra words")
        self.assertEqual(result["error_rate"], 1.0)

    def test_empty_hypothesis_with_deletions(self) -> None:
        result = word_error_rate("hello world", "")
        self.assertAlmostEqual(result["error_rate"], 1.0)


class OverlapRoutingTests(unittest.TestCase):
    def test_perfect_routing(self) -> None:
        preds = ["high_overlap_candidate", "low_overlap_cluster"]
        result = evaluate_overlap_routing(preds, preds)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_precision_recall_f1(self) -> None:
        refs = ["high_overlap_candidate", "high_overlap_candidate", "low_overlap_cluster"]
        preds = ["high_overlap_candidate", "low_overlap_cluster", "high_overlap_candidate"]
        result = evaluate_overlap_routing(preds, refs)
        self.assertAlmostEqual(result["precision"], 0.5, places=4)  # tp=1, fp=1
        self.assertAlmostEqual(result["recall"], 0.5, places=4)     # tp=1, fn=1
        self.assertAlmostEqual(result["f1"], 0.5, places=4)
        self.assertAlmostEqual(result["accuracy"], 1 / 3, places=4)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_overlap_routing(["low_overlap_cluster"], [])

    def test_invalid_label_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_overlap_routing(["nope"], ["low_overlap_cluster"])

    def test_empty_predictions_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_overlap_routing([], ["low_overlap_cluster"])


class SpeakerAttributionTests(unittest.TestCase):
    def test_label_permutation_is_resolved(self) -> None:
        # Hypothesis uses swapped names but the partition is perfect.
        reference = ["A", "A", "B", "B"]
        hypothesis = ["spk1", "spk1", "spk0", "spk0"]
        self.assertEqual(speaker_attribution_accuracy(reference, hypothesis)["accuracy"], 1.0)

    def test_partial_accuracy(self) -> None:
        reference = ["A", "A", "B", "B"]
        hypothesis = ["A", "A", "A", "B"]  # best mapping gets 3/4
        self.assertAlmostEqual(speaker_attribution_accuracy(reference, hypothesis)["accuracy"], 0.75, places=4)

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            speaker_attribution_accuracy(["A"], ["A", "B"])

    def test_empty_lists_raises(self) -> None:
        with self.assertRaises(ValueError):
            speaker_attribution_accuracy([], [])

    def test_five_speaker_permutation(self) -> None:
        refs = ["A", "B", "C", "D", "E"]
        hyps = ["E", "D", "C", "B", "A"]  # perfect permutation exists
        self.assertEqual(speaker_attribution_accuracy(refs, hyps)["accuracy"], 1.0)


class EvidenceSupportTests(unittest.TestCase):
    def test_perfect_support(self) -> None:
        preds = [{"evidence_ids": ["e1", "e2"]}, {"evidence_ids": ["e3"]}]
        refs = [{"evidence_ids": ["e1", "e2"]}, {"evidence_ids": ["e3"]}]
        result = evaluate_evidence_support(preds, refs)
        self.assertEqual(result["evidence_precision"], 1.0)
        self.assertEqual(result["evidence_recall"], 1.0)
        self.assertEqual(result["evidence_f1"], 1.0)
        self.assertEqual(result["evidence_hit_rate"], 1.0)
        self.assertEqual(result["unsupported_claim_rate"], 0.0)
        self.assertEqual(result["hallucination_rate"], 0.0)
        self.assertEqual(result["support"], 2)
        self.assertEqual(result["answerable"], 2)
        self.assertEqual(result["claims"], 2)

    def test_partial_precision_and_recall(self) -> None:
        # Cites one correct (e1) and one extra (x9); gold is e1 and e2.
        preds = [{"evidence_ids": ["e1", "x9"]}]
        refs = [{"evidence_ids": ["e1", "e2"]}]
        result = evaluate_evidence_support(preds, refs)
        self.assertAlmostEqual(result["evidence_precision"], 0.5)
        self.assertAlmostEqual(result["evidence_recall"], 0.5)
        self.assertAlmostEqual(result["evidence_f1"], 0.5)
        self.assertEqual(result["evidence_hit_rate"], 1.0)
        self.assertEqual(result["unsupported_claim_rate"], 0.0)

    def test_explicit_source_separates_hallucination_from_wrong_citation(self) -> None:
        # Item 0 cites a real but wrong segment; item 1 cites a nonexistent one.
        preds = [{"evidence_ids": ["e2"]}, {"evidence_ids": ["e_fake"]}]
        refs = [{"evidence_ids": ["e1"]}, {"evidence_ids": ["e1"]}]
        result = evaluate_evidence_support(preds, refs, source_evidence_ids={"e1", "e2", "e3"})
        self.assertEqual(result["unsupported_claim_rate"], 1.0)
        self.assertEqual(result["hallucination_rate"], 0.5)

    def test_correct_abstention_is_not_a_claim(self) -> None:
        preds = [{"insufficient_evidence": True, "evidence_ids": []}]
        refs = [{"evidence_ids": []}]
        result = evaluate_evidence_support(preds, refs)
        self.assertEqual(result["claims"], 0)
        self.assertEqual(result["answerable"], 0)
        self.assertEqual(result["unsupported_claim_rate"], 0.0)
        self.assertEqual(result["hallucination_rate"], 0.0)
        self.assertEqual(result["calibration_samples"], 0)

    def test_confidence_calibration_error(self) -> None:
        preds = [
            {"evidence_ids": ["e1"], "confidence": "high"},
            {"evidence_ids": ["e1"], "confidence": "high"},
            {"evidence_ids": ["x"], "confidence": "high"},
            {"evidence_ids": ["e1"], "confidence": "low"},
        ]
        refs = [
            {"evidence_ids": ["e1"]},
            {"evidence_ids": ["e1"]},
            {"evidence_ids": ["e2"]},
            {"evidence_ids": ["e1"]},
        ]
        result = evaluate_evidence_support(preds, refs)
        # high bin (0.9): acc 2/3, gap 0.2333, n=3; low bin (0.3): acc 1.0, gap 0.7, n=1
        self.assertEqual(result["calibration_samples"], 4)
        self.assertAlmostEqual(result["calibration_error"], 0.35)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_evidence_support([{"evidence_ids": []}], [])

    def test_empty_predictions_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_evidence_support([], [])

    def test_content_support_perfect(self) -> None:
        """ID correct and claim text is semantically close to evidence text."""
        text_map = {"e1": "the budget was approved by the entire team"}
        preds = [{"evidence_ids": ["e1"], "text": "the budget is approved"}]
        refs = [{"evidence_ids": ["e1"], "text": "the budget is approved"}]
        result = evaluate_evidence_support(preds, refs, evidence_text_map=text_map)
        self.assertEqual(result["evidence_precision"], 1.0)
        self.assertGreater(result["content_precision"], 0.0)
        self.assertGreater(result["content_hit_rate"], 0.0)
        self.assertEqual(result["content_unsupported_rate"], 0.0)

    def test_content_support_id_match_but_content_mismatch(self) -> None:
        """ID is correct but claim text is unrelated to evidence text."""
        text_map = {"e1": "the budget was approved by the entire team"}
        preds = [{"evidence_ids": ["e1"], "text": "we will ship next friday"}]
        refs = [{"evidence_ids": ["e1"], "text": "we will ship next friday"}]
        result = evaluate_evidence_support(preds, refs, evidence_text_map=text_map)
        # ID-level should be perfect
        self.assertEqual(result["evidence_precision"], 1.0)
        self.assertEqual(result["evidence_hit_rate"], 1.0)
        # Content-level should drop because texts are unrelated
        self.assertEqual(result["content_precision"], 0.0)
        self.assertEqual(result["content_hit_rate"], 0.0)
        self.assertEqual(result["content_unsupported_rate"], 1.0)

    def test_content_support_backward_compatible(self) -> None:
        """Without evidence_text_map, no content keys appear and old metrics are unchanged."""
        preds = [{"evidence_ids": ["e1", "e2"]}, {"evidence_ids": ["e3"]}]
        refs = [{"evidence_ids": ["e1", "e2"]}, {"evidence_ids": ["e3"]}]
        result = evaluate_evidence_support(preds, refs)
        self.assertEqual(result["evidence_precision"], 1.0)
        for key in result:
            self.assertFalse(key.startswith("content_"), f"unexpected content key: {key}")

    def test_content_support_missing_text_field(self) -> None:
        """When prediction text is missing, content metrics are zero (no fallback to ref)."""
        text_map = {"e1": "some evidence text"}
        preds = [{"evidence_ids": ["e1"]}]  # no text field
        refs = [{"evidence_ids": ["e1"], "text": "ref text"}]
        result = evaluate_evidence_support(preds, refs, evidence_text_map=text_map)
        self.assertEqual(result["evidence_precision"], 1.0)
        self.assertEqual(result["content_precision"], 0.0)
        self.assertEqual(result["content_hit_rate"], 0.0)

    def test_content_unsupported_counts_forced_answer_without_gold(self) -> None:
        """When gold evidence is empty but the system still answers, content_unsupported_rate=1.0."""
        text_map = {"e1": "some evidence text"}
        preds = [{"evidence_ids": ["e1"], "text": "claim"}]
        refs = [{"evidence_ids": [], "text": "claim"}]
        result = evaluate_evidence_support(preds, refs, evidence_text_map=text_map)
        self.assertEqual(result["unsupported_claim_rate"], 1.0)
        self.assertEqual(result["content_unsupported_rate"], 1.0)


class UncertaintyPreservationTests(unittest.TestCase):
    def test_preserved_and_collapsed(self) -> None:
        preds = [
            {"confidence": "low"},                 # uncertain item, flagged -> preserved
            {"confidence": "high"},                # uncertain item, not flagged -> collapsed
            {"speaker": "MIXED"},                  # uncertain item, flagged via MIXED
            {"confidence": "high"},                # certain item, correctly not flagged
        ]
        refs = [
            {"uncertain": True},
            {"uncertain": True},
            {"uncertain": True},
            {"uncertain": False},
        ]
        result = evaluate_uncertainty_preservation(preds, refs)
        self.assertEqual(result["uncertain_total"], 3)
        self.assertEqual(result["certain_total"], 1)
        self.assertAlmostEqual(result["preservation_rate"], 2 / 3)
        self.assertAlmostEqual(result["collapse_rate"], 1 / 3)
        self.assertEqual(result["false_uncertainty_rate"], 0.0)

    def test_signals_detected(self) -> None:
        preds = [
            {"insufficient_evidence": True},
            {"uncertainty_note": "speaker attribution uncertain"},
            {"candidates": [{"text": "a"}]},
        ]
        refs = [{"uncertain": True}, {"uncertain": True}, {"uncertain": True}]
        result = evaluate_uncertainty_preservation(preds, refs)
        self.assertEqual(result["preservation_rate"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_false_uncertainty_flagging(self) -> None:
        preds = [{"confidence": "low"}, {"confidence": "high"}]
        refs = [{"uncertain": False}, {"uncertain": False}]
        result = evaluate_uncertainty_preservation(preds, refs)
        self.assertEqual(result["false_uncertainty_rate"], 0.5)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_uncertainty_preservation([{"confidence": "low"}], [])


class CandidateUsefulnessTests(unittest.TestCase):
    def test_oracle_beats_top1(self) -> None:
        segments = [
            {
                "candidates": [
                    {"text": "the budget is approved", "speaker": "SPEAKER_00", "confidence": 0.8},
                    {"text": "the project is approved", "speaker": "SPEAKER_01", "confidence": 0.5},
                ]
            }
        ]
        refs = [{"text": "the project is approved", "speaker": "SPEAKER_01"}]
        result = evaluate_candidate_usefulness(segments, refs)
        self.assertEqual(result["evaluated"], 1)
        # top-1 = highest confidence (0.8) candidate -> 1 substitution / 4 words = 0.25
        self.assertAlmostEqual(result["top1_wer"], 0.25)
        # oracle = best candidate matches reference exactly -> 0.0
        self.assertAlmostEqual(result["oracle_wer"], 0.0)
        self.assertAlmostEqual(result["wer_reduction"], 0.25)
        self.assertEqual(result["speaker_coverage"], 1.0)

    def test_segment_without_candidates_is_counted_but_not_scored(self) -> None:
        segments = [{"candidates": []}]
        refs = [{"text": "anything", "speaker": "SPEAKER_00"}]
        result = evaluate_candidate_usefulness(segments, refs)
        self.assertEqual(result["support"], 1)
        self.assertEqual(result["evaluated"], 0)
        self.assertEqual(result["speaker_coverage"], 0.0)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_candidate_usefulness([{"candidates": []}], [])


if __name__ == "__main__":
    unittest.main()
