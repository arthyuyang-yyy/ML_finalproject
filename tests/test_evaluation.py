"""Tests for objective evaluation metrics.

Run with::

    python -m unittest tests.test_evaluation
"""

import unittest

from src.evaluation import (
    character_error_rate,
    edit_distance,
    evaluate_evidence_support,
    evaluate_overlap_routing,
    speaker_attribution_accuracy,
    word_error_rate,
)


class EditDistanceTests(unittest.TestCase):
    def test_identical_sequences(self) -> None:
        result = edit_distance(list("abc"), list("abc"))
        self.assertEqual(result["distance"], 0)

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


class EvidenceSupportTests(unittest.TestCase):
    def test_empty_predictions_raise(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_evidence_support([], [])

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_evidence_support([{"evidence_ids": ["a"]}], [])

    def test_string_evidence_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_evidence_support(
                [{"evidence_ids": "a", "confidence": 1.0}],
                [{"evidence_ids": ["a"]}],
            )

    def test_perfectly_grounded_answer(self) -> None:
        result = evaluate_evidence_support(
            [{"evidence_ids": ["m1-1", "m1-2"], "confidence": 0.9}],
            [{"evidence_ids": ["m1-1", "m1-2"], "answerable": True}],
        )
        self.assertEqual(result["num_claims"], 1)
        self.assertEqual(result["evidence_precision"], 1.0)
        self.assertEqual(result["evidence_recall"], 1.0)
        self.assertEqual(result["evidence_f1"], 1.0)
        self.assertEqual(result["evidence_hit_rate"], 1.0)
        self.assertEqual(result["hallucination_rate"], 0.0)

    def test_partial_evidence_precision_and_recall(self) -> None:
        # Cites two ids, one correct: precision 0.5; gold has two, one hit: recall 0.5.
        result = evaluate_evidence_support(
            [{"evidence_ids": ["m1-1", "wrong"], "confidence": 0.6}],
            [{"evidence_ids": ["m1-1", "m1-2"], "answerable": True}],
        )
        self.assertAlmostEqual(result["evidence_precision"], 0.5)
        self.assertAlmostEqual(result["evidence_recall"], 0.5)
        self.assertEqual(result["evidence_hit_rate"], 1.0)
        self.assertEqual(result["hallucination_rate"], 0.0)

    def test_unsupported_claim_is_hallucination(self) -> None:
        result = evaluate_evidence_support(
            [{"evidence_ids": ["made-up"], "confidence": 0.95}],
            [{"evidence_ids": ["m1-1"], "answerable": True}],
        )
        self.assertEqual(result["hallucination_rate"], 1.0)
        self.assertEqual(result["evidence_hit_rate"], 0.0)

    def test_claim_on_unanswerable_question_is_hallucination(self) -> None:
        result = evaluate_evidence_support(
            [{"evidence_ids": ["m1-1"], "confidence": 0.8}],
            [{"evidence_ids": [], "answerable": False}],
        )
        self.assertEqual(result["num_claims"], 1)
        self.assertEqual(result["hallucination_rate"], 1.0)
        self.assertEqual(result["correct_abstention_rate"], 0.0)

    def test_correct_abstention_on_unanswerable(self) -> None:
        result = evaluate_evidence_support(
            [{"evidence_ids": [], "confidence": 0.0}],
            [{"evidence_ids": [], "answerable": False}],
        )
        self.assertEqual(result["num_abstentions"], 1)
        self.assertEqual(result["num_claims"], 0)
        self.assertEqual(result["correct_abstention_rate"], 1.0)
        self.assertEqual(result["hallucination_rate"], 0.0)

    def test_brier_rewards_calibrated_confidence(self) -> None:
        confident_correct = evaluate_evidence_support(
            [{"evidence_ids": ["m1-1"], "confidence": 1.0}],
            [{"evidence_ids": ["m1-1"], "answerable": True}],
        )
        confident_wrong = evaluate_evidence_support(
            [{"evidence_ids": ["wrong"], "confidence": 1.0}],
            [{"evidence_ids": ["m1-1"], "answerable": True}],
        )
        self.assertEqual(confident_correct["confidence_brier"], 0.0)
        self.assertEqual(confident_wrong["confidence_brier"], 1.0)

    def test_abstaining_on_answerable_lowers_recall(self) -> None:
        # Regression: gold evidence of an answerable-but-abstained question must
        # still count toward recall. Two answerable questions; the first cites
        # the correct evidence, the second abstains -> recall should be 0.5, not
        # 1.0 (which the old code returned by skipping the abstained sample).
        result = evaluate_evidence_support(
            [
                {"evidence_ids": ["a"], "confidence": 0.9},
                {"evidence_ids": [], "confidence": 0.0},
            ],
            [
                {"evidence_ids": ["a"], "answerable": True},
                {"evidence_ids": ["b"], "answerable": True},
            ],
        )
        self.assertAlmostEqual(result["evidence_recall"], 0.5)
        self.assertAlmostEqual(result["evidence_precision"], 1.0)
        self.assertAlmostEqual(result["evidence_hit_rate"], 0.5)
        self.assertEqual(result["num_claims"], 1)
        self.assertEqual(result["num_abstentions"], 1)

    def test_explicit_non_abstention_scores_no_evidence_answer_as_claim(self) -> None:
        # A substantive answer that cites no evidence, flagged abstained=False,
        # is an unsupported claim (hallucination), not an abstention.
        result = evaluate_evidence_support(
            [{"answer": "It was Bob.", "evidence_ids": [], "abstained": False, "confidence": 0.8}],
            [{"evidence_ids": ["a"], "answerable": True}],
        )
        self.assertEqual(result["num_claims"], 1)
        self.assertEqual(result["num_abstentions"], 0)
        self.assertEqual(result["hallucination_rate"], 1.0)
        self.assertAlmostEqual(result["evidence_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
