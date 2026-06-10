"""Evaluation metrics for recognition, routing, attribution, and evidence.

This module implements the *objective* recognition metrics (word/character error
rate, overlap-routing classification, best-mapping speaker-attribution accuracy)
together with the traceability metrics tied to the innovation: evidence
precision/recall/F1, evidence hit rate, hallucination rate, correct-abstention
rate, and confidence calibration (see :func:`evaluate_evidence_support`).

Remaining innovation metrics (uncertainty-preservation quality, candidate
usefulness) are still left as future work until their scoring rules are
finalized, so the shared interface does not lock in a definition prematurely.
"""

from itertools import permutations
from typing import Any

HIGH_OVERLAP = "high_overlap_candidate"
LOW_OVERLAP = "low_overlap_cluster"


def edit_distance(reference: list[Any], hypothesis: list[Any]) -> dict[str, int]:
    """Levenshtein alignment counts between two token sequences.

    Returns substitutions, deletions, insertions, the total distance, and the
    reference length. Used by both WER and CER.
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # match / substitution
            )

    # Backtrace to count operation types.
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "distance": subs + dels + ins,
        "reference_length": n,
    }


def word_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Word Error Rate with substitution/deletion/insertion breakdown."""
    counts = edit_distance(reference.split(), hypothesis.split())
    return _error_rate_result(counts)


def character_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Character Error Rate (whitespace ignored); suited to Chinese text."""
    ref_chars = [c for c in reference if not c.isspace()]
    hyp_chars = [c for c in hypothesis if not c.isspace()]
    counts = edit_distance(ref_chars, hyp_chars)
    return _error_rate_result(counts)


def evaluate_overlap_routing(predictions: list[str], references: list[str]) -> dict[str, Any]:
    """Evaluate low/high-overlap routing as binary classification.

    ``high_overlap_candidate`` is the positive class. Returns accuracy plus
    precision, recall, and F1 for detecting high-overlap segments.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")

    valid = {HIGH_OVERLAP, LOW_OVERLAP}
    for label in (*predictions, *references):
        if label not in valid:
            raise ValueError(f"labels must be one of {sorted(valid)}, got '{label}'")

    tp = sum(p == HIGH_OVERLAP and r == HIGH_OVERLAP for p, r in zip(predictions, references))
    fp = sum(p == HIGH_OVERLAP and r == LOW_OVERLAP for p, r in zip(predictions, references))
    fn = sum(p == LOW_OVERLAP and r == HIGH_OVERLAP for p, r in zip(predictions, references))
    correct = sum(p == r for p, r in zip(predictions, references))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": correct / len(predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(predictions),
    }


def speaker_attribution_accuracy(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Best-mapping speaker-attribution accuracy over aligned segments.

    Diarization labels are arbitrary, so accuracy is computed under the optimal
    one-to-one mapping between hypothesis and reference speaker labels (brute
    force over permutations; intended for a small number of speakers).
    """
    if len(reference) != len(hypothesis):
        raise ValueError("reference and hypothesis must have the same length")
    if not reference:
        raise ValueError("inputs must not be empty")

    ref_labels = sorted(set(reference))
    hyp_labels = sorted(set(hypothesis))
    if len(hyp_labels) <= len(ref_labels):
        smaller, larger, map_hyp_to_ref = hyp_labels, ref_labels, True
    else:
        smaller, larger, map_hyp_to_ref = ref_labels, hyp_labels, False

    best_correct = 0
    for perm in permutations(larger, len(smaller)):
        mapping = dict(zip(smaller, perm))
        if map_hyp_to_ref:
            correct = sum(mapping.get(h) == r for h, r in zip(hypothesis, reference))
        else:
            correct = sum(h == mapping.get(r) for h, r in zip(hypothesis, reference))
        best_correct = max(best_correct, correct)

    return {
        "accuracy": best_correct / len(reference),
        "support": len(reference),
        "reference_speakers": len(ref_labels),
        "hypothesis_speakers": len(hyp_labels),
    }


def evaluate_evidence_support(predictions: list[dict], references: list[dict]) -> dict:
    """Evaluate evidence grounding: hit rate, hallucination, and calibration.

    Scoring follows the system's traceability contract: every genuine answer
    must cite timestamped evidence, so by default a prediction that cites no
    evidence is read as an *abstention* ("I don't know") rather than a claim.

    Each ``prediction`` is a QA result (e.g. from
    :func:`src.rag_qa.answer_question_with_evidence`) and is read for:

    - ``evidence_ids`` -- the evidence the answer is grounded on;
    - ``confidence`` -- the answer's self-reported confidence in ``[0, 1]``;
    - ``abstained`` -- optional explicit flag. When absent, abstention is
      inferred from empty ``evidence_ids``. Set it to ``False`` to score a
      substantive answer that cites no evidence as an (unsupported) claim rather
      than an abstention.

    Each ``reference`` is the gold annotation for the same question:

    - ``evidence_ids`` -- the evidence that genuinely supports an answer;
    - ``answerable`` -- whether the question can be answered from memory at all
      (optional; defaults to ``True`` when gold evidence is present).

    A claim is *supported* when it cites at least one gold evidence id, and a
    *hallucination* when it cites none (including any claim made for an
    unanswerable question). Recall is computed over the gold evidence of *all*
    answerable questions, so abstaining on an answerable question lowers recall
    rather than being silently excluded. Returns micro-averaged evidence
    precision/recall/F1, the evidence hit rate, the hallucination rate, the
    correct-abstention rate on unanswerable questions, and a Brier calibration
    score (lower is better).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")

    num_claims = 0
    num_abstentions = 0
    unsupported_claims = 0
    hits = 0
    answerable_count = 0
    correct_abstentions = 0
    unanswerable_count = 0
    intersection_total = 0
    predicted_total = 0
    gold_total = 0
    brier_terms: list[float] = []

    for prediction, reference in zip(predictions, references):
        predicted_ids = _evidence_id_set(prediction.get("evidence_ids"))
        gold_ids = _evidence_id_set(reference.get("evidence_ids"))
        answerable = bool(reference.get("answerable", bool(gold_ids)))
        # A prediction abstains when explicitly flagged, or (by default) when it
        # cites no evidence. An explicit ``abstained=False`` keeps a no-evidence
        # answer in the claim path so it is scored as an unsupported claim.
        abstained = bool(prediction.get("abstained", not predicted_ids))

        if answerable:
            answerable_count += 1
            # Count gold evidence for every answerable question, even when the
            # system abstains, so abstaining on an answerable question lowers
            # recall instead of being silently excluded.
            gold_total += len(gold_ids)
        else:
            unanswerable_count += 1

        if abstained:
            num_abstentions += 1
            if not answerable:
                correct_abstentions += 1
            continue

        num_claims += 1
        overlap = predicted_ids & gold_ids
        supported = bool(overlap)
        if not supported:
            unsupported_claims += 1

        confidence = _clamp_unit(prediction.get("confidence", 0.0))
        brier_terms.append((confidence - (1.0 if supported else 0.0)) ** 2)

        if answerable:
            intersection_total += len(overlap)
            predicted_total += len(predicted_ids)
            if supported:
                hits += 1

    precision = intersection_total / predicted_total if predicted_total else 0.0
    recall = intersection_total / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "support": len(predictions),
        "num_claims": num_claims,
        "num_abstentions": num_abstentions,
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "evidence_hit_rate": hits / answerable_count if answerable_count else 0.0,
        "hallucination_rate": unsupported_claims / num_claims if num_claims else 0.0,
        "correct_abstention_rate": (
            correct_abstentions / unanswerable_count if unanswerable_count else 0.0
        ),
        "confidence_brier": sum(brier_terms) / len(brier_terms) if brier_terms else 0.0,
    }


def _evidence_id_set(evidence_ids: Any) -> set[str]:
    """Normalize an evidence-id field into a set of strings."""
    if not evidence_ids:
        return set()
    if isinstance(evidence_ids, (str, bytes)):
        raise ValueError("evidence_ids must be a list of ids, not a single string")
    return {str(evidence_id) for evidence_id in evidence_ids}


def _clamp_unit(value: Any) -> float:
    """Clamp a confidence-like value into ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(value)))


def _error_rate_result(counts: dict[str, int]) -> dict[str, Any]:
    """Attach a normalized error rate to raw edit-distance counts."""
    ref_length = counts["reference_length"]
    rate = counts["distance"] / ref_length if ref_length else float(counts["distance"] > 0)
    return {**counts, "error_rate": rate}
