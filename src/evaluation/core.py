"""Evaluation metrics for recognition, routing, and speaker attribution.

This module implements the *objective* metrics that do not depend on the team's
still-evolving innovation design: word/character error rate, overlap-routing
classification metrics, and best-mapping speaker-attribution accuracy.

Metrics tied to the innovation (evidence hit rate, unsupported-claim and
hallucination rate, uncertainty-preservation quality, candidate usefulness) are
intentionally left as TODO stubs until their scoring rules are finalized, so the
shared interface does not lock in a definition prematurely.
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
    """Evaluate evidence hit rate, unsupported claims, and hallucination.

    Deferred: scoring rules for timestamped evidence and supported claims are
    part of the traceability innovation and will be defined once that design is
    finalized, to avoid locking in the contract prematurely.
    """
    # TODO(innovation): define matching rules for timestamped evidence,
    # supported-claim detection, hallucination rate, and confidence calibration.
    raise NotImplementedError("Evidence evaluation is deferred until the traceability design is finalized.")


def _error_rate_result(counts: dict[str, int]) -> dict[str, Any]:
    """Attach a normalized error rate to raw edit-distance counts."""
    ref_length = counts["reference_length"]
    rate = counts["distance"] / ref_length if ref_length else float(counts["distance"] > 0)
    return {**counts, "error_rate": rate}
