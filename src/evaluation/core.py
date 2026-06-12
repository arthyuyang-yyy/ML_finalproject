"""Evaluation metrics for recognition, routing, attribution, and traceability.

This module implements word/character error rate, overlap-routing classification
metrics, best-mapping speaker-attribution accuracy, and evidence-support metrics
(precision/recall/F1, hit rate, unsupported-claim and hallucination rate, and
confidence calibration) following ``docs/experiment_plan.md`` (Experiment 5).

Uncertainty-preservation quality and candidate usefulness remain TODO stubs
until their scoring rules are finalized.
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


_CONFIDENCE_VALUES = {"high": 0.9, "medium": 0.6, "low": 0.3}


def _confidence_value(confidence: Any) -> float | None:
    """Map a confidence label or numeric score to a probability in [0, 1]."""
    if isinstance(confidence, bool):
        return None
    if isinstance(confidence, (int, float)):
        return max(0.0, min(1.0, float(confidence)))
    return _CONFIDENCE_VALUES.get(str(confidence).strip().lower())


def evaluate_evidence_support(
    predictions: list[dict],
    references: list[dict],
    source_evidence_ids: Any = None,
) -> dict[str, Any]:
    """Evaluate evidence traceability for a set of aligned claim/reference pairs.

    Each ``prediction`` is a claim (e.g. a QA answer, decision, or action item)
    carrying the evidence it cites; each ``reference`` is the gold annotation for
    the same claim. They are aligned by index. Relevant fields:

    - prediction ``evidence_ids``: evidence the claim cites;
    - prediction ``insufficient_evidence`` (optional): ``True`` marks an
      abstention rather than an asserted claim;
    - prediction ``confidence`` (optional): ``"high"``/``"medium"``/``"low"`` or
      a numeric score in [0, 1], used for calibration;
    - reference ``evidence_ids``: the gold supporting evidence (empty means the
      claim is not supportable and a faithful system should abstain).

    ``source_evidence_ids`` is the universe of evidence IDs that actually exist
    in the source segments; a cited ID outside it counts as a hallucination.

    Caveat: for a meaningful ``hallucination_rate`` callers MUST pass the full
    set of real source evidence IDs. When it is omitted it defaults to the union
    of all gold evidence IDs, which makes any real-but-non-gold citation look
    hallucinated and conflates hallucination with ``unsupported_claim_rate``.

    Scope: these metrics check whether the *cited evidence IDs* are correct; they
    do not yet verify that the claim *text* is entailed by the evidence content.
    Uncertainty-preservation and candidate-usefulness metrics are not covered
    here (see ``docs/experiment_plan.md`` Experiment 5 follow-ups).

    Returns evidence precision/recall/F1, evidence hit rate, unsupported-claim
    rate, hallucination rate, and confidence-calibration error (ECE), following
    the metric definitions in ``docs/experiment_plan.md`` (Experiment 5).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")

    gold_sets = [{str(value) for value in ref.get("evidence_ids", [])} for ref in references]
    cited_sets = [{str(value) for value in pred.get("evidence_ids", [])} for pred in predictions]
    if source_evidence_ids is None:
        source_universe = set().union(*gold_sets) if gold_sets else set()
    else:
        source_universe = {str(value) for value in source_evidence_ids}

    total_cited = sum(len(cited) for cited in cited_sets)
    total_gold = sum(len(gold) for gold in gold_sets)
    total_correct = sum(len(cited & gold) for cited, gold in zip(cited_sets, gold_sets))

    precision = total_correct / total_cited if total_cited else 0.0
    recall = total_correct / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Evidence hit rate: among answerable claims (gold evidence exists), the
    # fraction that cited at least one correct piece of evidence.
    answerable = [i for i, gold in enumerate(gold_sets) if gold]
    hits = sum(bool(cited_sets[i] & gold_sets[i]) for i in answerable)
    evidence_hit_rate = hits / len(answerable) if answerable else 0.0

    # Claim-level rates over asserted claims (abstentions are not claims).
    claims = [i for i, pred in enumerate(predictions) if not pred.get("insufficient_evidence")]
    unsupported = sum(not (cited_sets[i] & gold_sets[i]) for i in claims)
    hallucinated = sum(bool(cited_sets[i] - source_universe) for i in claims)
    unsupported_claim_rate = unsupported / len(claims) if claims else 0.0
    hallucination_rate = hallucinated / len(claims) if claims else 0.0

    calibration = _confidence_calibration(predictions, cited_sets, gold_sets)

    return {
        "support": len(predictions),
        "answerable": len(answerable),
        "claims": len(claims),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "evidence_hit_rate": evidence_hit_rate,
        "unsupported_claim_rate": unsupported_claim_rate,
        "hallucination_rate": hallucination_rate,
        **calibration,
    }


def _confidence_calibration(
    predictions: list[dict],
    cited_sets: list[set[str]],
    gold_sets: list[set[str]],
) -> dict[str, Any]:
    """Expected Calibration Error over predictions that expose a confidence.

    A claim is correct when it cites at least one gold evidence; an abstention is
    correct when there is no gold evidence to cite. Predictions are binned by
    their confidence value and ECE is the support-weighted gap between mean
    confidence and accuracy across bins.
    """
    bins: dict[float, list[bool]] = {}
    for pred, cited, gold in zip(predictions, cited_sets, gold_sets):
        value = _confidence_value(pred.get("confidence"))
        if value is None:
            continue
        if pred.get("insufficient_evidence"):
            correct = not gold
        else:
            correct = bool(cited & gold)
        bins.setdefault(value, []).append(correct)

    scored = sum(len(outcomes) for outcomes in bins.values())
    if not scored:
        return {"calibration_error": 0.0, "calibration_samples": 0}

    ece = sum(
        len(outcomes) * abs(value - sum(outcomes) / len(outcomes))
        for value, outcomes in bins.items()
    ) / scored
    return {"calibration_error": ece, "calibration_samples": scored}


def _error_rate_result(counts: dict[str, int]) -> dict[str, Any]:
    """Attach a normalized error rate to raw edit-distance counts."""
    ref_length = counts["reference_length"]
    rate = counts["distance"] / ref_length if ref_length else float(counts["distance"] > 0)
    return {**counts, "error_rate": rate}
