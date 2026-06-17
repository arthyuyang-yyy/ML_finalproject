"""Evaluation metrics for recognition, routing, attribution, and traceability.

This module implements word/character error rate, overlap-routing classification
metrics, best-mapping speaker-attribution accuracy, evidence-support metrics
(precision/recall/F1, hit rate, unsupported-claim and hallucination rate,
confidence calibration, and optional text-similarity proxy),
uncertainty-preservation quality, and candidate usefulness, following
``docs/experiment_plan.md`` (Experiments 2 and 5).

The optional text-similarity check (enabled via ``evidence_text_map``) measures
surface textual similarity, not semantic entailment. True content-level support
(NLI) is future work.
"""

from itertools import permutations
from typing import Any

from src.fallbacks.embeddings import HashingEmbeddingBackend

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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length unit vectors."""
    return sum(x * y for x, y in zip(a, b))


def _text_similarity(
    claim_text: str,
    evidence_text: str,
    threshold: float = 0.45,
    backend: HashingEmbeddingBackend | None = None,
) -> tuple[bool, float]:
    """Return the text-similarity score between *claim_text* and *evidence_text*.

    Uses the dependency-free :class:`HashingEmbeddingBackend` (character n-gram
    + CJK token hashing) so this check works without torch, transformers, or
    sentence-transformers.  Both texts are normalised and encoded; if either is
    empty the score is ``0.0``.

    .. note::
       This measures *surface textual similarity*, not semantic entailment or
       factual correctness.  Opposite statements can obtain high similarity
       scores (e.g. "the budget was approved" vs "the budget was not approved").
       Callers should treat the result as a loose filter, not a guarantee of
       semantic support.
    """
    claim_text = claim_text.strip()
    evidence_text = evidence_text.strip()
    if not claim_text or not evidence_text:
        return False, 0.0
    if backend is None:
        backend = HashingEmbeddingBackend()
    claim_vec, ev_vec = backend.encode([claim_text, evidence_text])
    score = _cosine_similarity(claim_vec, ev_vec)
    return score >= threshold, score


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
    evidence_text_map: dict[str, str] | None = None,
    text_similarity_threshold: float = 0.45,
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
    - prediction ``text`` (optional): the claim wording, used for content-level
      support checking when ``evidence_text_map`` is provided.  If missing,
      content-level similarity is scored as ``0.0``;
    - reference ``evidence_ids``: the gold supporting evidence (empty means the
      claim is not supportable and a faithful system should abstain);

    ``source_evidence_ids`` is the universe of evidence IDs that actually exist
    in the source segments; a cited ID outside it counts as a hallucination.

    ``evidence_text_map`` maps evidence IDs to their source text. When provided,
    an additional *text-similarity* check is performed: a citation counts as
    text-similarity-supported only when its ID is correct **and** its text has high
    surface textual similarity to the claim text (cosine similarity of
    :class:`HashingEmbeddingBackend` vectors >= ``text_similarity_threshold``).
    This is a lightweight proxy, **not** semantic entailment; opposite claims can
    obtain high similarity scores.  Callers requiring true NLI should replace
    this heuristic with a dedicated entailment model.

    Caveat: for a meaningful ``hallucination_rate`` callers MUST pass the full
    set of real source evidence IDs. When it is omitted it defaults to the union
    of all gold evidence IDs, which makes any real-but-non-gold citation look
    hallucinated and conflates hallucination with ``unsupported_claim_rate``.

    Returns evidence precision/recall/F1, evidence hit rate, unsupported-claim
    rate, hallucination rate, and confidence-calibration error (ECE).  When
    ``evidence_text_map`` is provided, additional ``text_similarity_*`` metrics are
    included (text-similarity precision/recall/F1, text-similarity hit rate, and
    text-similarity unsupported rate).  These metrics measure surface textual
    similarity, not semantic entailment.
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

    result = {
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

    # Text-similarity check (optional): compare claim text with cited evidence
    # text using surface similarity, not just whether the IDs match.
    if evidence_text_map is not None:
        text_similarity_cited = 0
        text_similarity_gold = 0
        text_similarity_correct = 0
        text_similarity_hits = 0
        text_similarity_unsupported = 0
        backend = HashingEmbeddingBackend()

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            claim_text = str(pred.get("text") or "").strip()
            cited = cited_sets[i]
            gold = gold_sets[i]

            cited_with_text = {eid for eid in cited if eid in evidence_text_map}
            gold_with_text = {eid for eid in gold if eid in evidence_text_map}

            text_similarity_cited += len(cited_with_text)
            text_similarity_gold += len(gold_with_text)

            has_text_similarity = False
            if claim_text and cited_with_text:
                claim_vec = backend.encode([claim_text])[0]
                for eid in cited_with_text & gold_with_text:
                    ev_vec = backend.encode([evidence_text_map[eid]])[0]
                    score = _cosine_similarity(claim_vec, ev_vec)
                    if score >= text_similarity_threshold:
                        text_similarity_correct += 1
                        has_text_similarity = True

            if gold_with_text and has_text_similarity:
                text_similarity_hits += 1
            if not pred.get("insufficient_evidence") and not has_text_similarity:
                text_similarity_unsupported += 1

        text_similarity_precision = text_similarity_correct / text_similarity_cited if text_similarity_cited else 0.0
        text_similarity_recall = text_similarity_correct / text_similarity_gold if text_similarity_gold else 0.0
        text_similarity_f1 = (
            2 * text_similarity_precision * text_similarity_recall / (text_similarity_precision + text_similarity_recall)
            if (text_similarity_precision + text_similarity_recall)
            else 0.0
        )

        answerable_text = [
            i for i, gold in enumerate(gold_sets)
            if any(eid in evidence_text_map for eid in gold)
        ]
        claims_text = [
            i for i in claims
            if any(eid in evidence_text_map for eid in gold_sets[i]) or not gold_sets[i]
        ]

        result.update({
            "text_similarity_precision": text_similarity_precision,
            "text_similarity_recall": text_similarity_recall,
            "text_similarity_f1": text_similarity_f1,
            "text_similarity_hit_rate": text_similarity_hits / len(answerable_text) if answerable_text else 0.0,
            "text_similarity_unsupported_rate": text_similarity_unsupported / len(claims_text) if claims_text else 0.0,
            "text_similarity_support": len(predictions),
            "text_similarity_answerable": len(answerable_text),
            "text_similarity_claims": len(claims_text),
            "text_similarity_cited": text_similarity_cited,
            "text_similarity_gold": text_similarity_gold,
            "text_similarity_correct": text_similarity_correct,
        })

    return result


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


def _marks_uncertainty(prediction: dict[str, Any]) -> bool:
    """Whether a prediction visibly preserves uncertainty rather than hiding it.

    A faithful system signals uncertainty through any of: abstention, low
    confidence, a non-empty uncertainty note, a ``MIXED`` speaker, or surviving
    candidate hypotheses.
    """
    if prediction.get("insufficient_evidence"):
        return True
    confidence = prediction.get("confidence")
    if isinstance(confidence, str) and confidence.strip().lower() == "low":
        return True
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and float(confidence) < 0.5:
        return True
    if str(prediction.get("uncertainty_note", "")).strip():
        return True
    if str(prediction.get("speaker", "")).strip().upper() == "MIXED":
        return True
    return bool(prediction.get("candidates"))


def evaluate_uncertainty_preservation(
    predictions: list[dict],
    references: list[dict],
) -> dict[str, Any]:
    """Measure whether genuinely uncertain claims stay marked as uncertain.

    Innovation point 2 requires that ambiguous high-overlap content is not
    silently collapsed into a single confident answer. Aligned by index, each
    ``reference`` carries a gold ``uncertain`` flag and each ``prediction`` is
    inspected for any uncertainty signal (see ``_marks_uncertainty``).

    Returns the preservation rate (recall over uncertain items), the collapse
    rate (uncertain items presented as confident), the false-uncertainty rate
    (confident items wrongly flagged), and precision/F1.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")

    marked = [_marks_uncertainty(pred) for pred in predictions]
    uncertain = [bool(ref.get("uncertain")) for ref in references]

    uncertain_total = sum(uncertain)
    certain_total = len(references) - uncertain_total
    preserved = sum(m for m, u in zip(marked, uncertain) if u)
    false_flags = sum(m for m, u in zip(marked, uncertain) if not u)

    preservation_rate = preserved / uncertain_total if uncertain_total else 0.0
    collapse_rate = 1.0 - preservation_rate if uncertain_total else 0.0
    false_uncertainty_rate = false_flags / certain_total if certain_total else 0.0
    flagged = preserved + false_flags
    precision = preserved / flagged if flagged else 0.0
    f1 = (
        2 * precision * preservation_rate / (precision + preservation_rate)
        if (precision + preservation_rate)
        else 0.0
    )
    return {
        "support": len(predictions),
        "uncertain_total": uncertain_total,
        "certain_total": certain_total,
        "preservation_rate": preservation_rate,
        "collapse_rate": collapse_rate,
        "false_uncertainty_rate": false_uncertainty_rate,
        "precision": precision,
        "f1": f1,
    }


def evaluate_candidate_usefulness(
    segments: list[dict],
    references: list[dict],
) -> dict[str, Any]:
    """Quantify whether multi-candidate high-overlap output keeps useful signal.

    Aligned by index, each ``segment`` exposes ``candidates`` (each with ``text``,
    ``speaker``, and ``confidence``) and each ``reference`` carries the gold
    ``text`` and ``speaker``. Segments without candidates are counted but cannot
    contribute a WER and never cover the speaker.

    Returns mean oracle WER (best candidate), mean top-1 WER (most confident
    candidate), their difference (the value of keeping several candidates), and
    speaker-hypothesis coverage.
    """
    if len(segments) != len(references):
        raise ValueError("segments and references must have the same length")
    if not segments:
        raise ValueError("segments must not be empty")

    oracle_wers: list[float] = []
    top1_wers: list[float] = []
    speaker_hits = 0
    for segment, reference in zip(segments, references):
        ref_text = str(reference.get("text", ""))
        ref_speaker = str(reference.get("speaker", ""))
        candidates = [c for c in segment.get("candidates", []) if str(c.get("text", "")).strip()]
        if any(str(c.get("speaker", "")) == ref_speaker and ref_speaker for c in segment.get("candidates", [])):
            speaker_hits += 1
        if not candidates:
            continue
        wers = [word_error_rate(ref_text, str(c["text"]))["error_rate"] for c in candidates]
        oracle_wers.append(min(wers))
        best = max(candidates, key=lambda c: float(c.get("confidence", 0.0) or 0.0))
        top1_wers.append(word_error_rate(ref_text, str(best["text"]))["error_rate"])

    evaluated = len(oracle_wers)
    oracle_wer = sum(oracle_wers) / evaluated if evaluated else 0.0
    top1_wer = sum(top1_wers) / evaluated if evaluated else 0.0
    return {
        "support": len(segments),
        "evaluated": evaluated,
        "oracle_wer": oracle_wer,
        "top1_wer": top1_wer,
        "wer_reduction": top1_wer - oracle_wer,
        "speaker_coverage": speaker_hits / len(segments),
    }


def _error_rate_result(counts: dict[str, int]) -> dict[str, Any]:
    """Attach a normalized error rate to raw edit-distance counts."""
    ref_length = counts["reference_length"]
    rate = counts["distance"] / ref_length if ref_length else float(counts["distance"] > 0)
    return {**counts, "error_rate": rate}


def evaluate_event_extraction(
    predicted_events: list[dict[str, Any]],
    gold_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Precision/recall/F1 for structured event extraction, overall and per type.

    A predicted event matches a gold event when they share the same
    ``event_type`` and cite at least one common ``evidence_id``. Matching is
    greedy and one-to-one so duplicate predictions cannot be rewarded twice.
    """
    pred = [_event_signature(event) for event in predicted_events]
    gold = [_event_signature(event) for event in gold_events]

    matched_gold: set[int] = set()
    matched_total = 0
    matched_by_type: dict[str, int] = {}
    for pred_type, pred_ids in pred:
        for index, (gold_type, gold_ids) in enumerate(gold):
            if index in matched_gold or pred_type != gold_type or not (pred_ids & gold_ids):
                continue
            matched_gold.add(index)
            matched_total += 1
            matched_by_type[pred_type] = matched_by_type.get(pred_type, 0) + 1
            break

    pred_counts = _count_types(pred)
    gold_counts = _count_types(gold)
    per_type = {
        event_type: _precision_recall_f1(
            matched_by_type.get(event_type, 0),
            pred_counts.get(event_type, 0),
            gold_counts.get(event_type, 0),
        )
        for event_type in sorted(set(pred_counts) | set(gold_counts))
    }

    return {
        "support_predicted": len(pred),
        "support_gold": len(gold),
        "matched": matched_total,
        **_precision_recall_f1(matched_total, len(pred), len(gold)),
        "per_type": per_type,
    }


def _event_signature(event: dict[str, Any]) -> tuple[str, set[str]]:
    return (
        str(event.get("event_type", "")),
        {str(value) for value in event.get("evidence_ids", [])},
    )


def _count_types(signatures: list[tuple[str, set[str]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event_type, _ in signatures:
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _precision_recall_f1(matched: int, predicted: int, gold: int) -> dict[str, float]:
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
