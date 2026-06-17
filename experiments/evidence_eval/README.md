# Evidence Evaluation Experiment

Runs the implemented evaluation metrics over a manually-annotated set and prints
a results table. Covers Experiment 5 (evidence support / hallucination) and the
traceability-related metrics from Experiment 2 (candidate usefulness) and
Experiment 3 (uncertainty preservation). The previous experiment plan has been retired as the project scope narrowed to a focused MVP; the implemented metrics are described below.

## Run

```bash
python experiments/evidence_eval/run_experiment.py
# or point at your own annotations and output directory:
python experiments/evidence_eval/run_experiment.py --annotations my.json --out-dir out/
```

Writes `results.json` and `results.md` next to the script (or to `--out-dir`).

## Metrics

Implemented in [`src/evaluation/core.py`](../../src/evaluation/core.py):

- `evaluate_evidence_support` — evidence precision/recall/F1, evidence hit rate,
  unsupported-claim rate, hallucination rate, confidence calibration (ECE).
- `evaluate_uncertainty_preservation` — preservation rate, collapse rate, and
  false-uncertainty rate for genuinely uncertain claims.
- `evaluate_candidate_usefulness` — oracle WER, top-1 WER, their difference, and
  speaker-hypothesis coverage for high-overlap candidates.

## Annotation schema (`annotations.json`)

```json
{
  "source_evidence_ids": ["ev_001", "..."],
  "qa_items": [
    {
      "id": "string",
      "question": "string",
      "prediction": {"evidence_ids": ["..."], "confidence": "high|medium|low",
                     "insufficient_evidence": false, "uncertainty_note": ""},
      "reference": {"evidence_ids": ["...gold..."], "uncertain": false}
    }
  ],
  "candidate_segments": [
    {
      "id": "string",
      "segment": {"candidates": [{"text": "...", "speaker": "...", "confidence": 0.0}]},
      "reference": {"text": "...gold...", "speaker": "..."}
    }
  ]
}
```

- `source_evidence_ids` MUST list every evidence ID that really exists in the
  source meeting; the hallucination rate depends on it. If omitted it falls back
  to the union of gold IDs, which conflates wrong-but-real citations with
  hallucinations.
- A `reference.evidence_ids` of `[]` marks an unanswerable claim — a faithful
  system should abstain (`insufficient_evidence: true`).

## Status and limitations

The bundled `annotations.json` is a **small hand-authored seed** that exercises
every metric (correct, over-citation, hallucination, abstention, overclaim,
preserved-uncertainty, collapsed-uncertainty, and candidate cases). It is **not**
a real evaluation corpus.

Before reporting final paper numbers:

1. Annotate real pipeline/QA outputs (the gold `reference` fields) for a proper
   evaluation split.
2. These metrics check cited evidence IDs, uncertainty signals, and candidate
   text — they do **not** yet verify that a claim's wording is entailed by the
   evidence content. True semantic entailment (NLI) is future work.
