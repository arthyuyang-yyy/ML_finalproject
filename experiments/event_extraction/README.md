# Rule-based Structured-Event Extraction Baseline (Step 16a)

A dependency-free, deterministic baseline that extracts structured meeting
events from Evidence segments using bilingual (zh/en) keyword and pattern rules
— no LLM required. It is the **rule arm** that Step 16b will compare against
plain LLM extraction and full-Evidence constrained LLM extraction.

## What it does

`src/events/rule_extractor.py`:

- classifies each low-overlap segment by cue priority
  `decision > action_item > open_question > speaker_stance`;
- extracts `action_item` `owner` (the segment speaker, or `"uncertain"`) and an
  optional `deadline` (zh/en date and relative-time patterns);
- represents every high-overlap segment as a low-confidence `uncertainty` event;
- emits the canonical meeting-event schema and is checked by the shared
  `validate_meeting_events_document` validator.

Scoring (`evaluate_event_extraction` in `src/evaluation/core.py`): a predicted
event matches a gold event when they share the same `event_type` and at least
one `evidence_id` (greedy one-to-one). Reports overall and per-type
precision/recall/F1.

## Run

```bash
python experiments/event_extraction/run_experiment.py
# optional: --annotations <path.json> --out-dir <dir>
```

Outputs `results.json` and `results.md`.

## Notes

`annotations.json` is a small hand-authored seed (one meeting). It deliberately
includes an implicitly-phrased decision ("那就先这样推进吧") that carries no
surface cue, so the baseline mis-classifies it as `speaker_stance` — this is the
expected limitation of a rule baseline and the motivation for the Step 16b
comparison against LLM extraction. Replace/extend with real human annotations
before reporting final paper numbers.
