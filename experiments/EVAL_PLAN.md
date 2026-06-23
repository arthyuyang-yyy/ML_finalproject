# Evaluation plan

## 4D experiment matrix

| Axis | Levels | Goal |
|---|---|---|
| A. ASR backend | `faster-whisper` · `funasr` · `whisperx` | which ASR is best on Chinese meeting audio |
| B. OSD strategy | `pyannote` · `energy_fallback` | does real OSD beat energy proxy |
| C. LLM resolver | `none` · `deepseek` | does LLM-picked candidate beat deterministic fallback |
| D. Speech separation | `none` · `sepformer` | does separation upstream help resolver |

`24 cells × 8 meetings = 192 runs`.

## Data

- Audio: `data/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/<mid>_MS801.wav` (8 ch, 16 kHz, ~26 min)
- Ground truth: `data/alimeeting/annotations/<mid>.json`
  - `turns: [{speaker, start_time, end_time}]` (4 speakers / meeting)
  - `overlap_regions: [{start_time, end_time}]` (multi-speaker)
  - `overlap_seconds`, `num_speakers`, `meeting_id`
- Window: per meeting, full 26 min or 5-min clip (`[0, 300]`); both supported

## Output fields per run

`main.py` writes (verbatim, **not** modified by this layer):

```text
preprocessed.wav
vad_segments.json
diarization.json
overlap.json
routed_segments.json
low_overlap_segments.json
high_overlap_candidates.json
evidence_segments.json
meeting_events.json
episodic_memory.json
clips/...
```

The experiment layer adds two extra files per run (written into the same dir):

- `run_meta.json` — config snapshot + wall_time + detector source counts
- `evaluation.json` — all scores (this layer's contribution)

## Scoring — reuse-first

We **do not** write new metric modules. We orchestrate existing functions:

| Metric | Source | Reused function | Inline? |
|---|---|---|---|
| `cer_concat` | `src.evaluation.core` | `character_error_rate` | — |
| `cer_low` / `cer_high` | filter by `processing_path` then concat, then `character_error_rate` | yes | 4 lines |
| `wer_concat` | `src.evaluation.core` | `word_error_rate` | — |
| `routing_f1` | `src.evaluation.core` | `evaluate_overlap_routing` | 12 lines (build GT labels) |
| `overlap_recall/precision/f1` | `scripts.evaluate_alimeeting_result` | `_covered_seconds_by_regions` | — |
| `speaker_best_mapping_accuracy` | `src.evaluation.core` | `speaker_attribution_accuracy` | 12 lines (1-to-1 IoU match) |
| `speaker_known_coverage` | `scripts.evaluate_alimeeting_result` | `_speaker_report` | — |
| `evidence_f1` / `unsupported_claim_rate` / `hallucination_rate` | `src.evaluation.core` | `evaluate_evidence_support` (source_universe = all evidence_ids) | — |
| `events_per_minute` / `supported_event_rate` / `unresolved_rate` / `llm_resolved_rate` | inline counters | — | 8 lines |
| `wall_time_s` / `rtf` | `run_meta.json` | — | 3 lines |

**Per-cell aggregate**: mean / std across 8 meetings.

**Per-axis slice**: fix 3 axes, vary 1, report delta. This is the *conclusion layer*.

## What is intentionally NOT done

- No new metric modules
- No ARI / NMI (best-mapping is the project's convention)
- No bin-level CER (concat-CER + char-level is the project convention)
- No "GT events" (we have none) → event evaluation stays operational only
- Sanity checks live in `run_meta.warnings`, not in `summary.csv`
