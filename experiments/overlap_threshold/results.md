# Overlap Threshold Calibration — Results (preliminary)

First end-to-end run on real data. **Preliminary**: 8 meetings, 2 in the test
split, so absolute numbers are noisy. Raw audio/annotations are not committed
(see `../../data/README.md`); only metrics are recorded here.

> **Scope — this is an exploratory, fixed-window, OSD-only experiment.** It uses
> fixed 2 s windows and the pyannote OSD score with diarization fusion **disabled**.
> The real Pipeline (`run_meeting_pipeline`) routes **VAD segments** using the
> **fused** overlap score, so segment length and score composition differ and the
> same threshold value does not carry over. Treat the calibrated value as an
> exploratory finding — **not** as a finished Step 7b, and **not** as a drop-in
> replacement of the default `0.4`. A proper routing-threshold calibration must
> rerun on the same VAD segments and fused scores the Pipeline actually uses.

## Setup

- **Dataset:** AliMeeting Eval, far-field mix (`Eval_Ali_far/audio_dir`), 8 meetings (~34 min each).
- **Ground truth:** speaker turns → overlap regions (≥2 simultaneous speakers), via `scripts/prepare_alimeeting.py`.
- **Segmentation:** fixed 2.0 s windows (`--window-seconds 2.0`), 7572 windows total.
- **Label:** a window is high-overlap when ≥ `gt_fraction = 0.5` of it is covered by ground-truth overlap.
- **Split:** by meeting, `test_ratio = 0.3`, `seed = 0` → train 6 / test 2 (train 5418 windows, test 2154).
- **Threshold sweep:** 0.05 … 0.95, step 0.05; best F1 on train, reported on held-out test.

## Results (held-out test)

| Detector | F1 @calibrated | F1 @default (0.4) | P / R @calibrated | Acc @calibrated |
| --- | --- | --- | --- | --- |
| **pyannote OSD** | **0.346** | 0.247 | 0.508 / 0.262 | 0.893 |
| energy fallback | 0.195 | **0.000** | 0.108 / 1.000 | 0.108 |

Both detectors calibrated to threshold **0.05**.

## Reading the numbers

1. **pyannote clearly beats the energy fallback** — F1 0.346 vs 0.195, and at the
   current default 0.4, pyannote 0.247 vs energy **0.000** (the capped energy proxy
   routes nothing to the high-overlap path). This justifies using pyannote OSD.
2. **The default threshold 0.4 is too high.** pyannote at 0.4 has recall 0.150
   (misses ~85% of true overlap); lowering to 0.05 lifts recall to 0.262 and F1 to
   0.346. Overlaps are short relative to a 2 s window, so the coverage-based
   `overlap_score` is generally small and a high cutoff starves recall.
3. **The calibrated threshold sits at the sweep floor (0.05)** — the true optimum is
   probably lower. A finer low-end sweep is the obvious next step.
4. **Absolute F1 is modest (~0.35) and honest.** Accuracy looks high (0.893) only
   because most windows are non-overlap; F1 is the meaningful metric here.

## Caveats

- Only **2 test meetings** — treat absolute values as indicative, not final.
- Results are sensitive to `gt_fraction` (0.5 is strict) and `window_seconds` (2.0).
  A sensitivity sweep (e.g. `gt_fraction` 0.3, 1 s windows) is pending.
- Threshold hit the sweep boundary; re-run with a finer/lower grid before quoting a
  final routing threshold.

## Follow-up: stratified split + per-overlap-level buckets

The random split above unluckily put two low-overlap meetings in test. Re-run with
`analyze_scores.py` (stratified split by meeting overlap intensity, fine 0.01 sweep,
buckets by true overlap level) on the dumped scores:

- **Test now spans the range:** R8007_M8010 (56% overlap) + R8008_M8013 (14%).

| Detector | calibrated | test F1 | P | R |
| --- | --- | --- | --- | --- |
| **pyannote** | 0.06 | **0.591** | 0.757 | 0.485 |
| energy | 0.11 | 0.469 | 0.307 | 0.998 |

pyannote's test F1 rises from 0.346 (random split) to **0.591** under the stratified
split. Read this cautiously: with only 8 meetings / 2 test, and the stratified split
added *after* seeing the first result, **0.591 is a preliminary figure**, not proof
that the earlier number was pure split noise. It suggests the random split was
unfavourable, but a clean conclusion needs more meetings and a pre-registered split.

**Detection (flagged-high) rate by true overlap level — the decisive view:**

| True overlap level | pyannote | energy |
| --- | --- | --- |
| heavy (≥50%) | **0.485** | 0.998 |
| moderate (30–50%) | 0.242 | 1.000 |
| light (10–30%) | 0.086 | 1.000 |
| trace (0–10%) | 0.068 | 1.000 |
| **none (0%) → false alarm** | **0.007** | **1.000** |

1. **pyannote is a genuine detector:** detection rate increases monotonically with
   overlap severity (0.7% on clean windows → 48.5% on heavy overlap) and it barely
   false-alarms on clean audio (0.7%).
2. **energy is degenerate:** it flags ~everything (false-alarm rate 1.0). Its F1
   (0.469) looks close to pyannote only because this test set is overlap-heavy, so
   "label all positive" scores well. The buckets expose that it does not
   discriminate at all — which a single F1 number hides.
3. pyannote still misses ~half of even heavy overlap (recall headroom): OSD is
   conservative and a 2 s window dilutes short overlaps. Smaller windows or a
   different score aggregation are worth trying.

Scores are dumped once (`run_calibration.py --dump-scored`) and reused by
`analyze_scores.py`, so split/sweep/bucket changes need no pyannote re-run.

## Next steps

- Smaller windows / alternative score aggregation to lift recall on heavy overlap.
- Sensitivity analysis over `gt_fraction` and `window_seconds`.
- More test meetings (use AISHELL-4 Test or more of AliMeeting) for stable numbers.

## Reproduction environment

pyannote.audio 3.1.x needs an **early-2024-era torch stack**; newer versions break
(torchvision op mismatch, removed `torchaudio.set_audio_backend`, `hf_hub_download`
dropping `use_auth_token`, speechbrain's k2 lazy-import). This combination works on
Windows / Python 3.11 / CPU:

```
pyannote.audio==3.1.1
torch==2.2.2  torchaudio==2.2.2  torchvision==0.17.2
pytorch-lightning==2.1.4  torchmetrics==1.2.1
huggingface_hub==0.23.4
speechbrain==0.5.16
numpy<2            # 1.26.4
```

Plus an `HF_TOKEN` with the `pyannote/overlapped-speech-detection` and
`pyannote/segmentation` model terms accepted. OSD inference over the 8 far-field
meetings runs in tens of minutes on CPU.
