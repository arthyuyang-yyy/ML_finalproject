# Overlap Threshold Calibration — Results (preliminary)

First end-to-end run on real data. **Preliminary**: 8 meetings, 2 in the test
split, so absolute numbers are noisy. Raw audio/annotations are not committed
(see `../../data/README.md`); only metrics are recorded here.

> **Two phases.** **Phase 1** (below: *Setup* → *Follow-up*) is the *exploratory*
> run — fixed 2 s windows, pyannote OSD score, diarization fusion **disabled**. It
> does **not** match the real Pipeline, so its threshold does not carry over and was
> never a finished Step 7b. **Phase 2** ([jump](#phase-2--production-aligned-step-7b))
> is the *production-aligned* run that closes Step 7b: it uses the **silero VAD
> segments** and the **fused** overlap score (OSD + diarization overlap + speaker
> change) that `run_meeting_pipeline` actually routes on. Read Phase 2 for the
> recommended threshold and the production-readiness verdict; Phase 1 is kept for
> history and for the pyannote-vs-energy comparison.

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

## Phase 2 — production-aligned Step 7b

This is the run that actually closes Step 7b: it calibrates the routing threshold on
the **same inputs the Pipeline uses** — **silero VAD segments** scored with the
**fused** overlap signal (pyannote OSD + diarization overlap + speaker-change), i.e.
`run_calibration.py --vad-method silero --fuse-diarization --detectors pyannote`.
Unlike Phase 1, `--fuse-diarization` now **fails loud** if the diarization backend is
missing or its gated-model terms are unaccepted, so a degraded "OSD-only" score can
never masquerade as a fused one.

### Setup

- **Dataset:** AliMeeting Eval, far-field mix, 8 meetings (same as Phase 1).
- **Segmentation:** silero VAD (production-aligned), **2172 segments** total.
- **Score:** fused overlap score (OSD + diarization overlap + speaker change).
- **Label / split:** `gt_fraction = 0.5`; split by meeting, `test_ratio = 0.3`, `seed = 0`.
- Scores dumped to `pyannote_scored.json` and reused by `analyze_scores.py` (no re-run).

### Results (held-out test)

Two evaluations of the *same* dumped scores — the seed-0 random split and the
overlap-stratified split — bracket the threshold and the achievable F1:

| Split | Calibrated threshold | test F1 | test P / R | F1 @default (0.4) |
| --- | --- | --- | --- | --- |
| random (by meeting, seed 0) | **0.25** | 0.433 | 0.500 / 0.382 | 0.103 |
| stratified (by overlap level) | **0.22** | **0.779** | 0.661 / 0.950 | — |

At the default **0.4** the detector is near-useless: recall **0.055** (misses ~95% of
overlap), routing only 0.4% of segments to the expensive path. Lowering to **~0.22–0.25**
lifts test F1 **4–7×** (0.10 → 0.43–0.78) for a routing cost of ~6% of segments.

### Detection rate by true overlap level (stratified test, threshold 0.22)

| True overlap level | flagged-high (pyannote, fused) |
| --- | --- |
| heavy (≥50%) | **0.950** |
| moderate (30–50%) | 0.643 |
| light (10–30%) | 0.157 |
| trace (0–10%) | 0.000 |
| **none (0%) → false alarm** | **0.000** |

This is the decisive view: detection rises monotonically with severity, catches **95%
of genuinely heavy overlap**, and **false-alarms at 0.0%** on clean windows — the exact
shape a cost/quality router wants. (The energy fallback, by contrast, stays degenerate:
it flags ~everything, so it cannot be used as the routing signal.)

### Step 7b verdict — production readiness

**Conditionally ready.** The *approach* is production-deployable; the *exact* threshold
is a calibrated recommendation, not a hard-validated constant.

- ✅ **Detector & signal:** pyannote fused overlap is a genuine, well-behaved detector
  (monotonic severity response, ~0% false alarm). The energy fallback is degenerate and
  must **not** be the router. Use pyannote fused.
- ✅ **Threshold direction:** the default `0.4` is clearly wrong (recall ~5%). Adopting
  **≈0.25** as the new production default is well-justified and strictly better across
  both splits; 0.22–0.25 is the defensible band.
- ⚠️ **Statistical confidence:** only **8 meetings / 2 in test**, and test F1 swings
  0.43↔0.78 by split — so treat the number as a **v1 default to ship with monitoring**,
  not a final constant. Validate on more meetings (AISHELL-4 Test / more AliMeeting)
  before freezing it.
- ⚠️ **Recall headroom:** even at the calibrated threshold ~5% of heavy overlap and most
  light overlap are missed; acceptable for a cost-gated router, revisit if downstream
  needs higher overlap recall.

**Recommendation:** ship `route_threshold ≈ 0.25` (replacing `0.4`) with pyannote fused
scoring as the Step 7b production default, flagged as a calibrated v1 pending wider
validation.

## Next steps

- Validate the ≈0.25 threshold on more meetings before freezing it as a constant.
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
transformers==4.41.2   # pin: 5.x requires torch>=2.4 and silently disables PyTorch on this stack
numpy<2            # 1.26.4
```

> ⚠️ Pin `transformers` too. If it floats to 5.x it logs `Disabling PyTorch
> because PyTorch >= 2.4 is required but found 2.2.2` and turns off the torch
> backend. `4.41.2` is the floor that still satisfies `sentence-transformers`
> (used by retrieval) while keeping torch 2.2.x.

Plus an `HF_TOKEN` with the `pyannote/overlapped-speech-detection` and
`pyannote/segmentation` model terms accepted. OSD inference over the 8 far-field
meetings runs in tens of minutes on CPU.
