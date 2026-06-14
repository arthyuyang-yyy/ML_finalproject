# Overlap Threshold Calibration — Results (preliminary)

First end-to-end run on real data. **Preliminary**: 8 meetings, 2 in the test
split, so absolute numbers are noisy. Raw audio/annotations are not committed
(see `../../data/README.md`); only metrics are recorded here.

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

## Next steps

- Finer low-end threshold sweep (0.01–0.10).
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
