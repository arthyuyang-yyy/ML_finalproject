# Overlap Routing-Threshold Calibration (Phase 2 / Step 7b)

Calibrate the dual-path routing threshold against human-annotated overlap, with a
proper **train/test split**, and report it honestly for both the real detector and
the fallback baseline.

## What it answers

- What overlap-score threshold best separates high- vs low-overlap segments?
- Does the calibrated threshold beat the current default (`0.4`) on held-out data?
- How far behind is the dependency-free **energy** fallback vs **pyannote** OSD?

## Two thresholds — don't conflate them

| Name | Role | Set how |
| --- | --- | --- |
| `gt_fraction` | **Label definition.** A segment is "truly high-overlap" when ground-truth overlap covers ≥ this fraction of it. | Fixed and reported (default `0.5`). |
| `route_threshold` | **What we calibrate.** The `overlap_score` cutoff used to route a segment. | Learned on the train split (max F1). |

Only `route_threshold` is learned from data. `gt_fraction` defines the answer key.

## Train/test split (why it matters)

The split is **by meeting, not by segment**, so correlated segments of one meeting
never appear on both sides. The threshold is chosen on the **train** meetings and
all reported numbers come from the **held-out test** meetings — otherwise the
metrics would be optimistic ("scoring high on questions you studied").

## Segmentation

Routing units are **fixed-length windows** (`--window-seconds`, default `2.0`),
matching the literature's frame-based overlap evaluation. This is deliberate: the
project's energy VAD produced *zero* segments on real far-field AliMeeting audio
(loud transients starve its peak-relative threshold), so VAD is not a reliable
unit here. Pass `--window-seconds 0` to fall back to VAD.

## Detector under calibration

`pyannote` OSD is the real detector and the one whose threshold matters for
deployment (needs `pip install pyannote.audio` + `HF_TOKEN`; see the main README).
The `energy` fallback (capped at 0.39) is run only as an honest baseline and is
expected to route nothing to the high-overlap path.

> The fusion with diarization turns is **disabled** here, so the score is the
> detector's own signal — not contaminated by ground-truth speaker timing.

## Run it

```bash
# 1. Prepare ground-truth annotations (see ../../data/README.md).
export ALIMEETING_ROOT=/path/to/Eval_Ali
python scripts/prepare_alimeeting.py

# 2. Calibrate (writes outputs/overlap_threshold/results.json, git-ignored).
python experiments/overlap_threshold/run_calibration.py \
    --annotations data/alimeeting/annotations \
    --audio-dir data/alimeeting/audio \
    --detectors pyannote energy \
    --test-ratio 0.3 --seed 0 --gt-fraction 0.5
```

Needs ≥ 2 meetings for a non-empty test split. Metric implementations are reused
from `src/evaluation/core.py` (`evaluate_overlap_routing`); the labeling, split,
and sweep logic are unit-tested in `tests/test_overlap_calibration.py` (no audio
or pyannote required).

## Honest scope

- The TextGrid → ground-truth pipeline is validated on synthetic fixtures; verify
  on a real `Eval_Ali` file before trusting full numbers.
- Results generalize to deployment only if the calibration corpus resembles the
  target meetings (Chinese multi-party here) **and** pyannote stays the detector.
  Swapping detectors or domains requires recalibration.
