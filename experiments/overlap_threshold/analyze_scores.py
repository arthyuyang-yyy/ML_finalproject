"""Offline analysis of dumped overlap scores.

Reads ``<detector>_scored.json`` produced by ``run_calibration.py --dump-scored``
(each item: ``meeting_id`` / ``overlap_score`` / ``gt_fraction``), so it needs no
audio or pyannote and reruns instantly. It adds three things on top of the basic
calibration:

* **Stratified split** — train/test balanced by each meeting's overlap intensity,
  instead of a random split that can hand the test set the easy (low-overlap)
  meetings.
* **Fine threshold sweep** — a denser low-end grid, since overlap scores are small.
* **Per-overlap-level buckets** — the detector's flagged-high rate as a function of
  the *true* overlap severity, answering "how well is overlap detected in the
  genuinely high-overlap windows?"

Usage::

    python experiments/overlap_threshold/analyze_scores.py \
        --scores-dir outputs/overlap_threshold --gt-fraction 0.5 --step 0.01
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.core import HIGH_OVERLAP, LOW_OVERLAP  # noqa: E402
from experiments.overlap_threshold.run_calibration import (  # noqa: E402
    calibrate_threshold,
    default_thresholds,
    evaluate_at_threshold,
)

# Buckets over the ground-truth overlap fraction, as [lo, hi).
OVERLAP_BUCKETS: list[tuple[str, float, float]] = [
    ("heavy (>=50%)", 0.5, 1.01),
    ("moderate (30-50%)", 0.3, 0.5),
    ("light (10-30%)", 0.1, 0.3),
    ("trace (0-10%)", 1e-9, 0.1),
    ("none (0%)", -1.0, 1e-9),
]


def relabel(scored: list[dict[str, Any]], gt_fraction: float) -> list[dict[str, Any]]:
    """Derive the binary high/low label from the continuous ground-truth fraction."""
    return [
        {**item, "gt_label": HIGH_OVERLAP if float(item["gt_fraction"]) >= gt_fraction else LOW_OVERLAP}
        for item in scored
    ]


def meeting_overlap_ratios(scored: list[dict[str, Any]]) -> dict[str, float]:
    """Mean ground-truth overlap fraction per meeting (its overlap intensity)."""
    by_meeting: dict[str, list[float]] = {}
    for item in scored:
        by_meeting.setdefault(str(item["meeting_id"]), []).append(float(item["gt_fraction"]))
    return {meeting: sum(values) / len(values) for meeting, values in by_meeting.items()}


def stratified_split(
    meeting_ratios: dict[str, float],
    test_ratio: float = 0.3,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split meetings into (train, test) balanced by overlap intensity.

    Meetings are sorted by overlap ratio and divided into ``n_test`` contiguous
    strata; one meeting is drawn from each stratum, so the test set spans the
    overlap range instead of clustering at one end.
    """
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    ids = sorted(meeting_ratios, key=lambda m: (meeting_ratios[m], m))
    n = len(ids)
    if n < 2:
        return ids, []
    n_test = min(max(1, round(n * test_ratio)), n - 1)
    rng = random.Random(seed)
    test: list[str] = []
    for i in range(n_test):
        group = ids[i * n // n_test : (i + 1) * n // n_test] or [ids[min(i, n - 1)]]
        test.append(rng.choice(group))
    test_set = set(test)
    return sorted(m for m in ids if m not in test_set), sorted(test_set)


def flagged_rate_by_bucket(scored: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Per true-overlap bucket: fraction of windows the detector flags high-overlap.

    For overlap buckets this is detection sensitivity (recall by severity); for the
    ``none`` bucket it is the false-alarm rate on genuinely clean windows.
    """
    rows: list[dict[str, Any]] = []
    for name, lo, hi in OVERLAP_BUCKETS:
        items = [s for s in scored if lo <= float(s["gt_fraction"]) < hi]
        if not items:
            rows.append({"bucket": name, "n": 0, "flagged_high_rate": None})
            continue
        flagged = sum(1 for s in items if float(s["overlap_score"]) >= threshold)
        rows.append({"bucket": name, "n": len(items), "flagged_high_rate": round(flagged / len(items), 3)})
    return rows


def analyze_detector(
    scored: list[dict[str, Any]],
    gt_fraction: float,
    test_ratio: float,
    seed: int,
    step: float,
) -> dict[str, Any]:
    """Stratified split, fine calibration, and per-bucket sensitivity for one detector."""
    labeled = relabel(scored, gt_fraction)
    ratios = meeting_overlap_ratios(labeled)
    train_ids, test_ids = stratified_split(ratios, test_ratio, seed)
    train = [s for s in labeled if s["meeting_id"] in set(train_ids)]
    test = [s for s in labeled if s["meeting_id"] in set(test_ids)]

    calibration = calibrate_threshold(train, default_thresholds(step))
    best = calibration["best_threshold"]
    report: dict[str, Any] = {
        "meeting_overlap_ratios": {m: round(r, 3) for m, r in sorted(ratios.items())},
        "train_meetings": train_ids,
        "test_meetings": test_ids,
        "calibrated_threshold": best,
        "train_metrics_at_calibrated": calibration["best_metrics"],
    }
    if test:
        report["test_metrics_at_calibrated"] = evaluate_at_threshold(test, best)
        report["test_buckets_at_calibrated"] = flagged_rate_by_bucket(test, best)
    return report


def run(scores_dir: str | Path, gt_fraction: float, test_ratio: float, seed: int, step: float) -> dict[str, Any]:
    """Analyze every ``*_scored.json`` dump in ``scores_dir``."""
    dumps = sorted(Path(scores_dir).glob("*_scored.json"))
    if not dumps:
        raise FileNotFoundError(
            f"no *_scored.json in {scores_dir}; run run_calibration.py --dump-scored first"
        )
    results: dict[str, Any] = {
        "config": {"gt_fraction": gt_fraction, "test_ratio": test_ratio, "seed": seed, "step": step},
        "detectors": {},
    }
    for dump in dumps:
        detector = dump.stem.replace("_scored", "")
        scored = json.loads(dump.read_text(encoding="utf-8"))
        results["detectors"][detector] = analyze_detector(scored, gt_fraction, test_ratio, seed, step)
    return results


def _print_summary(results: dict[str, Any]) -> None:
    for detector, report in results["detectors"].items():
        print(f"\n=== {detector} ===")
        print(f"  overlap ratios: {report['meeting_overlap_ratios']}")
        print(f"  train: {report['train_meetings']}")
        print(f"  test : {report['test_meetings']}")
        print(f"  calibrated threshold = {report['calibrated_threshold']}")
        test_cal = report.get("test_metrics_at_calibrated")
        if test_cal:
            print(f"  test F1={test_cal['f1']:.3f}  P={test_cal['precision']:.3f}  R={test_cal['recall']:.3f}")
            print("  detection rate by true overlap level (test):")
            for row in report["test_buckets_at_calibrated"]:
                rate = "n/a" if row["flagged_high_rate"] is None else f"{row['flagged_high_rate']:.3f}"
                print(f"    {row['bucket']:18} n={row['n']:>4}  flagged-high={rate}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-dir", default="outputs/overlap_threshold")
    parser.add_argument("--gt-fraction", type=float, default=0.5)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=float, default=0.01, help="fine threshold sweep step")
    parser.add_argument("--out", default="outputs/overlap_threshold/analysis.json")
    args = parser.parse_args()

    results = run(args.scores_dir, args.gt_fraction, args.test_ratio, args.seed, args.step)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
