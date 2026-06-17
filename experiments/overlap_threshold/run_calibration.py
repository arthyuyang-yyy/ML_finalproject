"""Phase 2 / Step 7b: overlap routing-threshold calibration with train/test split.

What this does
--------------
1. Reads prepared AliMeeting annotations (run ``scripts/prepare_alimeeting.py``
   first) to get per-meeting ground-truth overlap regions.
2. For each meeting, segments the audio (VAD) and scores every segment with an
   overlap detector — ``pyannote`` (the real overlapped-speech model) or the
   ``energy`` fallback baseline.
3. Labels each segment as truly high-overlap when ground-truth overlap covers at
   least ``gt_fraction`` of its duration (the label definition, fixed/reported).
4. **Splits meetings (not segments) into train and test** so segments from one
   meeting never straddle the split.
5. **Calibrates** the routing threshold on the *train* meetings (the threshold
   that maximizes F1), then **reports** accuracy / precision / recall / F1 on the
   held-out *test* meetings — both at the calibrated threshold and at the current
   default (0.4).

The threshold being calibrated (``route_threshold``) is distinct from the label
definition (``gt_fraction``); only the former is learned from data.

Pure functions (labeling, splitting, sweeping, evaluation) are dependency-free
and unit-tested. Audio loading and pyannote inference live in the I/O layer and
require the optional backends + ``HF_TOKEN`` (see the main README).

Usage::

    export ALIMEETING_ROOT=/path/to/Eval_Ali
    python scripts/prepare_alimeeting.py
    python experiments/overlap_threshold/run_calibration.py \
        --annotations data/alimeeting/annotations \
        --audio-dir data/alimeeting/audio
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.core import HIGH_OVERLAP, LOW_OVERLAP, evaluate_overlap_routing  # noqa: E402
from src.overlap.detector import DEFAULT_OVERLAP_THRESHOLD  # noqa: E402

Region = tuple[float, float]
LabeledSegment = dict[str, Any]


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested, no audio / no pyannote)
# --------------------------------------------------------------------------- #
def segment_overlap_fraction(segment: dict[str, Any], regions: list[Region]) -> float:
    """Fraction of a segment's duration covered by ground-truth overlap regions."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    duration = max(0.0, end - start)
    if duration == 0.0:
        return 0.0
    covered = 0.0
    for region_start, region_end in regions:
        lo = max(start, float(region_start))
        hi = min(end, float(region_end))
        if hi > lo:
            covered += hi - lo
    return min(1.0, covered / duration)


def label_segment(segment: dict[str, Any], regions: list[Region], gt_fraction: float) -> str:
    """``HIGH_OVERLAP`` when ground-truth overlap covers >= ``gt_fraction``."""
    return HIGH_OVERLAP if segment_overlap_fraction(segment, regions) >= gt_fraction else LOW_OVERLAP


def split_meetings(
    meeting_ids: list[str],
    test_ratio: float = 0.3,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split unique meeting IDs into (train, test) by meeting, deterministically.

    Splitting by meeting (not by segment) prevents leakage between correlated
    segments of the same meeting. With fewer than two meetings everything goes to
    train and test is empty (calibration is then not meaningfully held out).
    """
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    ids = sorted(set(meeting_ids))
    if len(ids) < 2:
        return ids, []
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_ratio))
    n_test = min(n_test, len(shuffled) - 1)  # keep at least one train meeting
    test = set(shuffled[:n_test])
    return sorted(i for i in ids if i not in test), sorted(test)


def evaluate_at_threshold(labeled: list[LabeledSegment], threshold: float) -> dict[str, Any]:
    """Route every segment at ``threshold`` and score against ground-truth labels.

    Also reports ``routing_cost``: the fraction of segments routed to the
    expensive high-overlap (multi-candidate) path at this threshold. A lower
    threshold catches more overlap (higher recall) but routes more segments to
    the costly path — this is the cost side of the cost/quality trade-off.
    """
    predictions = [
        HIGH_OVERLAP if float(item["overlap_score"]) >= threshold else LOW_OVERLAP
        for item in labeled
    ]
    references = [item["gt_label"] for item in labeled]
    metrics = evaluate_overlap_routing(predictions, references)
    routing_cost = (
        sum(1 for prediction in predictions if prediction == HIGH_OVERLAP) / len(predictions)
        if predictions
        else 0.0
    )
    return {**metrics, "routing_cost": round(routing_cost, 4)}


def default_thresholds(step: float = 0.05) -> list[float]:
    """Candidate routing thresholds in (0, 1), e.g. 0.05, 0.10, ... 0.95."""
    if not 0.0 < step < 1.0:
        raise ValueError("step must be in (0, 1)")
    count = int(round(1.0 / step))
    return [round(step * i, 4) for i in range(1, count)]


def calibrate_threshold(
    labeled: list[LabeledSegment],
    thresholds: list[float] | None = None,
    metric: str = "f1",
) -> dict[str, Any]:
    """Pick the threshold maximizing ``metric`` (tie-broken by accuracy) on ``labeled``."""
    if not labeled:
        raise ValueError("cannot calibrate on an empty segment set")
    candidates = thresholds or default_thresholds()
    sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key = (-1.0, -1.0)
    for threshold in candidates:
        metrics = evaluate_at_threshold(labeled, threshold)
        row = {"threshold": threshold, **metrics}
        sweep.append(row)
        key = (float(metrics[metric]), float(metrics["accuracy"]))
        if key > best_key:
            best_key = key
            best = row
    assert best is not None
    return {"best_threshold": best["threshold"], "best_metrics": best, "sweep": sweep}


def window_segments(
    duration: float,
    window_seconds: float,
    meeting_id: str = "meeting",
) -> list[dict[str, Any]]:
    """Tile ``[0, duration]`` into fixed-length windows.

    Fixed windows are the segmentation unit for overlap calibration: they match
    the literature's frame-based overlap evaluation and, unlike energy VAD, stay
    robust on long far-field meeting recordings (whose loud transients can starve
    a peak-relative VAD into producing no segments at all).
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    segments: list[dict[str, Any]] = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(start + window_seconds, duration)
        if end > start:
            segments.append({
                "meeting_id": meeting_id,
                "segment_id": f"{meeting_id}_w{index + 1:04d}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
            })
        start = end
        index += 1
    return segments


# --------------------------------------------------------------------------- #
# I/O layer (audio + pyannote; not unit-tested against real data)
# --------------------------------------------------------------------------- #
def score_meeting_segments(
    audio_path: str | Path,
    detector: str,
    window_seconds: float = 2.0,
    vad_method: str = "fixed",
    fuse_diarization: bool = False,
) -> list[dict[str, Any]]:
    """Segment one meeting's audio and attach an ``overlap_score`` per segment.

    ``vad_method`` selects the segmentation unit:

    - ``"fixed"`` (default): tile into ``window_seconds`` windows — matches the
      exploratory experiment and stays robust on long far-field recordings;
    - ``"energy"``: the project's energy VAD;
    - ``"silero"``: the faster-whisper silero VAD (production-aligned; requires
      the ``segment_waveform(method="silero")`` backend from PR #52).

    ``fuse_diarization`` mirrors the production pipeline: when set, pyannote
    diarization turns are fed into the scorer so the result is the **fused**
    overlap score (OSD + diarization overlap + speaker change), not the detector
    signal alone. It needs ``pyannote.audio`` + ``HF_TOKEN``; without them the
    diarization source returns nothing and the score degrades to OSD/energy only.

    ``detector='pyannote'`` runs the real OSD model; ``detector='energy'`` forces
    the energy fallback.
    """
    from src.audio.preprocess import load_audio, segment_waveform
    from src.overlap.detector import estimate_segment_overlap_scores

    samples, sample_rate = load_audio(str(audio_path))
    meeting_id = Path(audio_path).stem
    if vad_method == "silero":
        segments = segment_waveform(samples, sample_rate, meeting_id=meeting_id, method="silero")
    elif vad_method == "energy":
        segments = segment_waveform(samples, sample_rate, meeting_id=meeting_id)
    else:  # "fixed"
        duration = len(samples) / sample_rate if sample_rate else 0.0
        segments = window_segments(duration, window_seconds, meeting_id=meeting_id)
    if not segments:
        return []
    diarization_turns = _diarization_turns(audio_path) if fuse_diarization else None
    use_audio_path = str(audio_path) if detector == "pyannote" else None
    scored = estimate_segment_overlap_scores(
        samples, segments, sample_rate, audio_path=use_audio_path, diarization_turns=diarization_turns
    )
    if detector == "pyannote" and scored and scored[0].get("overlap_detector") != "pyannote":
        raise RuntimeError(
            "pyannote detector requested but the scorer fell back to "
            f"'{scored[0].get('overlap_detector')}'. Install pyannote.audio and set HF_TOKEN "
            "(see the Enable pyannote steps in the README)."
        )
    return scored


def _diarization_turns(audio_path: str | Path) -> list[dict[str, Any]] | None:
    """Pyannote speaker turns for fusion, or ``None`` when diarization is off.

    Returns ``None`` only when diarization is genuinely disabled (no
    ``HF_TOKEN``) — the fused score then degrades to OSD/energy only. But when
    ``--fuse-diarization`` was explicitly requested and the backend is *present
    yet failing* (missing ``pyannote.audio``, or gated model terms not accepted),
    silently degrading would emit misleading "fake fusion" numbers. We instead
    fail loud with an actionable message so the calibration is not trusted by
    mistake.
    """
    from src.diarization.core import diarize_with_pyannote
    from src.errors import BackendExecutionError, BackendUnavailableError

    try:
        return diarize_with_pyannote(str(audio_path)) or None
    except BackendUnavailableError as exc:
        raise RuntimeError(
            "--fuse-diarization needs pyannote.audio, which is not installed. "
            "Install it (see the README) or drop --fuse-diarization to calibrate on the OSD-only score."
        ) from exc
    except BackendExecutionError as exc:
        raise RuntimeError(
            "--fuse-diarization could not load/run the pyannote diarization model. "
            "The usual cause is unaccepted gated-model terms: log in to Hugging Face with the account "
            "that owns your HF_TOKEN and accept the conditions at "
            "https://hf.co/pyannote/speaker-diarization-3.1 and https://hf.co/pyannote/segmentation-3.0 . "
            "Then re-run in a fresh shell. Drop --fuse-diarization to calibrate on the OSD-only score instead."
        ) from exc


def load_annotation_records(annotations_dir: str | Path) -> list[dict[str, Any]]:
    """Load every prepared per-meeting annotation JSON."""
    paths = sorted(Path(annotations_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no prepared annotations in {annotations_dir}; run scripts/prepare_alimeeting.py first"
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def audio_matches_meeting(stem: str, meeting_id: str) -> bool:
    """Whether an audio file stem belongs to a meeting.

    AliMeeting audio appends a mic/speaker suffix to the meeting id used by the
    TextGrid — e.g. far ``R8001_M8004_MS801`` or near ``R8001_M8004_N_SPK8013``
    for meeting ``R8001_M8004`` — so match the id exactly or as a ``<id>_*``
    prefix (the trailing underscore prevents ``R8001_M8004`` matching
    ``R8001_M80040``).
    """
    return stem == meeting_id or stem.startswith(f"{meeting_id}_")


def make_audio_lookup(audio_dir: str | Path) -> Callable[[str], Path | None]:
    """Return a function mapping a meeting_id to its audio file under ``audio_dir``.

    Point ``audio_dir`` at the far-field ``audio_dir`` (one mixed file per
    meeting) for overlap detection; the per-speaker near-field files would
    otherwise match the same meeting id several times.
    """
    root = Path(audio_dir)

    def lookup(meeting_id: str) -> Path | None:
        candidates = sorted(
            path
            for pattern in ("*.wav", "*.flac")
            for path in root.rglob(pattern)
            if audio_matches_meeting(path.stem, meeting_id)
        )
        return candidates[0] if candidates else None

    return lookup


def build_labeled_dataset(
    records: list[dict[str, Any]],
    audio_lookup: Callable[[str], Path | None],
    detector: str,
    gt_fraction: float,
    window_seconds: float = 2.0,
    vad_method: str = "fixed",
    fuse_diarization: bool = False,
) -> list[LabeledSegment]:
    """Score and ground-truth-label every segment across all meetings."""
    labeled: list[LabeledSegment] = []
    for record in records:
        meeting_id = str(record["meeting_id"])
        audio_path = audio_lookup(meeting_id)
        if audio_path is None:
            print(f"  [skip] no audio found for meeting '{meeting_id}'")
            continue
        regions = [
            (float(region["start_time"]), float(region["end_time"]))
            for region in record.get("overlap_regions", [])
        ]
        meeting_segments = score_meeting_segments(
            audio_path, detector, window_seconds, vad_method, fuse_diarization
        )
        print(f"  {meeting_id}: {len(meeting_segments)} segments")
        for segment in meeting_segments:
            true_fraction = segment_overlap_fraction(segment, regions)
            labeled.append({
                "meeting_id": meeting_id,
                "overlap_score": float(segment["overlap_score"]),
                "gt_fraction": round(true_fraction, 4),
                "gt_label": HIGH_OVERLAP if true_fraction >= gt_fraction else LOW_OVERLAP,
            })
    return labeled


def calibrate_and_evaluate(
    labeled: list[LabeledSegment],
    train_ids: list[str],
    test_ids: list[str],
    thresholds: list[float] | None = None,
    default_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    """Calibrate on train meetings, report on held-out test meetings."""
    train = [item for item in labeled if item["meeting_id"] in set(train_ids)]
    test = [item for item in labeled if item["meeting_id"] in set(test_ids)]
    if not train:
        raise ValueError("no training segments after the split")

    calibration = calibrate_threshold(train, thresholds)
    best_threshold = calibration["best_threshold"]
    report: dict[str, Any] = {
        "num_train_segments": len(train),
        "num_test_segments": len(test),
        "calibrated_threshold": best_threshold,
        "default_threshold": default_threshold,
        "train_metrics_at_calibrated": calibration["best_metrics"],
        "sweep": calibration["sweep"],
    }
    if test:
        report["test_metrics_at_calibrated"] = evaluate_at_threshold(test, best_threshold)
        report["test_metrics_at_default"] = evaluate_at_threshold(test, default_threshold)
    return report


def run(
    annotations_dir: str | Path,
    audio_dir: str | Path,
    detectors: tuple[str, ...] = ("pyannote", "energy"),
    test_ratio: float = 0.3,
    seed: int = 0,
    gt_fraction: float = 0.5,
    window_seconds: float = 2.0,
    thresholds: list[float] | None = None,
    dump_dir: str | Path | None = None,
    vad_method: str = "fixed",
    fuse_diarization: bool = False,
) -> dict[str, Any]:
    """Run threshold calibration for each detector and return a combined report.

    When ``dump_dir`` is set, each detector's per-window scored dataset
    (``meeting_id``/``overlap_score``/``gt_fraction``) is written there so the
    expensive detector pass can be reused for offline analysis (stratified split,
    finer sweeps, per-overlap-level buckets) without re-running the model.

    ``vad_method`` and ``fuse_diarization`` align the run with the production
    pipeline (silero VAD segments + fused overlap score) for a proper Step 7b
    calibration, versus the default exploratory fixed-window / OSD-only setup.
    """
    records = load_annotation_records(annotations_dir)
    audio_lookup = make_audio_lookup(audio_dir)
    meeting_ids = [str(record["meeting_id"]) for record in records]
    train_ids, test_ids = split_meetings(meeting_ids, test_ratio, seed)

    results: dict[str, Any] = {
        "config": {
            "detectors": list(detectors),
            "test_ratio": test_ratio,
            "seed": seed,
            "gt_fraction": gt_fraction,
            "window_seconds": window_seconds,
            "vad_method": vad_method,
            "fuse_diarization": fuse_diarization,
            "num_meetings": len(set(meeting_ids)),
        },
        "train_meetings": train_ids,
        "test_meetings": test_ids,
        "detectors": {},
    }
    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
    for detector in detectors:
        print(f"\n=== detector: {detector} ===")
        labeled = build_labeled_dataset(
            records, audio_lookup, detector, gt_fraction, window_seconds, vad_method, fuse_diarization
        )
        if not labeled:
            print(f"  [warn] no labeled segments for detector '{detector}'")
            continue
        if dump_dir is not None:
            dump_path = Path(dump_dir) / f"{detector}_scored.json"
            dump_path.write_text(json.dumps(labeled, ensure_ascii=False), encoding="utf-8")
            print(f"  dumped {len(labeled)} scored windows -> {dump_path}")
        results["detectors"][detector] = calibrate_and_evaluate(
            labeled, train_ids, test_ids, thresholds
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", default="data/alimeeting/annotations",
                        help="directory of prepared per-meeting annotation JSON")
    parser.add_argument("--audio-dir", default="data/alimeeting/audio",
                        help="directory containing the meeting audio files")
    parser.add_argument("--detectors", nargs="+", default=["pyannote", "energy"],
                        choices=["pyannote", "energy"], help="overlap detectors to calibrate")
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gt-fraction", type=float, default=0.5,
                        help="ground-truth overlap coverage to label a segment high-overlap")
    parser.add_argument("--window-seconds", type=float, default=2.0,
                        help="fixed window length when --vad-method=fixed")
    parser.add_argument("--vad-method", default="fixed", choices=["fixed", "energy", "silero"],
                        help="segmentation: fixed windows (default), energy VAD, or silero VAD "
                             "(silero needs the segment_waveform method from PR #52)")
    parser.add_argument("--fuse-diarization", action="store_true",
                        help="feed pyannote diarization into the scorer for the production-aligned "
                             "fused overlap score (needs pyannote.audio + HF_TOKEN)")
    parser.add_argument("--out-dir", default="outputs/overlap_threshold",
                        help="where to write results.json (git-ignored by default)")
    parser.add_argument("--dump-scored", action="store_true",
                        help="also dump per-window scores to out-dir for offline analysis")
    args = parser.parse_args()

    if args.vad_method != "fixed" and args.window_seconds != parser.get_default("window_seconds"):
        print(f"[warn] --window-seconds={args.window_seconds} is ignored when "
              f"--vad-method={args.vad_method}; it only applies to --vad-method=fixed.")

    results = run(
        annotations_dir=args.annotations,
        audio_dir=args.audio_dir,
        detectors=tuple(args.detectors),
        test_ratio=args.test_ratio,
        seed=args.seed,
        gt_fraction=args.gt_fraction,
        window_seconds=args.window_seconds,
        dump_dir=args.out_dir if args.dump_scored else None,
        vad_method=args.vad_method,
        fuse_diarization=args.fuse_diarization,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ summary ================")
    for detector, report in results["detectors"].items():
        cal = report["calibrated_threshold"]
        print(f"[{detector}] calibrated threshold = {cal} "
              f"(default {report['default_threshold']})")
        test_cal = report.get("test_metrics_at_calibrated")
        test_def = report.get("test_metrics_at_default")
        if test_cal and test_def:
            print(f"    test F1  @calibrated={test_cal['f1']:.3f}  @default={test_def['f1']:.3f}")
            print(f"    test P/R @calibrated={test_cal['precision']:.3f}/{test_cal['recall']:.3f}")
        else:
            print("    (no held-out test meetings; need >= 2 meetings)")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
