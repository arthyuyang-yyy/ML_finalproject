"""Per-run scorer for the experiments layer.

For each (cell × meeting) run directory, this script:
  1. reads `main.py` outputs and `data/alimeeting/annotations/<mid>.json` (GT)
  2. calls project existing functions in `src.evaluation` /
     `scripts.evaluate_alimeeting_result` to produce every metric
  3. inlines only the small filter / match steps those functions need
  4. writes `evaluation.json` into the same run directory

Nothing in `src/` or `scripts/` is touched. Idempotent (skips runs whose
`evaluation.json` already exists).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Make project root importable
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.core import (  # noqa: E402
    HIGH_OVERLAP,
    LOW_OVERLAP,
    character_error_rate,
    evaluate_evidence_support,
    evaluate_overlap_routing,
    speaker_attribution_accuracy,
    word_error_rate,
)

# Reuse the project evaluator's region / normalization helpers
from scripts.evaluate_alimeeting_result import (  # noqa: E402
    _covered_seconds_by_regions,
    _normalize_text,
    _speaker_report,
    _total_region_seconds,
)

# Reuse the project TextGrid parser (extract xmin/xmax/text per interval)
from scripts.prepare_alimeeting import (  # noqa: E402
    _INTERVAL_RE,
    _INTERVAL_TIER_RE,
    _NAME_RE,
)

# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """IoU over two half-open time intervals."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _window_contains(window: tuple[float, float] | None, ts: float, te: float) -> bool:
    """Return True if ``[ts, te)`` overlaps ``window``.

    ``window is None`` means "evaluate over the whole meeting" — this is the
    default; the previous ``(0, 300)`` window clipped every metric to the
    first five minutes and was responsible for WER saturating at 1.0 because
    the clipped reference had too few words.
    """
    if window is None:
        return True
    lo, hi = window
    return te > lo and ts < hi


def _clip_regions(
    regs: list[dict], window: tuple[float, float] | None
) -> list[tuple[float, float]]:
    """Clip regions to ``window``; pass-through when window is None."""
    if window is None:
        out: list[tuple[float, float]] = []
        for r in regs:
            s = float(r.get("start_time", r.get("start", 0.0)))
            e = float(r.get("end_time", r.get("end", 0.0)))
            if e > s:
                out.append((s, e))
        return out
    lo, hi = window
    out: list[tuple[float, float]] = []
    for r in regs:
        s = max(float(r.get("start_time", r.get("start", 0.0))), lo)
        e = min(float(r.get("end_time", r.get("end", 0.0))), hi)
        if e > s:
            out.append((s, e))
    return out


def _overlap_threshold_for(overlap_json: list[dict]) -> float:
    """Heuristic: threshold used in routing.  We pull it from the first segment
    that has a non-zero overlap_score; otherwise fall back to 0.5.
    """
    for seg in overlap_json:
        sc = seg.get("overlap_score")
        if isinstance(sc, (int, float)) and sc > 0:
            return 0.5
    return 0.5


def _strip_mic_suffix(meeting_id: str) -> str:
    # Eval_Ali_far recordings use a wide range of mic indices (MS801..MS810),
    # so strip any ``_MS\d{3}`` suffix rather than a fixed allow-list.
    if re.search(r"_MS\d{3}$", meeting_id):
        return re.sub(r"_MS\d{3}$", "", meeting_id)
    return meeting_id


def _parse_textgrid_intervals(text: str) -> list[dict]:
    """Return [{speaker, start_time, end_time, text}] for non-empty intervals.

    Mirrors ``parse_textgrid`` in ``scripts.prepare_alimeeting`` but keeps the
    interval text (that function only keeps regions).
    """
    out: list[dict] = []
    markers = [m.start() for m in _INTERVAL_TIER_RE.finditer(text)]
    for index, start in enumerate(markers):
        end = markers[index + 1] if index + 1 < len(markers) else len(text)
        block = text[start:end]
        name_match = _NAME_RE.search(block)
        speaker = name_match.group(1).strip() if name_match else f"tier_{index}"
        for interval in _INTERVAL_RE.finditer(block):
            xmin, xmax, content = interval.groups()
            content = content.strip()
            if not content:
                continue
            out.append({
                "speaker": speaker,
                "start_time": float(xmin),
                "end_time": float(xmax),
                "text": content,
            })
    out.sort(key=lambda i: (i["start_time"], i["end_time"]))
    return out


def _find_textgrid(rep: Path, base_meeting_id: str) -> Path | None:
    for sub in ("Eval_Ali_far", "Eval_Ali_near"):
        p = rep / "data" / "alimeeting" / "Eval_Ali" / sub / "textgrid_dir" / f"{base_meeting_id}.TextGrid"
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------------------
# metric blocks
# --------------------------------------------------------------------------------------


def _cer_block(
    evidence: list[dict], gt: dict, tg_intervals: list[dict],
    window: tuple[float, float] | None
) -> dict[str, Any]:
    """CER / WER overall + per routing-path subset.

    Reference text comes from the TextGrid intervals (the annotations JSON has
    only speaker + time, not text — see scripts/prepare_alimeeting.py).
    ``window=None`` evaluates the whole meeting.
    """

    def _text_concat(items: list[dict]) -> str:
        parts = [
            (it.get("text") or "").strip()
            for it in items
            if it.get("text") and _window_contains(
                window, float(it.get("start_time", 0.0)), float(it.get("end_time", it.get("start_time", 0.0) + 1e6))
            )
        ]
        return _normalize_text(" ".join(parts))

    def _ref_concat(predicate=None) -> str:
        out: list[str] = []
        for it in tg_intervals:
            ts, te = float(it["start_time"]), float(it["end_time"])
            if not _window_contains(window, ts, te):
                continue
            if predicate is not None and not predicate(it):
                continue
            txt = (it.get("text") or "").strip()
            if txt:
                out.append(txt)
        return _normalize_text(" ".join(out))

    def _subset(predicate) -> dict[str, Any]:
        hyp_items = [e for e in evidence if predicate(e)]
        hyp = _text_concat(hyp_items)
        ref = _ref_concat()
        cer = character_error_rate(ref, hyp) if ref else character_error_rate("", hyp)
        # WER on raw space-split Chinese text is meaningless: with no tokeniser
        # the entire ref collapses to a single token and edit_distance/1 == 1
        # whenever the strings differ by even one character. Only compute WER
        # when the reference contains at least 2 whitespace-separated tokens
        # (i.e. contains real word boundaries).
        ref_tokens = ref.split()
        if len(ref_tokens) >= 2:
            wer_dict = word_error_rate(ref, hyp)
            wer_val: float | None = wer_dict.get("error_rate", 0.0)
        else:
            wer_val = None
        # `character_error_rate` doesn't return hypothesis_length — count here
        hyp_chars = sum(1 for c in hyp if not c.isspace())
        ref_chars = cer.get("reference_length", 0)
        return {
            "cer": cer.get("error_rate", 0.0),
            "wer": wer_val,
            "subs": cer.get("substitutions", 0),
            "ins": cer.get("insertions", 0),
            "del": cer.get("deletions", 0),
            "ref_chars": ref_chars,
            "hyp_chars": hyp_chars,
            "ref_tokens": len(ref_tokens),
        }

    return {
        "concat": _subset(lambda _: True),
        "low": _subset(lambda e: e.get("processing_path") == "low_overlap_cluster"),
        "high": _subset(lambda e: e.get("processing_path") == "high_overlap_candidate"),
        "window": [window[0], window[1]] if window is not None else None,
    }


def _routing_block(
    evidence: list[dict], gt: dict, window: tuple[float, float] | None
) -> dict[str, Any]:
    """routing P/R/F1 against GT overlap labels."""
    gt_regs = _clip_regions(gt.get("overlap_regions", []), window)

    predictions: list[str] = []
    references: list[str] = []
    for ev in evidence:
        ts, te = float(ev.get("start_time", 0.0)), float(ev.get("end_time", 0.0))
        if not _window_contains(window, ts, te):
            continue
        hyp_path = ev.get("processing_path") or ""
        hyp = HIGH_OVERLAP if "high" in hyp_path else LOW_OVERLAP
        ref = LOW_OVERLAP
        for s, e in gt_regs:
            if ts < e and te > s:
                ref = HIGH_OVERLAP
                break
        predictions.append(hyp)
        references.append(ref)
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "support": 0}
    return evaluate_overlap_routing(predictions, references)


def _osd_block(
    routed: list[dict], gt: dict, window: tuple[float, float] | None, threshold: float
) -> dict[str, Any]:
    """OSD region-level recall/precision/F1 vs GT overlap_regions."""
    gt_regs = _clip_regions(gt.get("overlap_regions", []), window)
    hyp_regs: list[tuple[float, float]] = []
    for seg in routed:
        sc = seg.get("overlap_score") or 0.0
        if sc >= threshold:
            ts = float(seg.get("start_time", 0.0))
            te = float(seg.get("end_time", 0.0))
            if window is not None:
                ts = max(ts, window[0])
                te = min(te, window[1])
            if not _window_contains(window, ts, te):
                continue
            if te > ts:
                hyp_regs.append((ts, te))
    gt_total = _total_region_seconds(gt_regs)
    hyp_total = _total_region_seconds(hyp_regs)
    cov = _covered_seconds_by_regions(gt_regs, hyp_regs)
    rec = cov / gt_total if gt_total else 0.0
    prec = cov / hyp_total if hyp_total else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "gt_overlap_seconds": gt_total,
        "hyp_overlap_seconds": hyp_total,
        "threshold": threshold,
    }


def _speaker_block(
    evidence: list[dict], gt: dict, window: tuple[float, float] | None
) -> dict[str, Any]:
    """Best-mapping speaker attribution + known/unknown coverage.

    Aligns each evidence segment to the GT turn with maximum IoU, then calls
    `speaker_attribution_accuracy` (best-mapping brute force over permutations).
    Also calls `_speaker_report` (with the project's expected signature) for
    coverage numbers.
    """
    turns_in_win = [
        t for t in gt.get("turns", [])
        if _window_contains(window, float(t["start_time"]), float(t["end_time"]))
    ]
    ev_in_window = [
        e for e in evidence
        if _window_contains(window, float(e.get("start_time", 0.0)),
                            float(e.get("end_time", e.get("start_time", 0.0) + 1e6)))
    ]

    ref_speakers: list[str] = []
    hyp_speakers: list[str] = []
    for ev in ev_in_window:
        ts = float(ev.get("start_time", 0.0))
        te = float(ev.get("end_time", ts + 1e6))
        best_turn = None
        best_iou = 0.0
        for t in turns_in_win:
            iou = _iou((ts, te), (float(t["start_time"]), float(t["end_time"])))
            if iou > best_iou:
                best_iou = iou
                best_turn = t
        if best_turn is None or best_iou < 0.05:
            ref_speakers.append("UNK")
        else:
            ref_speakers.append(str(best_turn.get("speaker", "UNK")))
        hyp_speakers.append(str(ev.get("speaker", "UNK") or "UNK"))

    result: dict[str, Any] = {}
    if ref_speakers and hyp_speakers:
        result["best_mapping_accuracy"] = speaker_attribution_accuracy(
            ref_speakers, hyp_speakers
        )
    else:
        result["best_mapping_accuracy"] = {
            "accuracy": 0.0,
            "support": 0,
            "reference_speakers": 0,
            "hypothesis_speakers": 0,
        }

    # Project's coverage helper: (reference_turns, hyp_evidence, hyp_offset)
    report = (
        _speaker_report(turns_in_win, ev_in_window, 0.0) if ev_in_window else {}
    )
    result["known_speaker_coverage"] = report.get("known_speaker_coverage", 0.0)
    result["unknown_speaker_coverage"] = report.get("unknown_speaker_coverage", 0.0)
    return result


def _evidence_block(events: dict, evidence: list[dict]) -> dict[str, Any]:
    """Hallucination / unsupported rates via the project's evidence evaluator.

    No gold events exist, so references are empty; the meaningful metrics are
    `unsupported_claim_rate` and `hallucination_rate`.
    """
    source_universe = {ev.get("evidence_id", "") for ev in evidence if ev.get("evidence_id")}
    pred_events = events.get("events", []) if isinstance(events, dict) else []
    predictions = [
        {
            "evidence_ids": [str(eid) for eid in ev.get("evidence_ids", [])],
            "insufficient_evidence": False,
            "text": ev.get("description") or ev.get("summary") or ev.get("title", ""),
        }
        for ev in pred_events
    ]
    references = [{"evidence_ids": []} for _ in predictions]
    if not predictions:
        return {
            "events": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hit_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "hallucination_rate": 0.0,
        }
    out = evaluate_evidence_support(
        predictions, references, source_evidence_ids=source_universe
    )
    out["events"] = len(predictions)
    return out


def _audio_duration_s(
    evidence: list[dict], tg_intervals: list[dict], window: tuple[float, float] | None
) -> float:
    """Audio duration used for per-minute / RTF normalisation.

    When ``window`` is None we use the maximum evidence end-time (or TextGrid
    end-time, whichever is larger) so per-minute rates reflect the whole
    meeting rather than the (former) 5-minute window.
    """
    span = 0.0
    for e in evidence:
        end = float(e.get("end_time", 0.0))
        span = max(span, end)
    for it in tg_intervals:
        span = max(span, float(it.get("end_time", 0.0)))
    if window is not None:
        span = max(span, window[1])
    return max(span, 1e-6)


def _events_block(
    events: dict, evidence: list[dict], window: tuple[float, float] | None
) -> dict[str, Any]:
    if window is not None:
        lo, hi = window
        minutes = max((hi - lo) / 60.0, 1e-6)
        ev_in_window = [
            e for e in evidence if lo <= float(e.get("start_time", 0.0)) < hi
        ]
    else:
        minutes = max(_audio_duration_s(evidence, [], None) / 60.0, 1e-6)
        ev_in_window = list(evidence)
    pred_events = events.get("events", []) if isinstance(events, dict) else []
    n_ev = len(ev_in_window) or 1

    src_counts = {"llm_resolved": 0, "fallback_resolved": 0, "unresolved": 0, "other": 0}
    for ev in ev_in_window:
        s = str(ev.get("source", "")).strip().lower() or "other"
        src_counts[s if s in src_counts else "other"] += 1

    supported = 0
    for ev in pred_events:
        if any(
            eid in {e.get("evidence_id", "") for e in evidence}
            for eid in ev.get("evidence_ids", [])
        ):
            supported += 1
    return {
        "events": len(pred_events),
        "per_minute": len(pred_events) / minutes,
        "supported_event_rate": supported / len(pred_events) if pred_events else 0.0,
        "llm_resolved_rate": src_counts["llm_resolved"] / n_ev,
        "fallback_resolved_rate": src_counts["fallback_resolved"] / n_ev,
        "unresolved_rate": src_counts["unresolved"] / n_ev,
    }


def _timing_block(
    meta: dict, evidence: list[dict], tg_intervals: list[dict],
    window: tuple[float, float] | None
) -> dict[str, Any]:
    wall = float(meta.get("wall_time_s", 0.0))
    dur = max(_audio_duration_s(evidence, tg_intervals, window), 1e-6)
    return {"wall_time_s": wall, "rtf": wall / dur, "audio_duration_s": dur}


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def _resolve_outputs(run_dir: Path, meeting_id: str) -> tuple[Path, Path, Path, Path]:
    """Return (evidence_path, routed_path, overlap_path, events_path, meta_path).

    `run_one.py` invokes `main.py` with `cwd=run_dir`, and main.py writes its
    outputs to `outputs/<meeting_id>/` relative to that cwd. We try that first,
    then fall back to the flat layout.
    """
    nested = run_dir / "outputs" / meeting_id
    flat = run_dir
    primary = nested if (nested / "evidence_segments.json").exists() else flat
    meta = run_dir / "run_meta.json"  # always at run_dir root
    return (
        primary / "evidence_segments.json",
        primary / "routed_segments.json",
        primary / "overlap.json",
        primary / "meeting_events.json",
        meta,
    )


def evaluate_run(
    cell_id: str, meeting_id: str, run_dir: Path, gt_path: Path,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    evidence_p, routed_p, overlap_p, events_p, meta_p = _resolve_outputs(run_dir, meeting_id)
    evidence = _load(evidence_p)
    routed = _load(routed_p)
    overlap = _load(overlap_p)
    events = _load(events_p)
    meta = _load(meta_p) if meta_p.exists() else {}
    gt = _load(gt_path)

    # Reference text comes from the TextGrid (annotations JSON has no text)
    base_id = _strip_mic_suffix(meeting_id)
    tg_path = _find_textgrid(REPO, base_id)
    tg_intervals: list[dict] = []
    if tg_path is not None:
        tg_intervals = _parse_textgrid_intervals(tg_path.read_text(encoding="utf-8"))

    threshold = _overlap_threshold_for(overlap)
    if window is not None:
        ref_in_window = sum(
            1
            for it in tg_intervals
            if float(it["end_time"]) > window[0] and float(it["start_time"]) < window[1]
        )
    else:
        ref_in_window = len(tg_intervals)
    out = {
        "cell_id": cell_id,
        "meeting_id": meeting_id,
        "window": [window[0], window[1]] if window is not None else None,
        "textgrid_path": str(tg_path) if tg_path else None,
        "ref_intervals_in_window": ref_in_window,
        "cer": _cer_block(evidence, gt, tg_intervals, window),
        "routing": _routing_block(evidence, gt, window),
        "overlap": _osd_block(routed, gt, window, threshold),
        "speaker": _speaker_block(evidence, gt, window),
        "evidence": _evidence_block(events, evidence),
        "events": _events_block(events, evidence, window),
        "timing": _timing_block(meta, evidence, tg_intervals, window),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(REPO / "experiments" / "runs"))
    ap.add_argument(
        "--gt-dir",
        default=str(REPO / "data" / "alimeeting" / "annotations"),
        help="dir with <meeting_id>.json GT annotations",
    )
    ap.add_argument(
        "--window-start", type=float, default=None,
        help="start of evaluation window in seconds (default: full meeting)",
    )
    ap.add_argument(
        "--window-end", type=float, default=None,
        help="end of evaluation window in seconds (default: full meeting)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing evaluation.json")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    gt_dir = Path(args.gt_dir)
    if args.window_start is None and args.window_end is None:
        window: tuple[float, float] | None = None
    else:
        lo = 0.0 if args.window_start is None else args.window_start
        hi = float("inf") if args.window_end is None else args.window_end
        window = (lo, hi)

    if not runs_root.is_dir():
        print(f"runs root not found: {runs_root}", file=sys.stderr)
        return 1

    # `run_one.py` invokes `main.py` with `cwd=out_dir`, and main.py writes
    # its outputs to `outputs/<meeting_id>/` relative to that cwd. So the
    # actual layout is:
    #   runs/<cell>/<meeting>/
    #     run_meta.json                <- we wrote this
    #     outputs/<meeting>/*.json     <- main.py's outputs
    targets = []
    for cell_dir in sorted(p for p in runs_root.glob("*") if p.is_dir()):
        for meeting_dir in sorted(p for p in cell_dir.glob("*") if p.is_dir()):
            for candidate in (
                meeting_dir / "outputs" / meeting_dir.name / "evidence_segments.json",
                meeting_dir / "evidence_segments.json",
            ):
                if candidate.exists():
                    targets.append((cell_dir.name, meeting_dir.name, meeting_dir))
                    break
    if not targets:
        print(f"no completed runs under {runs_root}", file=sys.stderr)
        return 1

    n_done = n_skip = n_err = 0
    for cell_id, meeting_id, run_dir in targets:
        out_path = run_dir / "evaluation.json"
        if out_path.exists() and not args.force:
            n_skip += 1
            continue
        # GT files in data/alimeeting/annotations/ use the form
        # ``R8001_M8004.json``; meeting_ids coming out of ``main.py`` add the
        # mic suffix ``_MS\d{3}``. Strip it (any mic index) to find the GT.
        gt_meeting_id = _strip_mic_suffix(meeting_id)
        gt_path = gt_dir / f"{gt_meeting_id}.json"
        if not gt_path.exists():
            print(f"  ! no GT for {meeting_id} (looked for {gt_path.name}), skipping", file=sys.stderr)
            n_err += 1
            continue
        t0 = time.time()
        try:
            res = evaluate_run(cell_id, meeting_id, run_dir, gt_path, window)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {cell_id}/{meeting_id} failed: {exc}", file=sys.stderr)
            n_err += 1
            continue
        _dump(out_path, res)
        n_done += 1
        print(
            f"  - {cell_id}/{meeting_id}  "
            f"cer={res['cer']['concat']['cer']:.3f}  "
            f"routing_f1={res['routing']['f1']:.3f}  "
            f"overlap_f1={res['overlap']['f1']:.3f}  "
            f"spk_acc={res['speaker']['best_mapping_accuracy'].get('accuracy', 0.0):.3f}  "
            f"rtf={res['timing']['rtf']:.3f}  "
            f"[{time.time()-t0:.1f}s]"
        )
    print(f"\ndone={n_done} skip={n_skip} err={n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
