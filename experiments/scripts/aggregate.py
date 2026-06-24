"""Aggregate per-run evaluation.json files into per-(cell, meeting) tables.

Two outputs:

1. ``summary_per_cell_meeting.csv`` (and .md) — one row per (cell, meeting).
   Each row carries the cell config (asr / osd / resolver / separation), the
   meeting id, and the flattened metric bundle from ``evaluation.json``.

2. ``meeting_difficulty.csv`` — one row per meeting, derived purely from the
   ground-truth annotations + audio duration. Used to explain why a given
   meeting shows higher CER or lower diarization accuracy.

The script never re-runs the pipeline; it only reads artefacts that
``run_one.py`` already produced. Re-running it after partial progress is
therefore safe and cheap.

Usage:
    python aggregate.py                     # default paths
    python aggregate.py --runs-root ...     # custom location
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO / "experiments" / "runs"
DEFAULT_DATA_ROOT = REPO / "data" / "alimeeting"
DEFAULT_OUT_DIR = REPO / "experiments" / "results"


# ---- 4D matrix cell labels (kept in sync with build_matrix.py) --------------

CELL_AXES: dict[str, list[str]] = {
    "asr": ["faster-whisper", "funasr", "whisperx"],
    "osd": ["pyannote", "energy_fallback"],
    "resolver": ["none", "openai"],
    "separation": ["none", "sepformer"],
}


def cell_id_from_config(cell: dict[str, Any]) -> str:
    """Stable, sortable id from a cell config dict."""
    return (
        f"asr={cell.get('asr', 'mock')}"
        f"_osd={cell.get('osd', 'pyannote')}"
        f"_resolver={cell.get('resolver', 'none')}"
        f"_sep={cell.get('separation', 'none')}"
    )


# ---- Per-run loading --------------------------------------------------------


@dataclass
class RunRecord:
    cell_id: str
    meeting_id: str
    cell: dict[str, Any]
    eval: dict[str, Any]
    meta: dict[str, Any]
    run_dir: Path


def discover_runs(runs_root: Path) -> Iterable[RunRecord]:
    """Yield one RunRecord per completed run found under ``runs_root``.

    A run is considered complete when both ``run_meta.json`` (with non-null
    ``exit_code``) and ``evaluation.json`` are present. Partial runs (e.g.
    crashed mid-pipeline) are skipped so the aggregate only reflects healthy
    data; the matrix runner re-runs them on the next sweep.
    """
    if not runs_root.exists():
        return
    for cell_dir in sorted(runs_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        for meeting_dir in sorted(cell_dir.iterdir()):
            if not meeting_dir.is_dir():
                continue
            # Skip backup / hidden directories so re-runs don't double-count.
            if meeting_dir.name.startswith(".") or meeting_dir.name.endswith(".bak"):
                continue
            meta_p = meeting_dir / "run_meta.json"
            eval_p = meeting_dir / "evaluation.json"
            if not meta_p.exists() or not eval_p.exists():
                continue
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                evald = json.loads(eval_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if meta.get("exit_code") != 0:
                continue
            yield RunRecord(
                cell_id=cell_dir.name,
                meeting_id=meeting_dir.name,
                cell=meta.get("config", {}),
                eval=evald,
                meta=meta,
                run_dir=meeting_dir,
            )


# ---- Metric flattening ------------------------------------------------------


def _g(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safe nested dict lookup."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_evaluation(rec: RunRecord) -> dict[str, Any]:
    """Flatten ``evaluation.json`` into a single CSV-friendly row."""
    ev = rec.eval
    cer = ev.get("cer", {})
    routing = ev.get("routing", {})
    overlap = ev.get("overlap", {})
    speaker = ev.get("speaker", {})
    evidence = ev.get("evidence", {})
    events = ev.get("events", {})
    timing = ev.get("timing", {})
    return {
        "cell_id": rec.cell_id,
        "meeting_id": rec.meeting_id,
        "asr": rec.cell.get("asr", ""),
        "osd": rec.cell.get("osd", ""),
        "resolver": rec.cell.get("resolver", ""),
        "separation": rec.cell.get("separation", ""),
        "language": rec.cell.get("language", ""),
        "asr_device": rec.cell.get("asr_device", ""),
        "cer_concat": _g(cer, "concat", "cer"),
        "cer_low": _g(cer, "low", "cer"),
        "cer_high": _g(cer, "high", "cer"),
        "wer_concat": _g(cer, "concat", "wer"),
        "ref_chars": _g(cer, "concat", "ref_chars"),
        "hyp_chars": _g(cer, "concat", "hyp_chars"),
        "subs": _g(cer, "concat", "subs"),
        "ins": _g(cer, "concat", "ins"),
        "del": _g(cer, "concat", "del"),
        "spk_best_mapping_acc": _g(speaker, "best_mapping_accuracy", "accuracy"),
        "spk_known_coverage": _g(speaker, "known_speaker_coverage"),
        "spk_unknown_coverage": _g(speaker, "unknown_speaker_coverage"),
        "spk_ref_n": _g(speaker, "best_mapping_accuracy", "reference_speakers"),
        "spk_hyp_n": _g(speaker, "best_mapping_accuracy", "hypothesis_speakers"),
        "routing_accuracy": _g(routing, "accuracy"),
        "routing_f1": _g(routing, "f1"),
        "routing_support": _g(routing, "support"),
        "overlap_recall": _g(overlap, "recall"),
        "overlap_precision": _g(overlap, "precision"),
        "overlap_f1": _g(overlap, "f1"),
        "gt_overlap_s": _g(overlap, "gt_overlap_seconds"),
        "hyp_overlap_s": _g(overlap, "hyp_overlap_seconds"),
        "events_count": _g(events, "events"),
        "llm_resolved_rate": _g(events, "llm_resolved_rate"),
        "fallback_resolved_rate": _g(events, "fallback_resolved_rate"),
        "evidence_support": _g(evidence, "support"),
        "wall_time_s": _g(timing, "wall_time_s"),
        "rtf": _g(timing, "rtf"),
        "audio_duration_s": _g(timing, "audio_duration_s"),
        "exit_code": rec.meta.get("exit_code"),
    }


# ---- Difficulty table -------------------------------------------------------


def _audio_duration_s(wav_path: Path) -> float | None:
    """Best-effort audio duration using stdlib ``wave`` (no soundfile dep)."""
    import wave

    try:
        with wave.open(str(wav_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, EOFError, FileNotFoundError):
        return None


def _audio_path_for_meeting(data_root: Path, meeting_id: str) -> Path | None:
    """Find the per-meeting wav (mic suffix MS801 by default).

    ``meeting_id`` may already include ``_MS801`` etc.; we strip it and try the
    canonical mic-first layout under ``Eval_Ali_far/audio_dir``.
    """
    base = meeting_id
    for suf in ("_MS801", "_MS802", "_MS803"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    for mic in ("MS801", "MS802", "MS803"):
        candidate = data_root / "Eval_Ali" / "Eval_Ali_far" / "audio_dir" / f"{base}_{mic}.wav"
        if candidate.exists():
            return candidate
    return None


def _compute_difficulty(ann: dict[str, Any], duration_s: float | None) -> dict[str, Any]:
    turns = ann.get("turns", [])
    overlap_s = float(ann.get("overlap_seconds", 0.0) or 0.0)
    overlap_regions = ann.get("overlap_regions", []) or []
    n_turns = len(turns)
    durations = [float(t.get("end_time", 0.0)) - float(t.get("start_time", 0.0)) for t in turns]
    mean_turn_s = sum(durations) / n_turns if n_turns else 0.0
    dur = duration_s or (max((float(t.get("end_time", 0.0)) for t in turns), default=0.0))
    turns_per_min = (n_turns / dur * 60.0) if dur > 0 else 0.0
    overlap_ratio = overlap_s / dur if dur > 0 else 0.0
    return {
        "meeting_id": ann.get("meeting_id", ""),
        "duration_s": round(dur, 3),
        "num_speakers": ann.get("num_speakers", 0),
        "num_turns": n_turns,
        "turns_per_min": round(turns_per_min, 3),
        "mean_turn_s": round(mean_turn_s, 3),
        "overlap_s": round(overlap_s, 3),
        "overlap_ratio": round(overlap_ratio, 4),
        "num_overlap_regions": len(overlap_regions),
        "speech_s": round(sum(durations), 3),
        "speech_ratio": round(sum(durations) / dur, 4) if dur > 0 else 0.0,
    }


def build_difficulty_table(data_root: Path) -> list[dict[str, Any]]:
    """One row per ground-truth annotation, joined with audio duration."""
    ann_dir = data_root / "annotations"
    if not ann_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        wav = _audio_path_for_meeting(data_root, ann["meeting_id"])
        dur = _audio_duration_s(wav) if wav else None
        rows.append(_compute_difficulty(ann, dur))
    rows.sort(key=lambda r: r["overlap_ratio"], reverse=True)
    return rows


# ---- Writers ----------------------------------------------------------------


CSV_FIELDS = [
    "cell_id", "meeting_id", "asr", "osd", "resolver", "separation", "language", "asr_device",
    "cer_concat", "cer_low", "cer_high", "wer_concat",
    "ref_chars", "hyp_chars", "subs", "ins", "del",
    "spk_best_mapping_acc", "spk_known_coverage", "spk_unknown_coverage",
    "spk_ref_n", "spk_hyp_n",
    "routing_accuracy", "routing_f1", "routing_support",
    "overlap_recall", "overlap_precision", "overlap_f1",
    "gt_overlap_s", "hyp_overlap_s",
    "events_count", "llm_resolved_rate", "fallback_resolved_rate",
    "evidence_support",
    "wall_time_s", "rtf", "audio_duration_s",
    "exit_code",
]


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        cells = []
        for c in r:
            if isinstance(c, float):
                cells.append(f"{c:.3f}")
            else:
                cells.append(str(c))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_summary(
    out_dir: Path,
    flat_rows: list[dict[str, Any]],
    difficulty_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Experiment summary", ""]
    md.append(f"- runs collected: **{len(flat_rows)}**")
    cells = sorted({r["cell_id"] for r in flat_rows})
    meetings = sorted({r["meeting_id"] for r in flat_rows})
    md.append(f"- distinct cells: **{len(cells)}**")
    md.append(f"- distinct meetings: **{len(meetings)}**")
    md.append("")

    if difficulty_rows:
        md.append("## Meeting difficulty (sorted by overlap_ratio desc)")
        md.append("")
        md.append(_md_table(
            ["meeting_id", "duration_s", "num_speakers", "num_turns",
             "turns_per_min", "overlap_s", "overlap_ratio", "mean_turn_s"],
            [[r["meeting_id"], r["duration_s"], r["num_speakers"], r["num_turns"],
              r["turns_per_min"], r["overlap_s"], r["overlap_ratio"], r["mean_turn_s"]]
             for r in difficulty_rows],
        ))

    md.append("## Per-(cell × meeting) scores")
    md.append("")
    md.append(_md_table(
        ["cell_id", "meeting_id", "cer", "wer", "spk_acc", "routing_f1",
         "overlap_f1", "events", "llm_resolved", "wall_s"],
        [[r["cell_id"], r["meeting_id"], r.get("cer_concat"), r.get("wer_concat"),
          r.get("spk_best_mapping_acc"), r.get("routing_f1"),
          r.get("overlap_f1"), r.get("events_count"),
          r.get("llm_resolved_rate"), r.get("wall_time_s")]
         for r in flat_rows],
    ))
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


# ---- Main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    records = list(discover_runs(runs_root))
    flat = [flatten_evaluation(r) for r in records]
    flat.sort(key=lambda r: (r["cell_id"], r["meeting_id"]))

    write_csv(flat, out_dir / "summary_per_cell_meeting.csv", CSV_FIELDS)

    difficulty = build_difficulty_table(data_root)
    write_csv(
        difficulty,
        out_dir / "meeting_difficulty.csv",
        ["meeting_id", "duration_s", "num_speakers", "num_turns",
         "turns_per_min", "mean_turn_s", "overlap_s", "overlap_ratio",
         "num_overlap_regions", "speech_s", "speech_ratio"],
    )
    write_markdown_summary(out_dir, flat, difficulty)

    print(f"wrote {len(flat)} per-(cell, meeting) rows to {out_dir / 'summary_per_cell_meeting.csv'}")
    print(f"wrote {len(difficulty)} difficulty rows to {out_dir / 'meeting_difficulty.csv'}")
    print(f"wrote markdown summary to {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
