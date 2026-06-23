"""Strict full-flow tuning on a 4-minute AliMeeting clip.

This experiment intentionally rejects fallback output. A run is marked failed if
the pipeline falls back from pyannote overlap detection, mock ASR, high-overlap
candidate generation, resolver output, or event extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_alimeeting_result import main as evaluate_alimeeting_result  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.run_pipeline import run_meeting_pipeline  # noqa: E402
from src.utils import load_dotenv  # noqa: E402

SOURCE_AUDIO = Path(
    "/Users/lymn/MLData/meeting-memory/datasets/alimeeting/eval/Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav"
)
TEXTGRID_DIR = Path(
    "/Users/lymn/MLData/meeting-memory/datasets/alimeeting/eval/Eval_Ali/Eval_Ali_near/textgrid_dir"
)
MEETING_ID = "R8001_M8004"
WINDOW_START = 1110.0
WINDOW_END = 1350.0
SECOND_WINDOW_START = 300.0
SECOND_WINDOW_END = 540.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audio", type=Path, default=SOURCE_AUDIO)
    parser.add_argument("--textgrid-dir", type=Path, default=TEXTGRID_DIR)
    parser.add_argument("--meeting-id", default=MEETING_ID)
    parser.add_argument("--window-start", type=float, default=WINDOW_START)
    parser.add_argument("--window-end", type=float, default=WINDOW_END)
    parser.add_argument("--window-preset", choices=["primary", "second"], default="primary")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/fullflow_tuning"))
    parser.add_argument("--asr", default="funasr", choices=["funasr", "faster-whisper", "whisperx", "whisper"])
    parser.add_argument("--language", default="zh")
    parser.add_argument("--faster-whisper-model", default="small")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--gemma-backend", default=os.environ.get("GEMMA_BACKEND", "deepseek"))
    parser.add_argument("--gemma-model", default=os.environ.get("GEMMA_MODEL") or None)
    parser.add_argument("--gemma-base-url", default=os.environ.get("GEMMA_BASE_URL") or None)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run the full staged sweep")
    parser.add_argument("--only", action="append", default=[], help="Run only the named spec; can be repeated")
    parser.add_argument("--force", action="store_true", help="Re-run specs even when a completed run_result.json exists")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.window_preset == "second":
        args.window_start = SECOND_WINDOW_START
        args.window_end = SECOND_WINDOW_END

    load_dotenv()
    _install_strict_candidate_guard()

    output_root = args.output_root.expanduser().resolve()
    clip_path = output_root / "clips" / f"{args.meeting_id}_{int(args.window_start)}_{int(args.window_end)}_4min.wav"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_clip(args.source_audio.expanduser(), clip_path, args.window_start, args.window_end)

    if not args.skip_preflight:
        _preflight(args)

    run_specs = _staged_run_specs()
    if args.only:
        wanted = set(args.only)
        run_specs = [spec for spec in run_specs if spec["name"] in wanted]
        missing = wanted - {spec["name"] for spec in run_specs}
        if missing:
            raise ValueError(f"unknown run spec(s): {sorted(missing)}")
    if args.max_runs > 0:
        run_specs = run_specs[: args.max_runs]

    results: list[dict[str, Any]] = _load_existing_results(output_root)
    best_spec = deepcopy(_base_params())
    stages = [
        ("baseline", [spec for spec in run_specs if spec["stage"] == "baseline"]),
        ("vad", [spec for spec in run_specs if spec["stage"] == "vad"]),
        ("asr_padding", [spec for spec in run_specs if spec["stage"] == "asr_padding"]),
        ("routing", [spec for spec in run_specs if spec["stage"] == "routing"]),
        ("separation", [spec for spec in run_specs if spec["stage"] == "separation"]),
    ]

    for _, specs in stages:
        if not specs:
            continue
        stage_results = []
        for spec in specs:
            params = deepcopy(best_spec)
            params.update(spec["params"])
            row = _run_one(args, clip_path, output_root, spec["name"], params)
            results = [existing for existing in results if existing.get("name") != row["name"]]
            results.append(row)
            stage_results.append(row)
            _write_report(output_root, results)
        valid = [row for row in stage_results if row["status"] == "completed"]
        if valid:
            winner = max(valid, key=lambda row: row["score"])
            best_spec.update(winner["params"])

    _write_report(output_root, results)
    return 0 if any(row["status"] == "completed" for row in results) else 1


def _base_params() -> dict[str, Any]:
    return {
        "vad_max_segment_s": 30.0,
        "vad_speech_pad_ms": 400,
        "vad_min_silence_ms": 500,
        "asr_context_padding_s": 0.2,
        "overlap_threshold": 0.4,
        "suspected_overlap_threshold": 0.2,
        "high_overlap_min_segment_s": 1.0,
        "high_overlap_decode_context_s": 2.0,
        "suspected_overlap_min_confidence_gain": 0.15,
        "suspected_overlap_max_text_cer": 0.35,
        "enable_denoise": False,
        "denoise_strength": 0.5,
        "speech_separation_backend": "none",
    }


def _staged_run_specs() -> list[dict[str, Any]]:
    return [
        {"stage": "baseline", "name": "run_001_baseline", "params": {}},
        {"stage": "vad", "name": "run_002_vad_max20", "params": {"vad_max_segment_s": 20.0}},
        {
            "stage": "vad",
            "name": "run_003_vad_tight",
            "params": {"vad_max_segment_s": 15.0, "vad_speech_pad_ms": 300, "vad_min_silence_ms": 500},
        },
        {
            "stage": "vad",
            "name": "run_004_vad_wide",
            "params": {"vad_max_segment_s": 30.0, "vad_speech_pad_ms": 600, "vad_min_silence_ms": 800},
        },
        {"stage": "asr_padding", "name": "run_005_pad_0p1", "params": {"asr_context_padding_s": 0.1}},
        {"stage": "asr_padding", "name": "run_006_pad_0p5", "params": {"asr_context_padding_s": 0.5}},
        {
            "stage": "routing",
            "name": "run_007_route_recall",
            "params": {"overlap_threshold": 0.3, "suspected_overlap_threshold": 0.15},
        },
        {
            "stage": "routing",
            "name": "run_008_route_precision",
            "params": {"overlap_threshold": 0.5, "suspected_overlap_threshold": 0.25},
        },
        {
            "stage": "routing",
            "name": "run_009_short_overlap",
            "params": {"high_overlap_min_segment_s": 0.5},
        },
        {
            "stage": "routing",
            "name": "run_010_long_overlap",
            "params": {"high_overlap_min_segment_s": 1.5},
        },
        {
            "stage": "routing",
            "name": "run_011_strict_high_min2",
            "params": {"high_overlap_min_segment_s": 2.0, "suspected_overlap_threshold": 0.3},
        },
        {"stage": "separation", "name": "run_012_nmf", "params": {"speech_separation_backend": "nmf"}},
        {
            "stage": "routing",
            "name": "run_013_strict_pad_0p1",
            "params": {
                "high_overlap_min_segment_s": 2.0,
                "suspected_overlap_threshold": 0.3,
                "asr_context_padding_s": 0.1,
            },
        },
        {
            "stage": "routing",
            "name": "run_014_strict_pad_0p5",
            "params": {
                "high_overlap_min_segment_s": 2.0,
                "suspected_overlap_threshold": 0.3,
                "asr_context_padding_s": 0.5,
            },
        },
        {
            "stage": "routing",
            "name": "run_015_strict_route_precision",
            "params": {
                "high_overlap_min_segment_s": 2.0,
                "suspected_overlap_threshold": 0.3,
                "overlap_threshold": 0.5,
            },
        },
        {
            "stage": "routing",
            "name": "run_018_short_overlap_context",
            "params": {
                "high_overlap_min_segment_s": 2.0,
                "high_overlap_decode_context_s": 2.0,
                "suspected_overlap_threshold": 0.3,
                "overlap_threshold": 0.5,
            },
        },
        {
            "stage": "routing",
            "name": "run_016_strict_vad_tight",
            "params": {
                "high_overlap_min_segment_s": 2.0,
                "suspected_overlap_threshold": 0.3,
                "vad_max_segment_s": 15.0,
                "vad_speech_pad_ms": 300,
                "vad_min_silence_ms": 500,
            },
        },
        {
            "stage": "separation",
            "name": "run_017_strict_nmf",
            "params": {
                "high_overlap_min_segment_s": 2.0,
                "suspected_overlap_threshold": 0.3,
                "speech_separation_backend": "nmf",
            },
        },
    ]


def _run_one(
    args: argparse.Namespace,
    clip_path: Path,
    output_root: Path,
    run_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    meeting_id = f"{args.meeting_id}_{int(args.window_start)}_{int(args.window_end)}_{run_name}"
    run_root = output_root / run_name
    outputs_root = run_root / "outputs"
    memory_root = run_root / "memory"
    cached = _read_completed_run(run_root / "run_result.json", params, args.force)
    if cached:
        return cached
    row: dict[str, Any] = {
        "name": run_name,
        "meeting_id": meeting_id,
        "params": params,
        "output_dir": str(outputs_root / meeting_id),
        "status": "running",
    }
    try:
        cfg = PipelineConfig(
            outputs_root=outputs_root,
            memory_root=memory_root,
            language=args.language,
            low_overlap_asr_model=args.asr,
            faster_whisper_model_size=args.faster_whisper_model,
            faster_whisper_device=args.asr_device,
            faster_whisper_compute_type=args.asr_compute_type,
            gemma_backend=args.gemma_backend,
            gemma_model=args.gemma_model,
            gemma_base_url=args.gemma_base_url,
            **params,
        )
        result = run_meeting_pipeline(str(clip_path), meeting_id, config=cfg)
        output_dir = Path(result["output_dir"])
        strict_report = _validate_no_fallback(output_dir)
        eval_report = _evaluate(args, output_dir, params["overlap_threshold"])
        metrics = _summarize_metrics(output_dir, eval_report)
        score = _score(metrics)
        row.update({
            "status": "completed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "strict": strict_report,
            "metrics": metrics,
            "score": score,
        })
    except Exception as exc:
        row.update({
            "status": "failed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
    _write_json(run_root / "run_result.json", row)
    return row


def _write_clip(source: Path, target: Path, start: float, end: float) -> None:
    if target.is_file():
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    data, sample_rate = sf.read(str(source), always_2d=False)
    left = max(0, int(round(start * sample_rate)))
    right = max(left, int(round(end * sample_rate)))
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), data[left:right], sample_rate)


def _preflight(args: argparse.Namespace) -> None:
    missing = []
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        missing.append("HF_TOKEN or HUGGINGFACE_TOKEN")
    if args.gemma_backend in {"", "none", "fallback"}:
        missing.append("real gemma backend")
    if args.gemma_backend == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")
    if args.gemma_backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError("strict experiment preflight failed; missing " + ", ".join(missing))

    if args.gemma_backend == "ollama":
        base_url = (args.gemma_base_url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        result = subprocess.run(
            [sys.executable, "-c", _OLLAMA_PROBE, base_url],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("strict experiment preflight failed; Ollama is not reachable")


def _install_strict_candidate_guard() -> None:
    import src.candidates.generator as generator

    def fail_fallback(*_: Any, **__: Any) -> list[dict[str, Any]]:
        raise RuntimeError("strict experiment forbids high-overlap fallback candidates")

    generator.fallback_candidates = fail_fallback


def _validate_no_fallback(output_dir: Path) -> dict[str, Any]:
    overlap = _read_json(output_dir / "overlap.json")
    diarization = _read_json(output_dir / "diarization.json")
    evidence = _read_json(output_dir / "evidence_segments.json")
    events = _read_json(output_dir / "meeting_events.json")

    detectors = sorted({str(item.get("overlap_detector", "")) for item in overlap})
    if detectors != ["pyannote"]:
        raise RuntimeError(f"non-pyannote overlap detector used: {detectors}")
    if not diarization:
        raise RuntimeError("pyannote diarization produced no turns")
    if any("[mock transcript" in str(item.get("text", "")) for item in evidence):
        raise RuntimeError("mock ASR transcript detected")
    bad_sources = {
        str(item.get("source", ""))
        for item in evidence
        if str(item.get("source", "")) in {"fallback_resolved", "baseline_preserved"}
    }
    if bad_sources:
        raise RuntimeError(f"fallback resolver source detected: {sorted(bad_sources)}")
    for item in evidence:
        for candidate in item.get("candidates", []) or []:
            note = str(candidate.get("uncertainty_note", "")).lower()
            if "fallback" in note:
                raise RuntimeError("fallback high-overlap candidate detected")
    summary = str(events.get("meeting_summary", "")) if isinstance(events, dict) else ""
    if "high-overlap segment(s) remain uncertain" in summary:
        raise RuntimeError("deterministic fallback event document detected")
    if isinstance(events, dict) and str(events.get("uncertainty_note", "")).lower().find("fallback") >= 0:
        raise RuntimeError("fallback event extraction detected")

    return {
        "overlap_detector": "pyannote",
        "diarization_turns": len(diarization),
        "evidence_segments": len(evidence),
        "events": len(events.get("events", [])) if isinstance(events, dict) else 0,
    }


def _evaluate(args: argparse.Namespace, output_dir: Path, overlap_threshold: float) -> dict[str, Any]:
    eval_path = output_dir / "alimeeting_eval.json"
    stdout = _capture_stdout(
        evaluate_alimeeting_result,
        [
            "--output-dir",
            str(output_dir),
            "--textgrid-dir",
            str(args.textgrid_dir.expanduser()),
            "--meeting-id",
            args.meeting_id,
            "--window-start",
            str(args.window_start),
            "--window-end",
            str(args.window_end),
            "--overlap-threshold",
            str(overlap_threshold),
        ],
    )
    payload = json.loads(stdout)
    _write_json(eval_path, payload)
    return payload


def _summarize_metrics(output_dir: Path, eval_report: dict[str, Any]) -> dict[str, Any]:
    evidence = _read_json(output_dir / "evidence_segments.json")
    events = _read_json(output_dir / "meeting_events.json")
    high = [item for item in evidence if item.get("processing_path") == "high_overlap_candidate"]
    unknown = [item for item in evidence if item.get("speaker") == "UNKNOWN"]
    return {
        "cer": float(eval_report["asr"]["cer"]),
        "overlap_recall": eval_report["overlap"]["recall"],
        "known_speaker_coverage": float(eval_report["speaker"]["known_speaker_coverage"]),
        "unknown_speaker_coverage": float(eval_report["speaker"]["unknown_speaker_coverage"]),
        "num_evidence_segments": len(evidence),
        "high_overlap_ratio": round(len(high) / len(evidence), 4) if evidence else 0.0,
        "unknown_segment_ratio": round(len(unknown) / len(evidence), 4) if evidence else 0.0,
        "num_events": len(events.get("events", [])) if isinstance(events, dict) else 0,
    }


def _score(metrics: dict[str, Any]) -> float:
    cer = max(0.0, min(2.0, float(metrics["cer"])))
    recall = metrics["overlap_recall"]
    recall_value = 0.0 if recall is None else float(recall)
    known = float(metrics["known_speaker_coverage"])
    high_ratio = float(metrics["high_overlap_ratio"])
    unknown_ratio = float(metrics["unknown_segment_ratio"])
    event_score = min(1.0, float(metrics["num_events"]) / 12.0)
    return round(
        (1.0 - min(1.0, cer)) * 0.45
        + recall_value * 0.25
        + known * 0.15
        + event_score * 0.05
        - abs(high_ratio - 0.35) * 0.05
        - unknown_ratio * 0.05,
        4,
    )


def _write_report(output_root: Path, results: list[dict[str, Any]]) -> None:
    completed = [row for row in results if row["status"] == "completed"]
    best = max(completed, key=lambda row: row["score"]) if completed else None
    payload = {
        "clip": {
            "source_audio": str(SOURCE_AUDIO),
            "default_window": [WINDOW_START, WINDOW_END],
            "second_window": [SECOND_WINDOW_START, SECOND_WINDOW_END],
            "default_duration_seconds": WINDOW_END - WINDOW_START,
        },
        "best": best,
        "runs": results,
    }
    _write_json(output_root / "summary.json", payload)


def _load_existing_results(output_root: Path) -> list[dict[str, Any]]:
    summary = output_root / "summary.json"
    if not summary.is_file():
        return []
    try:
        payload = _read_json(summary)
    except Exception:
        return []
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    return runs if isinstance(runs, list) else []


def _read_completed_run(path: Path, params: dict[str, Any], force: bool) -> dict[str, Any] | None:
    if force or not path.is_file():
        return None
    try:
        row = _read_json(path)
    except Exception:
        return None
    if row.get("status") != "completed":
        return None
    return row if row.get("params") == params else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _capture_stdout(func: Any, args: list[str]) -> str:
    from contextlib import redirect_stdout
    from io import StringIO

    buffer = StringIO()
    with redirect_stdout(buffer):
        code = func(args)
    if code != 0:
        raise RuntimeError(f"evaluation failed with exit code {code}")
    return buffer.getvalue()


_OLLAMA_PROBE = """
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], method='HEAD')
with urllib.request.urlopen(req, timeout=5):
    pass
"""


if __name__ == "__main__":
    raise SystemExit(main())
