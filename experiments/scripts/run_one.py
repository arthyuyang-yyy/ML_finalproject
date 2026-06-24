"""Single (cell × meeting) run wrapper for the experiment layer.

Invokes `main.py` once for a given cell config + audio, and writes
`run_meta.json` next to the pipeline outputs (with wall_time, config, and a
post-run detector-source tally).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO / "main.py"


def _main_args(cell: dict, audio: str, meeting_id: str) -> list[str]:
    """Build the CLI args for `main.py` from a cell config dict.

    Expected cell keys (any missing → '' or default):
        asr:           'faster-whisper' | 'funasr' | 'whisperx' | 'mock'
        asr_device:    'cpu' | 'cuda'
        asr_compute_type: 'int8' | 'float16' | 'float32'
        faster_whisper_model: 'small' | 'medium' | ...
        osd:           'pyannote' | 'energy_fallback' | 'none'   (we map to env)
        resolver:      'none' | 'openai'                          (we map to gemma-backend)
        separation:    'none' | 'sepformer' | 'nmf' | 'mock'
        language:      'zh' | 'en' | 'und'
    """
    args: list[str] = [sys.executable, str(MAIN), audio, "--meeting-id", meeting_id]
    asr = cell.get("asr") or "mock"
    args += ["--asr", asr]
    if asr == "faster-whisper":
        if cell.get("faster_whisper_model"):
            args += ["--faster-whisper-model", cell["faster_whisper_model"]]
    # Both faster-whisper and funasr honour --asr-device / --asr-compute-type.
    if asr in {"faster-whisper", "funasr"}:
        if cell.get("asr_device"):
            args += ["--asr-device", cell["asr_device"]]
        if cell.get("asr_compute_type"):
            args += ["--asr-compute-type", cell["asr_compute_type"]]
    if cell.get("language"):
        args += ["--language", cell["language"]]
    sep = cell.get("separation") or "none"
    if sep != "none":
        args += ["--speech-separation", sep]
    res = cell.get("resolver") or "none"
    if res == "openai":
        args += ["--gemma-backend", "openai"]
        # The default ``--gemma-model=gemma3:4b`` is an Ollama alias; the
        # current DeepSeek v4 API rejects it with HTTP 400. Route openai
        # resolver runs at the DeepSeek-aware model unless the cell overrides.
        if not cell.get("gemma_model"):
            args += ["--gemma-model", "deepseek-v4-flash"]
    elif res == "none":
        args += ["--gemma-backend", "none"]
    else:
        args += ["--gemma-backend", res]
    if cell.get("gemma_model"):
        args += ["--gemma-model", cell["gemma_model"]]
    return args


def _tally_detector_sources(run_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    p = run_dir / "overlap.json"
    if p.exists():
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
            for it in arr:
                k = str(it.get("overlap_detector", "unknown"))
                counts[k] = counts.get(k, 0) + 1
        except Exception:  # noqa: BLE001
            pass
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--meeting-id", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-root", default=str(REPO / "experiments" / "runs"))
    # cell config (json string, e.g. '{"asr":"funasr","osd":"pyannote","resolver":"openai","separation":"none","language":"zh"}')
    ap.add_argument("--cell-json", required=True)
    # optional OSD mode override (passed via env)
    ap.add_argument("--osd", default=None, help="force OSD mode in detector env: pyannote | energy_fallback")
    ap.add_argument("--deepseek", action="store_true", help="inject DeepSeek env vars")
    args = ap.parse_args()

    cell = json.loads(args.cell_json)
    if args.osd:
        cell["osd"] = args.osd
    cell_id = args.cell_id
    meeting_id = args.meeting_id
    out_dir = Path(args.out_root) / cell_id / meeting_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build env — load project .env first so HF_TOKEN, OPENAI_API_KEY etc.
    # are populated before we copy the parent environment to the child.
    sys.path.insert(0, str(REPO))
    from src.utils import load_dotenv

    load_dotenv(REPO / ".env")
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    # OSD mode toggle (project reads OSD_MODE? — we just put it in env for a
    # downstream toggle; main.py currently uses OSD detector based on its own
    # internal default unless --osd-mode is added. We keep it documented in meta.)
    if cell.get("osd"):
        env["MM_OSD_MODE"] = cell["osd"]
    if args.deepseek or cell.get("resolver") == "openai":
        env.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        env.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        env.setdefault("OPENAI_MODEL", "deepseek-chat")

    # Build CLI
    cli = _main_args(cell, args.audio, meeting_id)

    # Pre-run meta
    meta = {
        "cell_id": cell_id,
        "meeting_id": meeting_id,
        "audio_path": args.audio,
        "config": cell,
        "cli": cli[2:],  # strip [python, main.py]
        "start_time": time.time(),
        "end_time": None,
        "wall_time_s": None,
        "exit_code": None,
        "detector_source_counts": {},
    }
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Run main.py with cwd=out_dir so its outputs land there
    # main.py writes to outputs/<meeting_id>/ relative to cwd
    print("+", " ".join(cli), flush=True)
    t0 = time.time()
    proc = subprocess.run(cli, cwd=str(out_dir), env=env, check=False)
    wall = time.time() - t0

    meta["end_time"] = time.time()
    meta["wall_time_s"] = wall
    meta["exit_code"] = proc.returncode
    meta["detector_source_counts"] = _tally_detector_sources(out_dir / "outputs" / meeting_id)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> exit={proc.returncode} wall={wall:.1f}s detector_counts={meta['detector_source_counts']}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
