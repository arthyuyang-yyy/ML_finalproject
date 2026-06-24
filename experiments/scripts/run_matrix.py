"""Run the full 4D experiment matrix over the AliMeeting Eval_Ali_far set.

Designed for long, unattended execution under ``nohup``:

* multiple CUDA devices (--gpus 0,1): each device runs a worker that pulls
  work off a shared queue, so two (cell, meeting) runs sit on two GPUs at
  once and never oversubscribe a single device;
* stdout/stderr of every run are tee'd into ``runs/<cell>/<meeting>/run.log``
  while the matrix runner itself logs to ``experiments/runs/_matrix.log``;
* resumable — any (cell, meeting) with the full output bundle on disk is
  skipped via ``is_complete``;
* per-run progress is mirrored to ``experiments/runs/_matrix_status.json`` so
  a polling script (or a human) can inspect partial progress without parsing
  free-form log lines;
* SIGINT/SIGTERM cleanly stop after the current run; the matrix state is
  flushed before exit so a re-launch picks up where it left off.

Typical invocation (two-GPU)::

    nohup python experiments/scripts/run_matrix.py \
        --gpus 0,1 --deepseek \
        > experiments/runs/_matrix.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
RUNS_ROOT = EXPERIMENTS / "runs"
MATRIX_JSON = EXPERIMENTS / "matrix.json"
RUN_ONE = EXPERIMENTS / "scripts" / "run_one.py"
STATUS_JSON = RUNS_ROOT / "_matrix_status.json"
MATRIX_LOG = RUNS_ROOT / "_matrix.log"

EVAL_ALI_FAR_AUDIO = REPO / "data" / "alimeeting" / "Eval_Ali" / "Eval_Ali_far" / "audio_dir"

# Hard-coded set for reproducibility. The user can override via --meeting
# (single meeting) or --all-meetings (every wav in Eval_Ali_far/audio_dir).
DEFAULT_MEETINGS: list[str] = [
    "R8001_M8004_MS801",
    "R8003_M8001_MS801",
    "R8007_M8010_MS803",
    "R8007_M8011_MS806",
    "R8008_M8013_MS807",
    "R8009_M8018_MS809",
    "R8009_M8019_MS810",
    "R8009_M8020_MS810",
]  # yapf: disable

_stop_requested: list[bool] = [False]
_status_lock = threading.Lock()


def _on_signal(signum: int, _frame: Any) -> None:
    _stop_requested[0] = True
    print(f"[matrix] received signal {signum}; will stop after the current run",
          file=sys.stderr, flush=True)


# ---- Manifest helpers -------------------------------------------------------


def load_matrix() -> list[dict[str, Any]]:
    if not MATRIX_JSON.exists():
        raise SystemExit(
            f"{MATRIX_JSON} not found; run `python experiments/scripts/build_matrix.py` first."
        )
    doc = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    return doc["cells"]


def resolve_meetings(args: argparse.Namespace) -> list[str]:
    if args.meeting:
        return [args.meeting]
    if args.all_meetings:
        return sorted(p.stem for p in EVAL_ALI_FAR_AUDIO.glob("*.wav"))
    return list(DEFAULT_MEETINGS)


def audio_path_for(meeting_id: str) -> Path | None:
    """Resolve the audio path for ``meeting_id`` or return None if missing.

    A missing file is *not* fatal — the runner simply skips that (cell, meeting)
    pair and continues. The hard ``SystemExit`` that lived here before would
    kill the entire matrix the first time a default meeting id drifted away
    from the actual mic suffix, which is exactly what just happened.
    """
    p = EVAL_ALI_FAR_AUDIO / f"{meeting_id}.wav"
    if not p.exists():
        return None
    return p


# ---- Resumability -----------------------------------------------------------


def is_complete(cell_id: str, meeting_id: str) -> bool:
    """A run is considered complete when its full output bundle is on disk.

    Earlier this only trusted ``run_meta.exit_code == 0``, which is fragile
    because a crashed matrix process or a half-written final meta would force
    the next launch to re-run a perfectly good pipeline. We now treat the
    existence of every artefact the pipeline emits as the source of truth,
    with ``run_meta.exit_code == 0`` as a soft confirmation.
    """
    run_dir = RUNS_ROOT / cell_id / meeting_id
    out_dir = run_dir / "outputs" / meeting_id
    required = [
        out_dir / "evidence_segments.json",
        out_dir / "meeting_events.json",
        out_dir / "episodic_memory.json",
    ]
    if not all(p.exists() for p in required):
        return False
    meta_p = run_dir / "run_meta.json"
    if not meta_p.exists():
        return False
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    # ``exit_code`` may still be ``None`` if the previous matrix runner was
    # killed mid-meta-write — that's fine, the output bundle is the truth.
    return meta.get("exit_code") in (0, None)


# ---- Status persistence -----------------------------------------------------


def write_status(records: list[dict[str, Any]]) -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    with _status_lock:
        STATUS_JSON.write_text(
            json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(),
                        "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---- Run launching ----------------------------------------------------------


def run_one(
    cell: dict[str, Any],
    meeting_id: str,
    gpu: str,
    deepseek: bool,
) -> dict[str, Any]:
    """Invoke ``run_one.py`` for a single (cell, meeting)."""
    run_dir = RUNS_ROOT / cell["cell_id"] / meeting_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    cli = [
        sys.executable, "-u", str(RUN_ONE),
        "--cell-id", cell["cell_id"],
        "--meeting-id", meeting_id,
        "--audio", str(audio_path_for(meeting_id)),
        "--cell-json", json.dumps(cell, ensure_ascii=False),
    ]
    if deepseek:
        cli.append("--deepseek")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")

    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    print(f"[matrix] launch {cell['cell_id']} × {meeting_id}  → {log_path}",
          flush=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n=== launch @ {started_iso}  cli={' '.join(cli[2:])} ===\n")
        logf.flush()
        # ``start_new_session=True`` decouples the child from the matrix runner's
        # controlling terminal; combined with ``nohup``-style usage on the
        # parent this gives long-running stability across SSH drops.
        proc = subprocess.Popen(
            cli,
            cwd=str(REPO),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise
    wall = time.time() - started_at
    print(f"[matrix] done   {cell['cell_id']} × {meeting_id}  rc={rc} wall={wall:.1f}s",
          flush=True)
    return {
        "cell_id": cell["cell_id"],
        "meeting_id": meeting_id,
        "exit_code": rc,
        "wall_time_s": round(wall, 2),
        "log_path": str(log_path),
        "started_at": started_iso,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


# ---- Main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None,
                    help="single CUDA device (legacy alias for --gpus)")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated CUDA devices, one worker per device")
    ap.add_argument("--meeting", default=None, help="restrict to one meeting id")
    ap.add_argument("--all-meetings", action="store_true",
                    help="iterate over every wav in Eval_Ali_far/audio_dir")
    ap.add_argument("--deepseek", action="store_true",
                    help="inject DeepSeek env vars into every run")
    ap.add_argument("--include-mock", action="store_true",
                    help="also run the mock sanity cell (off by default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate the work but don't launch anything")
    args = ap.parse_args()

    gpus = [g.strip() for g in (args.gpu or args.gpus).split(",") if g.strip()]
    if not gpus:
        gpus = ["0"]

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    cells = load_matrix()
    if not args.include_mock:
        cells = [c for c in cells if c["asr"] != "mock"]
    meetings = resolve_meetings(args)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    MATRIX_LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] gpus={gpus} cells={len(cells)} meetings={len(meetings)}",
          flush=True)

    # Pre-enumerate the (cell, meeting) work items so workers can pull them
    # off the queue without contention. Anything ``is_complete`` is dropped
    # here so the queue contains only real work.
    work: list[tuple[dict[str, Any], str]] = []
    skipped = 0
    for cell in cells:
        for meeting_id in meetings:
            if is_complete(cell["cell_id"], meeting_id):
                skipped += 1
                continue
            if audio_path_for(meeting_id) is None:
                print(f"[matrix] SKIP  no audio for {meeting_id}", flush=True)
                skipped += 1
                continue
            work.append((cell, meeting_id))

    print(f"[matrix] work={len(work)} skipped(complete|missing)={skipped}",
          flush=True)

    if args.dry_run:
        for cell, mid in work:
            print(f"[matrix] DRY  {cell['cell_id']} × {mid}", flush=True)
        return 0

    if not work:
        return 0

    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    work_q: queue.Queue[tuple[dict[str, Any], str] | None] = queue.Queue()
    for item in work:
        work_q.put(item)

    def _record(rec: dict[str, Any]) -> None:
        with records_lock:
            records.append(rec)
            write_status(records)

    def _worker(gpu: str) -> None:
        worker_tag = f"[gpu={gpu}]"
        while not _stop_requested[0]:
            try:
                item = work_q.get(timeout=1.0)
            except queue.Empty:
                return
            if item is None:
                work_q.task_done()
                return
            cell, meeting_id = item
            try:
                rec = run_one(cell, meeting_id, gpu, args.deepseek)
            except Exception as exc:  # noqa: BLE001
                print(f"[matrix] {worker_tag} EXC  {cell['cell_id']} × "
                      f"{meeting_id}: {exc}", flush=True)
                work_q.task_done()
                continue
            _record(rec)
            print(f"[matrix] {worker_tag} rc={rec['exit_code']} "
                  f"{cell['cell_id']} × {meeting_id} "
                  f"wall={rec['wall_time_s']:.0f}s", flush=True)
            work_q.task_done()

    threads = [threading.Thread(target=_worker, args=(g,), name=f"matrix-{g}",
                                daemon=True) for g in gpus]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=2.0)
                if _stop_requested[0]:
                    break
    except KeyboardInterrupt:
        _stop_requested[0] = True
        for t in threads:
            t.join(timeout=5.0)

    failed = sum(1 for r in records if r.get("exit_code") not in (0, None))
    print(f"[matrix] skipped(complete|missing)={skipped} launched={len(records)} "
          f"failed={failed} stop_requested={_stop_requested[0]}", flush=True)
    write_status(records)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
