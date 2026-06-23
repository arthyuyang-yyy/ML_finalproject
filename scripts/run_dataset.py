"""Run the meeting pipeline over a local JSONL manifest with resume support."""

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.datasets.manifest import read_manifest
from src.datasets.schema import DatasetRecord, expand_local_path
from src.pipeline.config import PipelineConfig
from src.pipeline.run_pipeline import run_meeting_pipeline


def run_manifest(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    asr: str = "mock",
    language: str | None = None,
    limit: int | None = None,
    meeting_id: str | None = None,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Run selected manifest records and persist status after every meeting."""
    records = read_manifest(manifest_path, require_files=True)
    selected = [record for record in records if meeting_id is None or record["meeting_id"] == meeting_id]
    if meeting_id is not None and not selected:
        raise ValueError(f"meeting_id not found in manifest: {meeting_id}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]

    root = Path(output_root).expanduser().resolve()
    statuses = root / "status"
    outputs = root / "outputs"
    memory = root / "memory"
    statuses.mkdir(parents=True, exist_ok=True)

    summary = {"completed": 0, "failed": 0, "skipped": 0}
    for record in selected:
        status_path = statuses / f"{record['meeting_id']}.json"
        previous = _read_status(status_path)
        if previous.get("status") == "completed":
            summary["skipped"] += 1
            continue
        if previous.get("status") == "failed" and not retry_failed:
            summary["skipped"] += 1
            continue

        started_at = _timestamp()
        _write_status(status_path, _status_payload(record, "running", started_at=started_at))
        try:
            result = run_meeting_pipeline(
                str(expand_local_path(record["audio_path"])),
                record["meeting_id"],
                config=PipelineConfig(
                    outputs_root=outputs,
                    memory_root=memory,
                    language=language or record["language"],
                    low_overlap_asr_model=asr,
                ),
            )
            _write_status(
                status_path,
                _status_payload(
                    record,
                    "completed",
                    started_at=started_at,
                    finished_at=_timestamp(),
                    output_dir=result["output_dir"],
                    num_evidence_segments=result["num_evidence_segments"],
                ),
            )
            summary["completed"] += 1
        except Exception as exc:
            _write_status(
                status_path,
                _status_payload(
                    record,
                    "failed",
                    started_at=started_at,
                    finished_at=_timestamp(),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                ),
            )
            summary["failed"] += 1
    return summary


def _status_payload(record: DatasetRecord, status: str, **details: Any) -> dict[str, Any]:
    return {
        "dataset": record["dataset"],
        "split": record["split"],
        "meeting_id": record["meeting_id"],
        "audio_path": record["audio_path"],
        "status": status,
        **details,
    }


def _read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local meeting datasets with isolated outputs and resume support.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--asr", default="mock", choices=["auto", "whisperx", "faster-whisper", "whisper", "funasr", "mock"])
    parser.add_argument("--language", help="Override the language stored in the manifest")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--meeting-id")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    summary = run_manifest(
        args.manifest,
        args.output_root,
        asr=args.asr,
        language=args.language,
        limit=args.limit,
        meeting_id=args.meeting_id,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(summary, ensure_ascii=False))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
