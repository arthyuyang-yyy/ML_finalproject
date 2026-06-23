"""JSONL I/O for local meeting-dataset manifests."""

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import DatasetRecord, validate_dataset_record


def read_manifest(path: str | Path, *, require_files: bool = False) -> list[DatasetRecord]:
    """Read and validate a JSONL manifest."""
    source = Path(path)
    records: list[DatasetRecord] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload: Any = json.loads(raw_line)
            record = validate_dataset_record(payload, require_files=require_files)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
        meeting_id = record["meeting_id"]
        if meeting_id in seen_ids:
            raise ValueError(f"duplicate meeting_id in manifest: {meeting_id}")
        seen_ids.add(meeting_id)
        records.append(record)
    if not records:
        raise ValueError("manifest must contain at least one dataset record")
    return records


def write_manifest(path: str | Path, records: Iterable[dict[str, Any]]) -> list[DatasetRecord]:
    """Validate and write records using stable JSONL formatting."""
    validated = [validate_dataset_record(record) for record in records]
    meeting_ids = [record["meeting_id"] for record in validated]
    if len(meeting_ids) != len(set(meeting_ids)):
        raise ValueError("manifest records must have unique meeting_id values")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in validated)
    target.write_text(content, encoding="utf-8")
    return validated


__all__ = ["read_manifest", "write_manifest"]
