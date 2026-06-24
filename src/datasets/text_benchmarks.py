"""Converters for text-only meeting QA and summarization benchmarks."""

import json
from pathlib import Path
from typing import Any


def prepare_qmsum(source_root: str | Path, output_path: str | Path, split: str = "test") -> int:
    """Convert QMSum meeting JSON files to a compact project JSONL format."""
    source = Path(source_root).expanduser().resolve() / "data" / "ALL" / split
    if not source.is_dir():
        raise ValueError(f"QMSum split directory does not exist: {source}")
    records: list[dict[str, Any]] = []
    for path in sorted(source.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        queries = [
            {"query": item["query"], "reference": item["answer"], "relevant_spans": []}
            for item in payload.get("general_query_list", [])
        ]
        queries.extend(
            {
                "query": item["query"],
                "reference": item["answer"],
                "relevant_spans": item.get("relevant_text_span", []),
            }
            for item in payload.get("specific_query_list", [])
        )
        records.append({
            "dataset": "qmsum",
            "split": split,
            "meeting_id": path.stem,
            "language": "en",
            "transcript": [
                {"speaker": turn.get("speaker", "UNKNOWN"), "text": turn.get("content", "")}
                for turn in payload.get("meeting_transcripts", [])
            ],
            "queries": queries,
            "topics": payload.get("topic_list", []),
        })
    _write_jsonl(output_path, records)
    return len(records)


def prepare_vcsum(source_root: str | Path, output_path: str | Path, split: str = "test") -> int:
    """Convert VCSum long-form records to a compact project JSONL format."""
    source = Path(source_root).expanduser().resolve() / "vcsum_data" / f"long_{split}.txt"
    if not source.is_file():
        raise ValueError(f"VCSum split file does not exist: {source}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid VCSum JSON on line {line_number}: {exc}") from exc
        records.append({
            "dataset": "vcsum",
            "split": split,
            "meeting_id": str(payload["id"]),
            "language": "zh",
            "transcript_groups": payload.get("context", []),
            "speaker_ids": payload.get("speaker", []),
            "reference_summary": payload.get("summary", ""),
            "topic_boundaries": payload.get("eos_index", []),
            "source_id": payload.get("av_num"),
        })
    _write_jsonl(output_path, records)
    return len(records)


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("text benchmark conversion produced no records")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


__all__ = ["prepare_qmsum", "prepare_vcsum"]
