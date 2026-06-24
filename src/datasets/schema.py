"""Canonical record shape for local meeting-dataset manifests."""

import os
import re
from pathlib import Path
from typing import Any, NotRequired, TypedDict


SAFE_MEETING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUPPORTED_AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


class DatasetRecord(TypedDict):
    dataset: str
    split: str
    meeting_id: str
    audio_path: str
    language: str
    annotation_path: NotRequired[str]
    num_speakers: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]


def expand_local_path(value: str, data_root: str | Path | None = None) -> Path:
    """Expand environment variables and resolve a manifest path locally."""
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "${" in expanded:
        raise ValueError(f"path contains an unresolved environment variable: {value}")
    path = Path(expanded)
    if not path.is_absolute() and data_root is not None:
        path = Path(data_root) / path
    return path.resolve()


def validate_dataset_record(record: Any, *, require_files: bool = False) -> DatasetRecord:
    """Validate one audio-dataset manifest record."""
    if not isinstance(record, dict):
        raise ValueError("dataset record must be a dictionary")

    required = ("dataset", "split", "meeting_id", "audio_path", "language")
    for field in required:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"dataset record field '{field}' must be a non-empty string")

    if not SAFE_MEETING_ID.fullmatch(record["meeting_id"]):
        raise ValueError("meeting_id may contain only letters, numbers, '.', '_' and '-'")

    audio_path = expand_local_path(record["audio_path"])
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(f"unsupported audio suffix: {audio_path.suffix or '<none>'}")
    if require_files and not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")

    annotation_path = record.get("annotation_path")
    if annotation_path is not None:
        if not isinstance(annotation_path, str) or not annotation_path.strip():
            raise ValueError("annotation_path must be a non-empty string when provided")
        if require_files and not expand_local_path(annotation_path).is_file():
            raise ValueError(f"annotation file does not exist: {expand_local_path(annotation_path)}")

    num_speakers = record.get("num_speakers")
    if num_speakers is not None and (not isinstance(num_speakers, int) or num_speakers <= 0):
        raise ValueError("num_speakers must be a positive integer when provided")
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary when provided")
    return record


__all__ = [
    "DatasetRecord",
    "SAFE_MEETING_ID",
    "SUPPORTED_AUDIO_SUFFIXES",
    "expand_local_path",
    "validate_dataset_record",
]
