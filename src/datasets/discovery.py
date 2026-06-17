"""Conservative local audio discovery for meeting datasets."""

import re
from pathlib import Path
from typing import Any

from .schema import DatasetRecord, SUPPORTED_AUDIO_SUFFIXES


ANNOTATION_SUFFIXES = (".rttm", ".stm", ".textgrid", ".txt", ".xml")


def discover_audio_records(
    root: str | Path,
    *,
    dataset: str,
    split: str,
    language: str,
    annotation_root: str | Path | None = None,
) -> list[DatasetRecord]:
    """Discover audio recursively without copying or rewriting dataset files."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"dataset root does not exist: {base}")

    annotation_files = _annotation_index(annotation_root)
    records: list[DatasetRecord] = []
    used_ids: set[str] = set()
    audio_files = sorted(path for path in base.rglob("*") if path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES)
    for audio_path in audio_files:
        relative = audio_path.relative_to(base)
        meeting_id = _unique_meeting_id(relative, used_ids)
        record: dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "meeting_id": meeting_id,
            "audio_path": str(audio_path),
            "language": language,
            "metadata": {"relative_audio_path": str(relative)},
        }
        annotation = _find_annotation(audio_path, annotation_files)
        if annotation is not None:
            record["annotation_path"] = str(annotation)
        records.append(record)  # type: ignore[arg-type]
    if not records:
        raise ValueError(f"no supported audio files found under {base}")
    return records


def _unique_meeting_id(relative_path: Path, used_ids: set[str]) -> str:
    parts = relative_path.with_suffix("").parts
    base_id = "__".join(_safe_part(part) for part in parts)
    candidate = base_id
    index = 2
    while candidate in used_ids:
        candidate = f"{base_id}_{index}"
        index += 1
    used_ids.add(candidate)
    return candidate


def _safe_part(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return normalized.strip("._-") or "meeting"


def _annotation_index(annotation_root: str | Path | None) -> dict[str, Path]:
    if annotation_root is None:
        return {}
    root = Path(annotation_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"annotation root does not exist: {root}")
    return {
        path.stem.lower(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ANNOTATION_SUFFIXES
    }


def _find_annotation(audio_path: Path, annotation_files: dict[str, Path]) -> Path | None:
    for suffix in ANNOTATION_SUFFIXES:
        candidate = audio_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    normalized_stem = re.sub(r"_(?:MS\d+|N_SPK\d+)$", "", audio_path.stem, flags=re.IGNORECASE)
    return annotation_files.get(normalized_stem.lower())


__all__ = ["discover_audio_records"]
