"""Evidence validation entry points and batch error reporting."""

from pathlib import Path
from typing import Any

from src.schema_validation import validate_candidate, validate_meeting, validate_metadata_segment


def validate_evidence_segments(
    segments: list[dict[str, Any]],
    require_audio_clips: bool = False,
) -> list[str]:
    """Return all validation errors instead of stopping at the first record."""
    errors: list[str] = []
    if not isinstance(segments, list):
        return ["evidence segments must be a list"]
    if not segments:
        return ["evidence segments must not be empty"]

    for index, segment in enumerate(segments):
        try:
            validate_metadata_segment(segment)
        except ValueError as exc:
            errors.append(f"segment[{index}]: {exc}")
            continue

        if segment["processing_path"] == "low_overlap_cluster" and not segment["text"].strip():
            errors.append(f"segment[{index}]: low-overlap evidence must contain text")
        if segment["processing_path"] == "high_overlap_candidate" and not segment["uncertainty_note"].strip():
            errors.append(f"segment[{index}]: high-overlap evidence must explain uncertainty")
        if require_audio_clips and not Path(segment["audio_clip_path"]).is_file():
            errors.append(f"segment[{index}]: audio clip does not exist: {segment['audio_clip_path']}")
    return errors


__all__ = [
    "validate_candidate",
    "validate_evidence_segments",
    "validate_meeting",
    "validate_metadata_segment",
]
