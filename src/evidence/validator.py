"""Evidence validation entry points: strict single-record, batch, and per-meeting."""

from pathlib import Path
from typing import Any

from .schema import (
    REQUIRED_CANDIDATE_FIELDS,
    REQUIRED_SEGMENT_FIELDS,
    VALID_PROCESSING_PATHS,
    _type_name,
)
from src.utils import validate_score


# -- single-record validators (raise on first problem) -----------------------


def validate_candidate(candidate: Any, index: int = 0) -> dict[str, Any]:
    """Validate a single high-overlap candidate dict.

    A candidate must carry a stable candidate ID, text, speaker hypothesis,
    confidence in ``[0, 1]``, and an uncertainty note explaining why the
    interpretation is ambiguous.
    """
    if not isinstance(candidate, dict):
        raise ValueError(f"candidate[{index}] must be a dict, got {type(candidate).__name__}")

    for field, expected_type in REQUIRED_CANDIDATE_FIELDS.items():
        if field not in candidate:
            raise ValueError(f"candidate[{index}] is missing required field '{field}'")
        if not isinstance(candidate[field], expected_type) or (
            field == "confidence" and isinstance(candidate[field], bool)
        ):
            raise ValueError(
                f"candidate[{index}].{field} must be {_type_name(expected_type)}, "
                f"got {type(candidate[field]).__name__}"
            )

    validate_score(candidate["confidence"], f"candidate[{index}].confidence")
    for field in ("candidate_id", "speaker", "text", "uncertainty_note"):
        if not candidate[field].strip():
            raise ValueError(f"candidate[{index}].{field} must be a non-empty string")
    if "decode_config" in candidate and not isinstance(candidate["decode_config"], dict):
        raise ValueError(f"candidate[{index}].decode_config must be a dict if present")
    return candidate


def validate_metadata_segment(record: Any) -> dict[str, Any]:
    """Validate one metadata segment (evidence packet) and return it unchanged.

    Raises ``ValueError`` describing the first problem found. Checks required
    fields and types, score ranges, time ordering, the processing path, and the
    structure of every candidate. High-overlap segments must keep at least one
    candidate so downstream modules can reason over the preserved uncertainty.
    """
    if not isinstance(record, dict):
        raise ValueError(f"segment must be a dict, got {type(record).__name__}")

    for field, expected_type in REQUIRED_SEGMENT_FIELDS.items():
        if field not in record:
            raise ValueError(f"segment is missing required field '{field}'")
        if not isinstance(record[field], expected_type) or isinstance(record[field], bool):
            raise ValueError(
                f"segment.{field} must be {_type_name(expected_type)}, "
                f"got {type(record[field]).__name__}"
            )

    if record["end_time"] < record["start_time"]:
        raise ValueError("segment.end_time must not be earlier than segment.start_time")
    if record["end_time"] == record["start_time"]:
        raise ValueError("segment duration must be greater than zero")

    for name in ("meeting_id", "segment_id", "evidence_id", "speaker"):
        if not record[name].strip():
            raise ValueError(f"segment.{name} must be a non-empty string")

    if record["processing_path"] not in VALID_PROCESSING_PATHS:
        raise ValueError(
            f"segment.processing_path must be one of {sorted(VALID_PROCESSING_PATHS)}, "
            f"got '{record['processing_path']}'"
        )

    for name in ("overlap_score", "asr_confidence", "speaker_confidence"):
        validate_score(record[name], f"segment.{name}")

    for index, candidate in enumerate(record["candidates"]):
        validate_candidate(candidate, index)

    if record["processing_path"] == "high_overlap_candidate" and not record["candidates"]:
        raise ValueError(
            "high_overlap_candidate segments must keep at least one candidate "
            "to preserve uncertainty for downstream reasoning"
        )
    if record["processing_path"] == "high_overlap_candidate":
        if not record["uncertainty_note"].strip():
            raise ValueError("high_overlap_candidate segments must explain their uncertainty")
        if not record["text"].strip() and record.get("source") in {"llm_resolved", "fallback_resolved"}:
            raise ValueError("resolved high_overlap_candidate segments must contain transcript text")
    else:
        if not record["text"].strip():
            raise ValueError("low_overlap_cluster segments must contain transcript text")
        if record["candidates"]:
            raise ValueError("low_overlap_cluster segments must not contain candidates")
        if record["uncertainty_note"].strip():
            raise ValueError("low_overlap_cluster segments must not carry an uncertainty note")

    return record


def validate_meeting(segments: Any) -> list[dict[str, Any]]:
    """Validate a list of metadata segments belonging to one or more meetings."""
    if not isinstance(segments, list):
        raise ValueError(f"meeting must be a list of segments, got {type(segments).__name__}")
    if not segments:
        raise ValueError("meeting must contain at least one segment")
    validated = [validate_metadata_segment(segment) for segment in segments]
    meeting_ids = {segment["meeting_id"] for segment in validated}
    if len(meeting_ids) != 1:
        raise ValueError("meeting segments must share one meeting_id")
    if len({segment["segment_id"] for segment in validated}) != len(validated):
        raise ValueError("meeting contains duplicate segment_id values")
    if len({segment["evidence_id"] for segment in validated}) != len(validated):
        raise ValueError("meeting contains duplicate evidence_id values")
    return validated


# -- batch validator (collects errors) ---------------------------------------


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

        if require_audio_clips and not Path(segment["audio_clip_path"]).is_file():
            errors.append(f"segment[{index}]: audio clip does not exist: {segment['audio_clip_path']}")
    return errors


__all__ = [
    "validate_candidate",
    "validate_evidence_segments",
    "validate_meeting",
    "validate_metadata_segment",
]
