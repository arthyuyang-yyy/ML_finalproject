"""Validate the shared metadata (evidence-packet) records used across the pipeline.

The canonical record shape is produced by
:func:`src.metadata_builder.build_metadata_segment`. These validators let any
module (data loaders, fixtures, baseline outputs) check that a record conforms
to the schema before it flows into routing, LLM post-processing, or memory.
"""

from typing import Any

from .utils import validate_score

VALID_PROCESSING_PATHS = {"low_overlap_cluster", "high_overlap_candidate"}

REQUIRED_SEGMENT_FIELDS = {
    "meeting_id": str,
    "segment_id": str,
    "evidence_id": str,
    "speaker": str,
    "start_time": (int, float),
    "end_time": (int, float),
    "text": str,
    "processing_path": str,
    "route_reason": str,
    "overlap_score": (int, float),
    "asr_confidence": (int, float),
    "speaker_confidence": (int, float),
    "audio_clip_path": str,
    "source_audio_path": str,
    "language": str,
    "candidates": list,
    "uncertainty_note": str,
}

REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id": str,
    "speaker": str,
    "text": str,
    "confidence": (int, float),
    "uncertainty_note": str,
}


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
        if record["speaker"] != "MIXED":
            raise ValueError("high_overlap_candidate segments must use speaker='MIXED'")
        if record["text"].strip():
            raise ValueError("high_overlap_candidate segments must keep the primary text empty")
        if not record["uncertainty_note"].strip():
            raise ValueError("high_overlap_candidate segments must explain their uncertainty")
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


def _type_name(expected_type: Any) -> str:
    """Return a readable name for a type or tuple of types."""
    if isinstance(expected_type, tuple):
        return " or ".join(t.__name__ for t in expected_type)
    return expected_type.__name__
