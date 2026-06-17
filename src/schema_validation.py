"""Compatibility facade for canonical evidence validation."""

from .evidence.schema import (
    REQUIRED_CANDIDATE_FIELDS,
    REQUIRED_SEGMENT_FIELDS,
    VALID_PROCESSING_PATHS,
)
from .evidence.validator import (
    validate_candidate,
    validate_evidence_segments,
    validate_meeting,
    validate_metadata_segment,
)

__all__ = [
    "REQUIRED_CANDIDATE_FIELDS",
    "REQUIRED_SEGMENT_FIELDS",
    "VALID_PROCESSING_PATHS",
    "validate_candidate",
    "validate_evidence_segments",
    "validate_meeting",
    "validate_metadata_segment",
]
