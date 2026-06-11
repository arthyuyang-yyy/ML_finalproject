"""Canonical evidence construction and validation facade."""

from .builder import build_evidence_file, build_evidence_segments, build_metadata_segment
from .schema import Candidate, EVIDENCE_SEGMENT_FIELDS, EvidenceSegment
from .validator import (
    validate_candidate,
    validate_evidence_segments,
    validate_meeting,
    validate_metadata_segment,
)

__all__ = [
    "Candidate",
    "EVIDENCE_SEGMENT_FIELDS",
    "EvidenceSegment",
    "build_metadata_segment",
    "build_evidence_file",
    "build_evidence_segments",
    "validate_candidate",
    "validate_evidence_segments",
    "validate_meeting",
    "validate_metadata_segment",
]
