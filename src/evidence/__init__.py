"""Evidence schema package facade."""

from src.metadata_builder import build_metadata_segment
from src.schema_validation import validate_candidate, validate_meeting, validate_metadata_segment

__all__ = ["build_metadata_segment", "validate_candidate", "validate_meeting", "validate_metadata_segment"]
