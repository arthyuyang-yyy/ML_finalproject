"""Overlap detection package facade."""

from src.overlap_detector import (
    DEFAULT_OVERLAP_THRESHOLD,
    detect_overlap_segments,
    detect_pyannote_overlap_regions,
    estimate_overlap_score,
    estimate_segment_overlap_scores,
)

__all__ = [
    "DEFAULT_OVERLAP_THRESHOLD",
    "detect_overlap_segments",
    "detect_pyannote_overlap_regions",
    "estimate_overlap_score",
    "estimate_segment_overlap_scores",
]
