"""Overlap detection package facade."""

from src.overlap_detector import detect_overlap_segments, estimate_overlap_score, estimate_segment_overlap_scores

__all__ = ["detect_overlap_segments", "estimate_overlap_score", "estimate_segment_overlap_scores"]
