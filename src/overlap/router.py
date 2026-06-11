"""Route segments according to their estimated overlap."""

from src.utils import validate_score
from .detector import DEFAULT_OVERLAP_THRESHOLD


def route_segment(overlap_score: float, threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> str:
    """Return the processing path selected by an overlap threshold."""
    validate_score(overlap_score, "overlap_score")
    validate_score(threshold, "threshold")
    if overlap_score >= threshold:
        return "high_overlap_candidate"
    return "low_overlap_cluster"
