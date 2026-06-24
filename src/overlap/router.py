"""Route segments according to their estimated overlap."""

from src.overlap.detector import DEFAULT_OVERLAP_THRESHOLD, MIN_AUTHORITATIVE_OVERLAP_SECONDS
from src.utils import validate_score


AUTHORITATIVE_OVERLAP_DETECTORS = {"pyannote", "provided_regions"}


def route_segment(
    overlap_score: float,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    *,
    overlap_detector: str | None = None,
    overlap_seconds: float = 0.0,
    min_authoritative_overlap_seconds: float = MIN_AUTHORITATIVE_OVERLAP_SECONDS,
) -> str:
    """Return the processing path selected by overlap evidence.

    pyannote/provided overlap regions are already thresholded by the detector.
    For those sources, any substantive overlap hit should route to the
    high-overlap path even when it covers less than the generic score threshold.
    """
    validate_score(overlap_score, "overlap_score")
    validate_score(threshold, "threshold")
    if overlap_seconds < 0.0:
        raise ValueError("overlap_seconds must be non-negative")
    if min_authoritative_overlap_seconds < 0.0:
        raise ValueError("min_authoritative_overlap_seconds must be non-negative")
    if (
        overlap_detector in AUTHORITATIVE_OVERLAP_DETECTORS
        and overlap_seconds >= min_authoritative_overlap_seconds
    ):
        return "high_overlap_candidate"
    if overlap_score >= threshold:
        return "high_overlap_candidate"
    return "low_overlap_cluster"
