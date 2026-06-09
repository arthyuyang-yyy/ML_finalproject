"""Route segments according to their estimated overlap."""


def route_segment(overlap_score: float, threshold: float = 0.5) -> str:
    """Return the processing path selected by an overlap threshold."""
    if not 0.0 <= overlap_score <= 1.0:
        raise ValueError("overlap_score must be between 0.0 and 1.0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    if overlap_score >= threshold:
        return "high_overlap_candidate"
    return "low_overlap_cluster"
