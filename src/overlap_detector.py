"""Overlap detection interfaces."""


def estimate_overlap_score(audio_path: str) -> float:
    """Estimate the probability of overlapping speech in an audio file."""
    # TODO: add a calibrated overlap-detection baseline.
    raise NotImplementedError("Overlap scoring is not implemented yet.")


def detect_overlap_segments(audio_path: str) -> list[dict]:
    """Return timestamped overlap regions and their overlap scores."""
    # TODO: run frame-level overlap detection and merge adjacent regions.
    raise NotImplementedError("Overlap segment detection is not implemented yet.")
