"""Candidate generation for ambiguous high-overlap segments."""


def generate_high_overlap_candidates(segment: dict) -> list[dict]:
    """Generate plausible transcript and speaker candidates for a segment.

    Each returned candidate must include ``transcript``,
    ``speaker_hypothesis``, ``confidence``, and ``uncertainty_note``.
    """
    # TODO: combine separated-stream ASR, decoding alternatives, and speakers.
    raise NotImplementedError("High-overlap candidate generation is not implemented yet.")
