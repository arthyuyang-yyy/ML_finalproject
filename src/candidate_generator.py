"""Candidate generation for ambiguous high-overlap segments."""


def generate_high_overlap_candidates(segment: dict) -> list[dict]:
    """Generate plausible transcript and speaker candidates for a segment.

    Each returned candidate includes ``candidate_id``, ``speaker``, ``text``,
    ``confidence``, and ``uncertainty_note``. This baseline preserves the
    top transcript and one alternate speaker hypothesis until heavier ASR or
    separation adapters can provide true decoding alternatives.
    """
    segment_id = str(segment.get("segment_id") or segment.get("evidence_id") or "segment")
    speaker = str(segment.get("speaker", "SPEAKER_00"))
    text = str(segment.get("text", ""))
    confidence = float(segment.get("asr_confidence", 0.5))
    alternate_speaker = "SPEAKER_01" if speaker == "SPEAKER_00" else "SPEAKER_00"
    return [
        {
            "candidate_id": f"{segment_id}_c1",
            "speaker": speaker,
            "text": text,
            "confidence": max(0.0, min(1.0, confidence)),
            "uncertainty_note": "Top ASR hypothesis retained for high-overlap segment.",
        },
        {
            "candidate_id": f"{segment_id}_c2",
            "speaker": alternate_speaker,
            "text": text,
            "confidence": max(0.0, min(1.0, confidence * 0.75)),
            "uncertainty_note": "Alternate speaker attribution preserved until separation is available.",
        },
    ]
