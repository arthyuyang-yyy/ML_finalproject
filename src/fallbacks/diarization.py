"""Deterministic speaker assignment when no diarization backend exists."""


def cluster_speakers(segments: list[dict]) -> list[dict]:
    """Assign speaker labels without a diarization model."""
    clustered: list[dict] = []
    for segment in segments:
        speaker = segment.get("speaker") or "UNKNOWN"
        confidence = float(segment.get("speaker_confidence", 0.78))
        clustered.append({
            **segment,
            "speaker": speaker,
            "speaker_confidence": max(0.0, min(1.0, confidence)),
        })
    return clustered


__all__ = ["cluster_speakers"]
