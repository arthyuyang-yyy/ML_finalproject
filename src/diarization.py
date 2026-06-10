"""Speaker diarization and clustering baselines."""


def diarize_audio(audio_path: str) -> list[dict]:
    """Return timestamped speaker labels and attribution confidence."""
    from .audio.preprocess import segment_audio

    return cluster_speakers(segment_audio(audio_path))


def cluster_speakers(segments: list[dict]) -> list[dict]:
    """Assign speaker-cluster labels to low-overlap speech segments."""
    clustered: list[dict] = []
    for index, segment in enumerate(segments):
        speaker = segment.get("speaker") or f"SPEAKER_{index % 2:02d}"
        confidence = float(segment.get("speaker_confidence", 0.55))
        clustered.append({
            **segment,
            "speaker": speaker,
            "speaker_confidence": max(0.0, min(1.0, confidence)),
        })
    return clustered
