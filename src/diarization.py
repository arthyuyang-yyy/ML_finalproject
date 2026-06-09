"""Speaker diarization and clustering interfaces."""


def diarize_audio(audio_path: str) -> list[dict]:
    """Return timestamped speaker labels and attribution confidence."""
    # TODO: implement a lightweight baseline before adding pyannote adapters.
    raise NotImplementedError("Speaker diarization is not implemented yet.")


def cluster_speakers(segments: list[dict]) -> list[dict]:
    """Assign speaker-cluster labels to low-overlap speech segments."""
    # TODO: extract speaker embeddings and cluster them.
    raise NotImplementedError("Speaker clustering is not implemented yet.")
