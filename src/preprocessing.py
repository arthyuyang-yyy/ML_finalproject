"""Audio validation, normalization, and segmentation interfaces."""


def preprocess_audio(audio_path: str) -> str:
    """Prepare an audio file for downstream processing and return its path."""
    # TODO: validate format, normalize channels/sample rate, and run VAD.
    raise NotImplementedError("Audio preprocessing is not implemented yet.")


def segment_audio(audio_path: str) -> list[dict]:
    """Return timestamped speech segments for a preprocessed audio file."""
    # TODO: implement VAD-based segmentation.
    raise NotImplementedError("Audio segmentation is not implemented yet.")
