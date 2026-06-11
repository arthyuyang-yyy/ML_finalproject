"""High-overlap speech separation interfaces."""


def separate_speakers(audio_path: str) -> list[str]:
    """Return paths to separated speaker streams for an audio file."""
    # TODO: add a replaceable speech-separation adapter.
    raise NotImplementedError("Speech separation is not implemented yet.")
