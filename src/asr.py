"""Automatic speech recognition interfaces."""


def transcribe_audio(audio_path: str) -> dict:
    """Return transcript text, timestamps, and confidence for an audio file."""
    # TODO: add a replaceable ASR adapter; do not hard-code Whisper.
    raise NotImplementedError("ASR transcription is not implemented yet.")
