"""Explicit error classes for backend availability and output quality."""


class BackendUnavailableError(Exception):
    """A required external dependency or model backend is not installed.

    Raised when a package is missing (e.g. ``pyannote.audio``) or a Hugging
    Face model cannot be loaded. Callers may catch this only when an explicit
    fallback policy is configured.
    """


class BackendExecutionError(Exception):
    """A backend initialised but failed during inference.

    Raised when a model was loaded successfully but ``transcribe``, ``predict``,
    or ``generate`` raised an exception.
    """


class BackendOutputError(Exception):
    """A backend returned output that cannot be parsed or validated.

    Raised when the raw response is not valid JSON, contains missing fields, or
    references evidence/document IDs that do not exist.
    """


__all__ = ["BackendUnavailableError", "BackendExecutionError", "BackendOutputError"]
