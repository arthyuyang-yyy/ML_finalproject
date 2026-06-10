"""Low-overlap path: ASR plus speaker attribution."""

from typing import Any

import numpy as np

from .asr import ASRAdapter, transcribe_segments
from .audio.preprocess import TARGET_SAMPLE_RATE
from .diarization import assign_speakers_to_segments

LOW_OVERLAP_PATH = "low_overlap_cluster"


def process_low_overlap_segments(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    asr_adapter: ASRAdapter,
    sample_rate: int = TARGET_SAMPLE_RATE,
    diarization_turns: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return low-overlap segments with text, speaker labels, timestamps, and confidence."""
    if not segments:
        return []

    speaker_segments = assign_speakers_to_segments(segments, diarization_turns=diarization_turns)
    transcribed = transcribe_segments(samples, speaker_segments, adapter=asr_adapter, sample_rate=sample_rate)
    return [
        {
            **segment,
            "processing_path": LOW_OVERLAP_PATH,
            "candidates": [],
            "uncertainty_note": "",
        }
        for segment in transcribed
    ]
