"""Audio clip export helpers for evidence segments."""

from pathlib import Path
from typing import Any

import numpy as np


def write_segment_clips(
    samples: np.ndarray,
    sample_rate: int,
    segments: list[dict[str, Any]],
    clips_dir: str | Path,
) -> list[dict[str, Any]]:
    """Write one float WAV clip per segment and return updated segments."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - only exercised without backend
        raise ImportError(
            "write_segment_clips requires the optional 'soundfile' backend; "
            "install it with `pip install soundfile`."
        ) from exc

    output_dir = Path(clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    for segment in segments:
        evidence_id = str(segment.get("evidence_id", segment.get("segment_id")))
        start = max(0, int(round(float(segment["start_time"]) * sample_rate)))
        end = max(start, int(round(float(segment["end_time"]) * sample_rate)))
        clip = samples[start:end]
        clip_path = output_dir / f"{evidence_id}.wav"
        sf.write(clip_path, clip, sample_rate, subtype="FLOAT")
        updated.append({**segment, "audio_clip_path": str(clip_path)})
    return updated
