"""Energy-based overlap estimation fallback.

Used when pyannote overlap detection is unavailable.
"""

from typing import Any

import numpy as np


def estimate_with_energy_fallback(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    sample_rate: int,
) -> list[dict[str, Any]]:
    """Conservative fallback when no overlap model is configured."""
    samples = _to_mono(samples)
    scored: list[dict[str, Any]] = []
    for segment in segments:
        start = max(0, int(round(float(segment["start_time"]) * sample_rate)))
        end = max(start, int(round(float(segment["end_time"]) * sample_rate)))
        clip = samples[start:end]
        score = energy_overlap_proxy(clip, sample_rate)
        scored.append({
            **segment,
            "overlap_score": _round_score(score),
            "overlap_detector": "energy_fallback",
        })
    return scored


def energy_overlap_proxy(clip: np.ndarray, sample_rate: int) -> float:
    """Return a weak overlap proxy based on sustained high energy variation."""
    if clip.size == 0:
        return 0.0

    frame_length = max(1, int(round(0.025 * sample_rate)))
    hop_length = max(1, int(round(0.010 * sample_rate)))
    if clip.size < frame_length:
        return 0.0

    starts = np.arange(0, clip.size - frame_length + 1, hop_length, dtype=np.int64)
    rms = np.empty(starts.size, dtype=np.float32)
    for i, start in enumerate(starts):
        frame = clip[start : start + frame_length]
        rms[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2))

    peak = float(rms.max())
    if peak == 0.0:
        return 0.0

    high_energy_ratio = float(np.mean(rms >= peak * 0.75))
    median = float(np.median(rms))
    dynamic_ratio = float(np.std(rms) / (median + 1e-8))
    return min(0.39, 0.08 + 0.22 * high_energy_ratio + 0.08 * min(1.0, dynamic_ratio))


def _to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def _round_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 3)


__all__ = ["energy_overlap_proxy", "estimate_with_energy_fallback"]
