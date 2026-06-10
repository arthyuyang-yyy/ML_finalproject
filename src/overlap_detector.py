"""Lightweight overlap detection baselines."""

from typing import Any

import numpy as np

from .audio.preprocess import TARGET_SAMPLE_RATE, energy_vad, load_audio, to_mono


def estimate_segment_overlap_scores(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Attach deterministic overlap scores to VAD segments.

    This is a lightweight placeholder for pyannote/WhisperX overlap detection.
    It uses segment duration and relative RMS energy as weak cues so the rest of
    the pipeline can exercise routing, candidates, and evidence handling now.
    """
    samples = to_mono(samples)
    if not segments:
        return []

    rms_values: list[float] = []
    for segment in segments:
        start = max(0, int(round(float(segment["start_time"]) * sample_rate)))
        end = max(start, int(round(float(segment["end_time"]) * sample_rate)))
        clip = samples[start:end]
        rms_values.append(float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))) if clip.size else 0.0)

    peak_rms = max(rms_values) or 1.0
    scored: list[dict[str, Any]] = []
    for segment, rms in zip(segments, rms_values):
        duration = max(0.0, float(segment["end_time"]) - float(segment["start_time"]))
        duration_score = min(1.0, duration / 8.0)
        energy_score = min(1.0, rms / peak_rms)
        overlap_score = round(0.15 + 0.35 * duration_score + 0.25 * energy_score, 3)
        scored.append({**segment, "overlap_score": min(0.95, overlap_score)})
    return scored


def estimate_overlap_score(audio_path: str) -> float:
    """Estimate the probability of overlapping speech in an audio file."""
    samples, sample_rate = load_audio(audio_path)
    segments = energy_vad(samples, sample_rate)
    scored = estimate_segment_overlap_scores(
        samples,
        [{"start_time": start, "end_time": end} for start, end in segments],
        sample_rate,
    )
    if not scored:
        return 0.0
    return round(max(float(segment["overlap_score"]) for segment in scored), 3)


def detect_overlap_segments(audio_path: str) -> list[dict]:
    """Return timestamped overlap regions and their overlap scores."""
    samples, sample_rate = load_audio(audio_path)
    regions = energy_vad(samples, sample_rate)
    segments = [
        {"segment_id": f"overlap-{index:04d}", "start_time": start, "end_time": end}
        for index, (start, end) in enumerate(regions)
    ]
    return [
        segment
        for segment in estimate_segment_overlap_scores(samples, segments, sample_rate)
        if float(segment["overlap_score"]) >= 0.5
    ]
