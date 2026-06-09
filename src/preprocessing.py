"""Audio validation, normalization, and VAD segmentation.

This module is upstream of overlap detection and routing: it turns a raw audio
file into normalized mono samples and a list of timestamped speech segments.
Those segments later receive speaker labels, transcripts, and confidence scores
in :mod:`src.metadata_builder`.

The waveform-level functions take plain NumPy arrays so they can be tested with
synthetic signals, without GPU or model downloads. File loading is kept behind a
lazy import so the module stays importable even when an audio backend is absent.
"""

from typing import Any

import numpy as np

TARGET_SAMPLE_RATE = 16000


def load_audio(audio_path: str, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 samples resampled to the target rate.

    Requires the optional ``soundfile`` backend. Returns ``(samples, sample_rate)``.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without the backend
        raise ImportError(
            "load_audio requires the optional 'soundfile' backend; "
            "install it with `pip install soundfile`."
        ) from exc

    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    samples = to_mono(samples)
    if sample_rate != target_sample_rate:
        samples = resample_linear(samples, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return samples, sample_rate


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average multi-channel audio down to a single channel."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        return samples.mean(axis=1).astype(np.float32)
    return samples


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono signal with linear interpolation.

    A dependency-free baseline; a higher-quality resampler can replace it later
    without changing the public interface.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.size / source_rate
    target_len = int(round(duration * target_rate))
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    source_times = np.arange(samples.size) / source_rate
    target_times = np.arange(target_len) / target_rate
    return np.interp(target_times, source_times, samples).astype(np.float32)


def peak_normalize(samples: np.ndarray, target_peak: float = 0.97) -> np.ndarray:
    """Scale a signal so its largest absolute amplitude equals ``target_peak``."""
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return (samples * (target_peak / peak)).astype(np.float32)


def frame_rms(samples: np.ndarray, frame_length: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame RMS energy and each frame's start sample index."""
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if samples.size < frame_length:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    starts = np.arange(0, samples.size - frame_length + 1, hop_length, dtype=np.int64)
    rms = np.empty(starts.size, dtype=np.float32)
    for i, start in enumerate(starts):
        frame = samples[start : start + frame_length]
        rms[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    return rms, starts


def energy_vad(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    threshold_ratio: float = 0.3,
    min_speech_ms: float = 200.0,
    min_silence_ms: float = 150.0,
    speech_pad_ms: float = 50.0,
) -> list[tuple[float, float]]:
    """Detect speech regions with an energy-threshold baseline VAD.

    Frames are marked speech when their RMS exceeds ``threshold_ratio`` of the
    peak frame RMS. Short gaps (< ``min_silence_ms``) are bridged, short regions
    (< ``min_speech_ms``) are dropped, and each region is padded by
    ``speech_pad_ms``. Returns ``[(start_s, end_s), ...]`` in seconds.
    """
    samples = to_mono(samples)
    if samples.size == 0:
        return []

    frame_length = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    hop_length = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    rms, starts = frame_rms(samples, frame_length, hop_length)
    if rms.size == 0:
        return []

    peak = float(rms.max())
    if peak == 0.0:
        return []
    voiced = rms >= (threshold_ratio * peak)

    duration_s = samples.size / sample_rate
    regions: list[tuple[float, float]] = []
    region_start: float | None = None
    for i, is_voiced in enumerate(voiced):
        frame_start_s = float(starts[i]) / sample_rate
        if is_voiced and region_start is None:
            region_start = frame_start_s
        elif not is_voiced and region_start is not None:
            regions.append((region_start, frame_start_s))
            region_start = None
    if region_start is not None:
        regions.append((region_start, duration_s))

    regions = _bridge_short_gaps(regions, min_silence_ms / 1000.0)
    regions = [r for r in regions if (r[1] - r[0]) >= (min_speech_ms / 1000.0)]
    return _pad_regions(regions, speech_pad_ms / 1000.0, duration_s)


def segment_waveform(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    meeting_id: str = "meeting",
    **vad_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run VAD and return lightweight, timestamped speech segments.

    Each segment carries ``meeting_id``, ``segment_id``, ``start_time``, and
    ``end_time``. Speaker labels, text, and confidence scores are filled in by
    later pipeline stages before becoming full metadata records.
    """
    regions = energy_vad(samples, sample_rate, **vad_kwargs)
    return [
        {
            "meeting_id": meeting_id,
            "segment_id": f"{meeting_id}-{index:04d}",
            "start_time": round(start, 3),
            "end_time": round(end, 3),
        }
        for index, (start, end) in enumerate(regions)
    ]


def segment_audio(audio_path: str, meeting_id: str = "meeting", **vad_kwargs: Any) -> list[dict[str, Any]]:
    """Load a file and return its timestamped speech segments (requires soundfile)."""
    samples, sample_rate = load_audio(audio_path)
    return segment_waveform(samples, sample_rate, meeting_id, **vad_kwargs)


def _bridge_short_gaps(regions: list[tuple[float, float]], min_silence_s: float) -> list[tuple[float, float]]:
    """Merge adjacent regions separated by a gap shorter than ``min_silence_s``."""
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_silence_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _pad_regions(regions: list[tuple[float, float]], pad_s: float, duration_s: float) -> list[tuple[float, float]]:
    """Pad each region by ``pad_s`` on both sides, clamped to the signal bounds."""
    padded: list[tuple[float, float]] = []
    for start, end in regions:
        padded.append((max(0.0, start - pad_s), min(duration_s, end + pad_s)))
    return _bridge_short_gaps(padded, 1e-9)
