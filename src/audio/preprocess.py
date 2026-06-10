"""Audio validation, normalization, file export, and VAD segmentation.

This module is upstream of overlap detection and routing: it turns a raw audio
file into normalized mono samples, optional float WAV output, and timestamped
speech segments. It combines the old pipeline-oriented implementation with the
newer audio-package location and higher-quality resampling path.
"""

from typing import Any

import numpy as np

TARGET_SAMPLE_RATE = 16000


def load_audio(
    audio_path: str,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    target_peak: float = 0.97,
) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 samples resampled to the target rate."""
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
        samples = resample(samples, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    if normalize:
        samples = peak_normalize(samples, target_peak=target_peak)
    return samples, sample_rate


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average multi-channel audio down to a single channel."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        return samples.mean(axis=1).astype(np.float32)
    return samples


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono signal, preferring polyphase filtering when SciPy exists."""
    samples = np.asarray(samples, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)

    try:
        from scipy.signal import resample_poly
    except ImportError:
        return resample_linear(samples, source_rate, target_rate)

    gcd = int(np.gcd(source_rate, target_rate))
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(samples, up, down).astype(np.float32)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono signal with dependency-free linear interpolation."""
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
    min_speech_ms: float = 1000.0,
    min_silence_ms: float = 300.0,
    speech_pad_ms: float = 50.0,
    merge_short_gap_s: float = 1.0,
    max_segment_s: float = 20.0,
    target_segment_s: float = 12.0,
) -> list[tuple[float, float]]:
    """Detect meeting-friendly speech regions with an energy-threshold VAD.

    The raw frame decisions are post-processed so meeting segments are useful
    downstream: adjacent pauses are bridged, short neighboring segments are
    merged where possible, isolated sub-second fragments are dropped, and long
    regions are split to keep clips around the 5-15 second range without
    exceeding ``max_segment_s``.
    """
    samples = to_mono(samples)
    if samples.size == 0:
        return []
    if min_speech_ms <= 0 or min_silence_ms < 0 or speech_pad_ms < 0:
        raise ValueError("VAD duration thresholds must be non-negative, with positive min_speech_ms")
    if merge_short_gap_s < 0 or max_segment_s <= 0 or target_segment_s <= 0:
        raise ValueError("VAD merge/split settings must be non-negative, with positive segment lengths")

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

    min_speech_s = min_speech_ms / 1000.0
    regions = _bridge_short_gaps(regions, min_silence_ms / 1000.0)
    regions = _merge_short_regions(regions, min_speech_s, merge_short_gap_s)
    regions = [r for r in regions if (r[1] - r[0]) >= min_speech_s]
    regions = _pad_regions(regions, speech_pad_ms / 1000.0, duration_s)
    return _split_long_regions(regions, max_segment_s, target_segment_s)


def segment_waveform(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    meeting_id: str = "meeting",
    segment_id_prefix: str | None = None,
    **vad_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run VAD and return lightweight, timestamped speech segments."""
    regions = energy_vad(samples, sample_rate, **vad_kwargs)
    prefix = segment_id_prefix or f"{meeting_id}_seg"
    return [
        {
            "meeting_id": meeting_id,
            "segment_id": f"{prefix}_{index + 1:03d}",
            "start_time": round(start, 3),
            "end_time": round(end, 3),
        }
        for index, (start, end) in enumerate(regions)
    ]


def segment_audio(audio_path: str, meeting_id: str = "meeting", **vad_kwargs: Any) -> list[dict[str, Any]]:
    """Load a file and return its timestamped speech segments."""
    samples, sample_rate = load_audio(audio_path)
    return segment_waveform(samples, sample_rate, meeting_id, **vad_kwargs)


def preprocess_audio(
    input_path: str,
    output_path: str,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    target_peak: float = 0.97,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load, mono-convert, resample, normalize, and write a float32 WAV file.

    ``target_sr`` is accepted for compatibility with the first
    ``src/audio/preprocess.py`` implementation.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without the backend
        raise ImportError(
            "preprocess_audio requires the optional 'soundfile' backend; "
            "install it with `pip install soundfile`."
        ) from exc

    if target_sr is not None:
        target_sample_rate = target_sr
    samples, sample_rate = load_audio(
        input_path,
        target_sample_rate=target_sample_rate,
        normalize=True,
        target_peak=target_peak,
    )
    sf.write(output_path, samples, sample_rate, subtype="FLOAT")
    return samples, sample_rate


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


def _merge_short_regions(
    regions: list[tuple[float, float]],
    min_duration_s: float,
    max_gap_s: float,
) -> list[tuple[float, float]]:
    """Merge sub-minimum regions with close neighbors before filtering."""
    if not regions:
        return []

    merged: list[tuple[float, float]] = []
    i = 0
    while i < len(regions):
        start, end = regions[i]
        while (end - start) < min_duration_s and i + 1 < len(regions):
            next_start, next_end = regions[i + 1]
            if next_start - end > max_gap_s:
                break
            end = next_end
            i += 1
        if merged and (end - start) < min_duration_s and start - merged[-1][1] <= max_gap_s:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
        i += 1
    return merged


def _split_long_regions(
    regions: list[tuple[float, float]],
    max_segment_s: float,
    target_segment_s: float,
) -> list[tuple[float, float]]:
    """Split long speech regions into bounded meeting-sized chunks."""
    split: list[tuple[float, float]] = []
    chunk_s = min(max_segment_s, target_segment_s)
    for start, end in regions:
        duration = end - start
        if duration <= max_segment_s:
            split.append((start, end))
            continue

        cursor = start
        while end - cursor > max_segment_s:
            next_end = min(cursor + chunk_s, end)
            split.append((cursor, next_end))
            cursor = next_end
        if end > cursor:
            split.append((cursor, end))
    return split


__all__ = [
    "TARGET_SAMPLE_RATE",
    "energy_vad",
    "frame_rms",
    "load_audio",
    "peak_normalize",
    "preprocess_audio",
    "resample",
    "resample_linear",
    "segment_audio",
    "segment_waveform",
    "to_mono",
]
