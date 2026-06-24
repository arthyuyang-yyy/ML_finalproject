"""Audio validation, normalization, file export, and VAD segmentation.

This module is upstream of overlap detection and routing: it turns a raw audio
file into normalized mono samples, optional float WAV output, and timestamped
speech segments. It combines the old pipeline-oriented implementation with the
newer audio-package location and higher-quality resampling path.
"""

from pathlib import Path
from typing import Any

import numpy as np

TARGET_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_HINT = "WAV, FLAC, OGG, MP3, M4A, AAC, MP4, or WMA"


def load_audio(
    audio_path: str,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    target_peak: float = 0.97,
    denoise: bool = False,
    denoise_strength: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Decode an audio file, then return normalized mono float32 samples.

    ``soundfile`` handles native formats first. Other containers/codecs such as
    M4A, AAC, MP4, and WMA fall back to PyAV. PyAV preserves the native sample
    rate; mono conversion and resampling remain here to avoid double resampling.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file does not exist or is not a file: {audio_path}")
    samples, sample_rate = _decode_audio(path)
    samples = to_mono(samples)
    if denoise:
        samples = reduce_stationary_noise(samples, sample_rate, strength=denoise_strength)
    if sample_rate != target_sample_rate:
        samples = resample(samples, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    if normalize:
        samples = peak_normalize(samples, target_peak=target_peak)
    return samples, sample_rate


def _decode_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Prefer soundfile and fall back to PyAV for compressed/container formats."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without the backend
        raise ImportError(
            "load_audio requires the optional 'soundfile' backend; "
            "install it with `pip install soundfile`."
        ) from exc

    try:
        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    except (sf.LibsndfileError, RuntimeError):
        try:
            return decode_audio_with_pyav(audio_path)
        except ImportError as exc:
            raise RuntimeError(
                f"unable to decode audio file '{audio_path}'. "
                f"soundfile could not decode '{audio_path.name}', and PyAV is unavailable. "
                "Install it with `pip install av`, or convert the file to WAV."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"unable to decode audio file '{audio_path}'. Ensure it contains a valid audio "
                f"stream in a supported format ({SUPPORTED_AUDIO_HINT})."
            ) from exc


def decode_audio_with_pyav(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Demux and decode the first audio stream to native-rate float32 PCM."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV is required to decode this audio container") from exc

    chunks: list[np.ndarray] = []
    with av.open(str(audio_path)) as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError("input container does not contain an audio stream")
        sample_rate = int(stream.codec_context.sample_rate or stream.rate or 0)
        if sample_rate <= 0:
            raise ValueError("decoded audio stream does not expose a valid sample rate")
        layout = stream.codec_context.layout
        layout_name = layout.name if layout is not None else "mono"
        decoder = av.audio.resampler.AudioResampler(
            format="fltp",
            layout=layout_name,
            rate=sample_rate,
        )
        for frame in container.decode(stream):
            for decoded in decoder.resample(frame):
                chunks.append(np.asarray(decoded.to_ndarray(), dtype=np.float32))
        for decoded in decoder.resample(None):
            chunks.append(np.asarray(decoded.to_ndarray(), dtype=np.float32))

    if not chunks:
        raise ValueError("decoded audio stream contains no samples")
    return np.concatenate(chunks, axis=1).T.astype(np.float32), sample_rate


def reduce_stationary_noise(samples: np.ndarray, sample_rate: int, strength: float = 0.5) -> np.ndarray:
    """Optionally reduce stationary noise; disabled by default."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("denoise strength must be in [0, 1]")
    try:
        import noisereduce as nr
    except ImportError as exc:
        raise ImportError(
            "denoising is enabled but 'noisereduce' is unavailable; "
            "install it with `pip install noisereduce` or disable denoising"
        ) from exc
    reduced = nr.reduce_noise(
        y=np.asarray(samples, dtype=np.float32),
        sr=sample_rate,
        stationary=True,
        prop_decrease=strength,
    )
    return np.asarray(reduced, dtype=np.float32)


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


def silero_vad(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    threshold: float = 0.5,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 200,
    max_segment_s: float = 15.0,
) -> list[tuple[float, float]]:
    """Detect speech regions with the silero VAD bundled in ``faster-whisper``.

    The speech/non-speech decision comes from a learned model rather than an
    energy fraction of the clip's peak, so a single loud transient (a door slam
    or mic bump in a far-field meeting) no longer inflates the threshold and
    starves the detector. The silero model operates at 16 kHz, which matches the
    pipeline's target sample rate.

    Requires ``faster-whisper`` (a heavy backend).
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:  # pragma: no cover - only without the heavy backend
        raise ImportError(
            "silero VAD needs faster-whisper; install it with `pip install faster-whisper`."
        ) from exc

    mono = to_mono(samples).astype(np.float32)
    if mono.size == 0:
        return []
    options = VadOptions(
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        max_speech_duration_s=max_segment_s,
    )
    timestamps = get_speech_timestamps(mono, options)
    return [
        (round(ts["start"] / sample_rate, 3), round(ts["end"] / sample_rate, 3))
        for ts in timestamps
    ]


def segment_waveform(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    meeting_id: str = "meeting",
    segment_id_prefix: str | None = None,
    **vad_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run silero VAD and return lightweight, timestamped speech segments.

    Extra keyword arguments are forwarded to :func:`silero_vad`
    (``threshold``, ``min_silence_ms``, ``speech_pad_ms``, ``max_segment_s``).
    """
    regions = silero_vad(samples, sample_rate, **vad_kwargs)
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
    denoise: bool = False,
    denoise_strength: float = 0.5,
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
        denoise=denoise,
        denoise_strength=denoise_strength,
    )
    sf.write(output_path, samples, sample_rate, subtype="FLOAT")
    return samples, sample_rate


__all__ = [
    "TARGET_SAMPLE_RATE",
    "decode_audio_with_pyav",
    "load_audio",
    "peak_normalize",
    "preprocess_audio",
    "reduce_stationary_noise",
    "resample",
    "resample_linear",
    "segment_audio",
    "segment_waveform",
    "silero_vad",
    "to_mono",
]
