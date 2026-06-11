"""Candidate generation for ambiguous high-overlap segments.

High-overlap audio should not be collapsed into one deterministic transcript.
The preferred path uses faster-whisper multiple times with different decoding
settings, then keeps distinct transcript/speaker hypotheses as candidates.
"""

import os
import tempfile
from functools import lru_cache
from typing import Any

import numpy as np

from src.audio.preprocess import TARGET_SAMPLE_RATE

DEFAULT_DECODE_CONFIGS: list[dict[str, Any]] = [
    {"beam_size": 1, "temperature": 0.0, "language": None},
    {"beam_size": 5, "temperature": 0.0, "language": None},
    {"beam_size": 5, "temperature": 0.4, "language": "zh"},
    {"beam_size": 5, "temperature": 0.8, "language": "en"},
]


def generate_high_overlap_candidates(
    segment: dict,
    samples: np.ndarray | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    language: str | None = None,
    decode_configs: list[dict[str, Any]] | None = None,
    max_candidates: int = 4,
    speaker_hypotheses: list[str] | None = None,
) -> list[dict]:
    """Generate transcript/speaker candidates for one high-overlap segment.

    If ``samples`` are provided and faster-whisper is installed, the clip is
    decoded with several beam/temperature/language settings. Otherwise the
    function returns conservative fallback candidates so downstream uncertainty
    handling and schemas remain usable during lightweight demos.
    """
    configs = _with_language_override(decode_configs or DEFAULT_DECODE_CONFIGS, language)
    if not speaker_hypotheses and str(segment.get("speaker", "")) not in {"", "MIXED"}:
        speaker_hypotheses = [str(segment["speaker"])]
    if samples is not None and samples.size:
        candidates = _generate_with_faster_whisper(
            segment, samples, sample_rate, configs, max_candidates, speaker_hypotheses
        )
        if candidates:
            return candidates
    return _fallback_candidates(segment, configs[:2], speaker_hypotheses)


def _generate_with_faster_whisper(
    segment: dict,
    samples: np.ndarray,
    sample_rate: int,
    decode_configs: list[dict[str, Any]],
    max_candidates: int,
    speaker_hypotheses: list[str] | None,
) -> list[dict]:
    """Run faster-whisper with multiple decode settings when available."""
    try:
        import soundfile as sf
        import faster_whisper  # noqa: F401 - availability check
    except ImportError:
        return []

    model_name = os.environ.get("FASTER_WHISPER_MODEL", "small")
    device = os.environ.get("FASTER_WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")

    try:
        model = _load_faster_whisper_model(model_name, device, compute_type)
    except Exception:
        return []

    segment_id = str(segment.get("segment_id") or segment.get("evidence_id") or "segment")
    seen_text: set[str] = set()
    candidates: list[dict] = []
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, samples, sample_rate, subtype="FLOAT")
        for config in decode_configs:
            try:
                decoded_segments, _ = model.transcribe(
                    tmp.name,
                    beam_size=int(config["beam_size"]),
                    temperature=float(config["temperature"]),
                    language=config.get("language"),
                )
                decoded = list(decoded_segments)
            except Exception:
                continue

            text = " ".join(str(seg.text).strip() for seg in decoded if str(seg.text).strip()).strip()
            normalized = " ".join(text.lower().split())
            if not normalized or normalized in seen_text:
                continue
            seen_text.add(normalized)

            confidence = _confidence_from_decoded_segments(decoded)
            index = len(candidates) + 1
            candidates.append({
                "candidate_id": f"{segment_id}_c{index}",
                "speaker": _speaker_hypothesis(index, speaker_hypotheses),
                "text": text,
                "confidence": confidence,
                "uncertainty_note": _decode_note(config, backend="faster-whisper"),
                "decode_config": {
                    "beam_size": int(config["beam_size"]),
                    "temperature": float(config["temperature"]),
                    "language": config.get("language") or "auto",
                },
            })
            if len(candidates) >= max_candidates:
                break
    return candidates


def _fallback_candidates(
    segment: dict,
    decode_configs: list[dict[str, Any]],
    speaker_hypotheses: list[str] | None = None,
) -> list[dict]:
    """Return explicit uncertainty-preserving candidates when ASR backend is absent."""
    segment_id = str(segment.get("segment_id") or segment.get("evidence_id") or "segment")
    text = str(segment.get("text", "")).strip()
    if not text:
        text = "[high-overlap transcript candidate pending ASR decode]"

    return [
        {
            "candidate_id": f"{segment_id}_c{index}",
            "speaker": _speaker_hypothesis(index, speaker_hypotheses),
            "text": text,
            "confidence": round(max(0.0, 0.62 - 0.07 * (index - 1)), 3),
            "uncertainty_note": _decode_note(config, backend="fallback"),
            "decode_config": {
                "beam_size": int(config["beam_size"]),
                "temperature": float(config["temperature"]),
                "language": config.get("language") or "auto",
            },
        }
        for index, config in enumerate(decode_configs, start=1)
    ]


def _confidence_from_decoded_segments(decoded_segments: list[Any]) -> float:
    """Best-effort confidence for faster-whisper segment objects."""
    scores: list[float] = []
    for segment in decoded_segments:
        avg_logprob = getattr(segment, "avg_logprob", None)
        no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
        if avg_logprob is None:
            continue
        scores.append(float(np.exp(float(avg_logprob))) * (1.0 - max(0.0, min(1.0, float(no_speech_prob)))))
    if not scores:
        return 0.5
    return round(max(0.0, min(1.0, sum(scores) / len(scores))), 3)


def _with_language_override(configs: list[dict[str, Any]], language: str | None) -> list[dict[str, Any]]:
    """Use requested language for the auto config while keeping zh/en probes."""
    if not language or language in {"und", "unknown", "auto"}:
        return configs
    updated = [dict(config) for config in configs]
    updated[0]["language"] = language
    return updated


def _speaker_hypothesis(index: int, speakers: list[str] | None = None) -> str:
    """Use diarization-supported speakers instead of inventing identities."""
    normalized = [str(speaker) for speaker in speakers or [] if str(speaker).strip()]
    if not normalized:
        return "UNKNOWN"
    return normalized[(index - 1) % len(normalized)]


@lru_cache(maxsize=4)
def _load_faster_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    """Load each faster-whisper model configuration once per process."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


generate_candidates = generate_high_overlap_candidates


def _decode_note(config: dict[str, Any], backend: str) -> str:
    """Human-readable uncertainty note for one candidate."""
    language = config.get("language") or "auto"
    return (
        "High-overlap segment; speaker attribution is uncertain. "
        f"Candidate generated by {backend} decode "
        f"(beam_size={config['beam_size']}, temperature={config['temperature']}, language={language})."
    )
