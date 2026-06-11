"""Overlap scoring and overlapped-speech detection adapters.

The preferred path uses pyannote's overlapped speech detection model and scores
each VAD segment by the fraction of its duration covered by detected overlap.
When pyannote or a Hugging Face token is unavailable, a conservative energy
fallback keeps the demo runnable without pretending to be a calibrated overlap
model.
"""

import os
from typing import Any

import numpy as np

from src.audio.preprocess import TARGET_SAMPLE_RATE, energy_vad, load_audio
from src.diarization.core import _load_pyannote_pipeline
from src.fallbacks.overlap import estimate_with_energy_fallback

DEFAULT_OVERLAP_THRESHOLD = 0.4
PYANNOTE_OVERLAP_MODEL = "pyannote/overlapped-speech-detection"

OverlapRegion = tuple[float, float]


def estimate_segment_overlap_scores(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    sample_rate: int = TARGET_SAMPLE_RATE,
    audio_path: str | None = None,
    overlap_regions: list[OverlapRegion] | None = None,
    diarization_turns: list[dict[str, Any]] | None = None,
    asr_instability: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Attach ``overlap_score`` to each VAD segment.

    If explicit ``overlap_regions`` are supplied, or pyannote can detect them
    from ``audio_path``, the score is overlap-covered duration divided by the
    segment duration. Otherwise a conservative energy fallback produces a weak
    uncertainty signal while marking the detector source accordingly.
    """
    if not segments:
        return []

    regions = overlap_regions
    detector = "provided_regions" if regions is not None else "energy_fallback"
    if regions is None and audio_path:
        regions = detect_pyannote_overlap_regions(audio_path)
        if regions is not None:
            detector = "pyannote"

    if regions is not None:
        base = [
            {**segment, "overlap_score": _round_score(_overlap_fraction(segment, regions)), "overlap_detector": detector}
            for segment in segments
        ]
        return _fuse_overlap_signals(base, diarization_turns or [], asr_instability or {})

    base = estimate_with_energy_fallback(samples, segments, sample_rate)
    return _fuse_overlap_signals(base, diarization_turns or [], asr_instability or {})


def detect_pyannote_overlap_regions(
    audio_path: str,
    model_name: str = PYANNOTE_OVERLAP_MODEL,
    auth_token: str | None = None,
) -> list[OverlapRegion] | None:
    """Return pyannote overlapped-speech regions, or ``None`` if unavailable."""
    token = auth_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        return None

    try:
        pipeline = _load_pyannote_pipeline(model_name, token)
        output = pipeline(audio_path)
    except Exception:
        return None

    return _coerce_pyannote_regions(output)


def _fuse_overlap_signals(
    segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
    asr_instability: dict[str, float],
) -> list[dict[str, Any]]:
    """Fuse OSD/energy, diarization overlap, speaker changes, and ASR instability."""
    fused: list[dict[str, Any]] = []
    for segment in segments:
        base = float(segment["overlap_score"])
        diarization = _diarization_overlap_fraction(segment, diarization_turns)
        speaker_change = _speaker_change_signal(segment, diarization_turns)
        instability = max(0.0, min(1.0, float(asr_instability.get(str(segment.get("segment_id", "")), 0.0))))
        available = bool(diarization_turns) or bool(asr_instability)
        score = max(base, 0.55 * base + 0.25 * diarization + 0.10 * speaker_change + 0.10 * instability) if available else base
        fused.append({
            **segment,
            "overlap_score": _round_score(score),
            "overlap_components": {
                "osd_or_energy": _round_score(base),
                "diarization_overlap": _round_score(diarization),
                "speaker_change": _round_score(speaker_change),
                "asr_instability": _round_score(instability),
            },
        })
    return fused


def _diarization_overlap_fraction(
    segment: dict[str, Any], diarization_turns: list[dict[str, Any]]
) -> float:
    """Approximate simultaneous multi-speaker coverage from diarization turns."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    duration = max(0.0, end - start)
    if duration == 0.0:
        return 0.0
    events: list[tuple[float, int]] = []
    for turn in diarization_turns:
        left = max(start, float(turn["start_time"]))
        right = min(end, float(turn["end_time"]))
        if right > left:
            events.extend([(left, 1), (right, -1)])
    active = 0
    previous = start
    overlap = 0.0
    for timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        if active >= 2:
            overlap += timestamp - previous
        active += delta
        previous = timestamp
    return min(1.0, overlap / duration)


def _speaker_change_signal(segment: dict[str, Any], diarization_turns: list[dict[str, Any]]) -> float:
    """Return a bounded signal for speaker transitions inside a segment."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    speakers = [
        str(turn["speaker"])
        for turn in sorted(diarization_turns, key=lambda item: float(item["start_time"]))
        if min(end, float(turn["end_time"])) > max(start, float(turn["start_time"]))
    ]
    changes = sum(left != right for left, right in zip(speakers, speakers[1:]))
    return min(1.0, changes / 2.0)


def estimate_overlap_score(audio_path: str) -> float:
    """Estimate the maximum per-segment overlap score in an audio file."""
    samples, sample_rate = load_audio(audio_path)
    regions = energy_vad(samples, sample_rate)
    segments = [
        {"segment_id": f"overlap_seg_{index + 1:03d}", "start_time": start, "end_time": end}
        for index, (start, end) in enumerate(regions)
    ]
    scored = estimate_segment_overlap_scores(samples, segments, sample_rate, audio_path=audio_path)
    if not scored:
        return 0.0
    return round(max(float(segment["overlap_score"]) for segment in scored), 3)


def detect_overlap_segments(audio_path: str, threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> list[dict]:
    """Return timestamped VAD segments routed as high-overlap candidates."""
    samples, sample_rate = load_audio(audio_path)
    regions = energy_vad(samples, sample_rate)
    segments = [
        {"segment_id": f"overlap_seg_{index + 1:03d}", "start_time": start, "end_time": end}
        for index, (start, end) in enumerate(regions)
    ]
    return [
        segment
        for segment in estimate_segment_overlap_scores(samples, segments, sample_rate, audio_path=audio_path)
        if float(segment["overlap_score"]) >= threshold
    ]


def _overlap_fraction(segment: dict[str, Any], overlap_regions: list[OverlapRegion]) -> float:
    """Compute how much of one VAD segment is covered by overlap regions."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    duration = max(0.0, end - start)
    if duration == 0.0:
        return 0.0

    covered = 0.0
    for region_start, region_end in _merge_regions(overlap_regions):
        intersection_start = max(start, region_start)
        intersection_end = min(end, region_end)
        if intersection_end > intersection_start:
            covered += intersection_end - intersection_start
    return min(1.0, covered / duration)


def _merge_regions(regions: list[OverlapRegion]) -> list[OverlapRegion]:
    """Merge overlapping or touching overlap regions."""
    if not regions:
        return []
    ordered = sorted((float(start), float(end)) for start, end in regions if end > start)
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _coerce_pyannote_regions(output: Any) -> list[OverlapRegion]:
    """Convert common pyannote outputs into ``[(start, end), ...]``."""
    regions: list[OverlapRegion] = []

    if hasattr(output, "itertracks"):
        for item in output.itertracks(yield_label=True):
            segment = item[0]
            regions.append((float(segment.start), float(segment.end)))
    elif hasattr(output, "get_timeline"):
        for segment in output.get_timeline():
            regions.append((float(segment.start), float(segment.end)))
    elif isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                regions.append((float(item["start"]), float(item["end"])))
            else:
                regions.append((float(item[0]), float(item[1])))

    return _merge_regions(regions)


def _round_score(value: float) -> float:
    """Clamp and round an overlap score."""
    return round(max(0.0, min(1.0, float(value))), 3)
