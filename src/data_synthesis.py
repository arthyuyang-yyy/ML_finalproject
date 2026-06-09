"""Synthesize controlled overlapping-speech mixtures with ground-truth labels.

Following the data-construction approach used in the reference work, this module
mixes single-speaker clips at a controlled overlap duration and SNR, and emits
ground-truth annotations (per-speaker segments, overlap regions, overlap ratio)
aligned with ``data/annotations/annotation_template.csv``.

It is a neutral utility: it makes no algorithmic decision and is consumed by
evaluation, candidate-generation, and routing experiments alike. Everything
operates on NumPy arrays and is testable with synthetic signals.
"""

from typing import Any

import numpy as np

TARGET_SAMPLE_RATE = 16000

ANNOTATION_COLUMNS = [
    "meeting_id",
    "segment_id",
    "start_time",
    "end_time",
    "speaker",
    "text",
    "is_overlap",
    "overlap_type",
    "topic",
    "decision",
    "action_item",
]


def signal_power(samples: np.ndarray) -> float:
    """Return the mean-square power of a signal."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.mean(samples**2))


def scale_to_snr(reference: np.ndarray, signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``signal`` so its power gives ``snr_db`` relative to ``reference``.

    SNR is defined as ``10*log10(P_reference / P_signal)``; ``snr_db=0`` yields
    equal power. A silent signal is returned unchanged.
    """
    ref_power = signal_power(reference)
    sig_power = signal_power(signal)
    if sig_power == 0.0 or ref_power == 0.0:
        return np.asarray(signal, dtype=np.float32)
    target_power = ref_power / (10.0 ** (snr_db / 10.0))
    gain = np.sqrt(target_power / sig_power)
    return (np.asarray(signal, dtype=np.float32) * gain).astype(np.float32)


def mix_two_speakers(
    speaker_a: np.ndarray,
    speaker_b: np.ndarray,
    overlap_s: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
    snr_db: float = 0.0,
    speaker_a_label: str = "SPEAKER_00",
    speaker_b_label: str = "SPEAKER_01",
    meeting_id: str = "synth",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mix two single-speaker clips with a controlled overlap and SNR.

    Speaker A starts at ``t=0``; speaker B starts so the two clips overlap by
    ``overlap_s`` seconds (``overlap_s=0`` is back-to-back). Speaker B is scaled
    to ``snr_db`` relative to A. Returns ``(mixture, annotation)`` where the
    annotation contains per-speaker segments, overlap regions, and overlap ratio.
    """
    if overlap_s < 0:
        raise ValueError("overlap_s must be >= 0")
    a = np.asarray(speaker_a, dtype=np.float32)
    b = np.asarray(speaker_b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("speaker_a and speaker_b must be 1-D mono signals")

    b = scale_to_snr(a, b, snr_db)
    overlap_samples = min(int(round(overlap_s * sample_rate)), a.size, b.size)
    start_b = a.size - overlap_samples
    total_len = max(a.size, start_b + b.size)

    mixture = np.zeros(total_len, dtype=np.float32)
    mixture[: a.size] += a
    mixture[start_b : start_b + b.size] += b

    seg_a = {"speaker": speaker_a_label, "start": 0.0, "end": a.size / sample_rate}
    seg_b = {"speaker": speaker_b_label, "start": start_b / sample_rate, "end": (start_b + b.size) / sample_rate}
    annotation = build_annotation([seg_a, seg_b], sample_rate, total_len, meeting_id)
    return mixture, annotation


def overlap_intervals(segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return time intervals where two or more speaker segments are active."""
    events: list[tuple[float, int]] = []
    for seg in segments:
        events.append((float(seg["start"]), 1))
        events.append((float(seg["end"]), -1))
    events.sort()

    intervals: list[tuple[float, float]] = []
    active = 0
    region_start: float | None = None
    for time, delta in events:
        was_overlapping = active >= 2
        active += delta
        now_overlapping = active >= 2
        if not was_overlapping and now_overlapping:
            region_start = time
        elif was_overlapping and not now_overlapping and region_start is not None:
            if time > region_start:
                intervals.append((region_start, time))
            region_start = None
    return intervals


def total_overlap_duration(segments: list[dict[str, Any]]) -> float:
    """Total seconds during which two or more speakers overlap."""
    return float(sum(end - start for start, end in overlap_intervals(segments)))


def overlap_ratio(segments: list[dict[str, Any]], duration_s: float) -> float:
    """Fraction of the session duration that contains overlapping speech."""
    if duration_s <= 0:
        return 0.0
    return min(1.0, total_overlap_duration(segments) / duration_s)


def build_annotation(
    segments: list[dict[str, Any]],
    sample_rate: int,
    total_samples: int,
    meeting_id: str,
) -> dict[str, Any]:
    """Assemble a ground-truth annotation dict for a synthesized mixture."""
    duration_s = total_samples / sample_rate
    overlaps = overlap_intervals(segments)
    labeled_segments = []
    for index, seg in enumerate(segments):
        seg_overlap = _overlap_seconds(seg, overlaps)
        labeled_segments.append({
            "meeting_id": meeting_id,
            "segment_id": f"{meeting_id}-{index:04d}",
            "speaker": seg["speaker"],
            "start_time": round(float(seg["start"]), 3),
            "end_time": round(float(seg["end"]), 3),
            "is_overlap": seg_overlap > 0.0,
            "overlap_type": _classify_overlap(seg, seg_overlap),
        })
    return {
        "meeting_id": meeting_id,
        "sample_rate": sample_rate,
        "duration": round(duration_s, 3),
        "segments": labeled_segments,
        "overlap_regions": [
            {"start_time": round(s, 3), "end_time": round(e, 3)} for s, e in overlaps
        ],
        "overlap_duration": round(total_overlap_duration(segments), 3),
        "overlap_ratio": round(overlap_ratio(segments, duration_s), 3),
    }


def to_annotation_rows(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten an annotation into CSV-style rows matching ANNOTATION_COLUMNS."""
    rows = []
    for seg in annotation["segments"]:
        row = {column: "" for column in ANNOTATION_COLUMNS}
        row.update({
            "meeting_id": seg["meeting_id"],
            "segment_id": seg["segment_id"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "speaker": seg["speaker"],
            "is_overlap": seg["is_overlap"],
            "overlap_type": seg["overlap_type"],
        })
        rows.append(row)
    return rows


def _overlap_seconds(segment: dict[str, Any], overlaps: list[tuple[float, float]]) -> float:
    """Seconds of ``segment`` that fall inside any overlap interval."""
    start, end = float(segment["start"]), float(segment["end"])
    total = 0.0
    for o_start, o_end in overlaps:
        total += max(0.0, min(end, o_end) - max(start, o_start))
    return total


def _classify_overlap(segment: dict[str, Any], overlap_seconds: float) -> str:
    """Label a segment as 'none', 'full', or 'partial' overlap."""
    duration = float(segment["end"]) - float(segment["start"])
    if overlap_seconds <= 0.0 or duration <= 0.0:
        return "none"
    if overlap_seconds >= duration - 1e-6:
        return "full"
    return "partial"
