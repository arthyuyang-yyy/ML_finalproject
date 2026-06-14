"""High-overlap path: preserve multiple candidates instead of one transcript."""

from typing import Any

import numpy as np

from .audio.preprocess import TARGET_SAMPLE_RATE
from .candidates.generator import generate_high_overlap_candidates
from .speech_separation import SpeechSeparator, separate_waveform

HIGH_OVERLAP_PATH = "high_overlap_candidate"
HIGH_OVERLAP_SPEAKER = "MIXED"
HIGH_OVERLAP_SPEAKER_CONFIDENCE = 0.35


def process_high_overlap_segments(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    sample_rate: int = TARGET_SAMPLE_RATE,
    language: str | None = None,
    diarization_turns: list[dict[str, Any]] | None = None,
    separate: bool = False,
    separator: SpeechSeparator | None = None,
) -> list[dict[str, Any]]:
    """Return high-overlap records with empty main text and candidate hypotheses.

    When ``separate`` is set, each clip is first split into per-speaker streams by
    the optional speech-separation baseline and candidates are generated from each
    cleaner stream; otherwise candidates come from the mixed clip directly.
    """
    processed: list[dict[str, Any]] = []
    for segment in segments:
        clip = _slice_segment(samples, segment, sample_rate)
        candidate_source = {
            **segment,
            "speaker": HIGH_OVERLAP_SPEAKER,
            "text": "",
            "asr_confidence": 0.0,
            "speaker_confidence": HIGH_OVERLAP_SPEAKER_CONFIDENCE,
            "processing_path": HIGH_OVERLAP_PATH,
        }
        speaker_hypotheses = _speakers_for_segment(segment, diarization_turns or [])
        if separate:
            candidates = _candidates_from_separated_streams(
                candidate_source, clip, sample_rate, language, speaker_hypotheses, separator
            )
        else:
            candidates = generate_high_overlap_candidates(
                candidate_source,
                samples=clip,
                sample_rate=sample_rate,
                language=language,
                speaker_hypotheses=speaker_hypotheses,
            )
        processed.append({
            **segment,
            "speaker": HIGH_OVERLAP_SPEAKER,
            "text": "",
            "processing_path": HIGH_OVERLAP_PATH,
            "asr_confidence": _aggregate_candidate_confidence(candidates),
            "speaker_confidence": HIGH_OVERLAP_SPEAKER_CONFIDENCE,
            "candidates": candidates,
            "separation_applied": bool(separate),
            "uncertainty_note": "High-overlap segment; speaker attribution is uncertain.",
        })
    return processed


def _candidates_from_separated_streams(
    candidate_source: dict[str, Any],
    clip: np.ndarray,
    sample_rate: int,
    language: str | None,
    speaker_hypotheses: list[str],
    separator: SpeechSeparator | None,
) -> list[dict[str, Any]]:
    """Separate the clip and generate candidates from each per-speaker stream.

    Each stream is tagged with one diarization speaker hypothesis (when known) so
    candidates keep traceable, non-invented speaker labels. Falls back to the
    mixed clip if separation yields nothing usable.
    """
    num_sources = max(2, len(speaker_hypotheses) or 2)
    streams = separate_waveform(clip, sample_rate, num_sources, backend=separator) if clip.size else []

    candidates: list[dict[str, Any]] = []
    for index, stream in enumerate(streams):
        stream_speaker = speaker_hypotheses[index] if index < len(speaker_hypotheses) else None
        candidates.extend(
            generate_high_overlap_candidates(
                candidate_source,
                samples=np.asarray(stream),
                sample_rate=sample_rate,
                language=language,
                speaker_hypotheses=[stream_speaker] if stream_speaker else None,
            )
        )
    if candidates:
        return _renumber_candidates(candidate_source, candidates)
    return generate_high_overlap_candidates(
        candidate_source,
        samples=clip,
        sample_rate=sample_rate,
        language=language,
        speaker_hypotheses=speaker_hypotheses,
    )


def _renumber_candidates(
    segment: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Give merged per-stream candidates unique, contiguous candidate IDs."""
    segment_id = str(segment.get("segment_id") or segment.get("evidence_id") or "segment")
    renumbered: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        renumbered.append({**candidate, "candidate_id": f"{segment_id}_c{index}"})
    return renumbered


def _speakers_for_segment(
    segment: dict[str, Any],
    diarization_turns: list[dict[str, Any]],
) -> list[str]:
    """Return diarization speakers that overlap the high-overlap segment."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    speakers: list[str] = []
    for turn in diarization_turns:
        if min(end, float(turn["end_time"])) <= max(start, float(turn["start_time"])):
            continue
        speaker = str(turn["speaker"])
        if speaker not in speakers:
            speakers.append(speaker)
    return speakers


def _slice_segment(samples: np.ndarray, segment: dict[str, Any], sample_rate: int) -> np.ndarray:
    """Extract one segment waveform from full meeting samples."""
    start = max(0, int(round(float(segment["start_time"]) * sample_rate)))
    end = max(start, int(round(float(segment["end_time"]) * sample_rate)))
    return samples[start:end]


def _aggregate_candidate_confidence(candidates: list[dict[str, Any]]) -> float:
    """Use the best candidate confidence as the segment-level ASR confidence."""
    if not candidates:
        return 0.0
    return round(max(float(candidate.get("confidence", 0.0)) for candidate in candidates), 3)
