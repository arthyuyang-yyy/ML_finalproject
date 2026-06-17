"""High-overlap path: generate candidates before resolver finalization."""

from typing import Any

import numpy as np

from .audio.preprocess import TARGET_SAMPLE_RATE
from .candidates.generator import (
    generate_high_overlap_candidates,
    generate_separated_source_candidates,
)
from .speech_separation import SpeechSeparationAdapter, separate_waveform

HIGH_OVERLAP_PATH = "high_overlap_candidate"
HIGH_OVERLAP_SPEAKER = "MIXED"
HIGH_OVERLAP_SPEAKER_CONFIDENCE = 0.35


def process_high_overlap_segments(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    sample_rate: int = TARGET_SAMPLE_RATE,
    language: str | None = None,
    diarization_turns: list[dict[str, Any]] | None = None,
    separation_adapter: SpeechSeparationAdapter | None = None,
    asr_config: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return high-overlap candidate records before LLM/fallback resolution."""
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
        sources = separate_waveform(clip, sample_rate, separation_adapter)
        candidates = generate_separated_source_candidates(
            candidate_source,
            sources,
            sample_rate=sample_rate,
            language=language,
            separation_backend=getattr(separation_adapter, "name", "none"),
            asr_config=asr_config,
        )
        # Supplement with multi-decode hypotheses unless separation already gave
        # at least two usable candidates AND one per separated source. A single
        # separated source (e.g. a SepFormer run that recovers one stream) must
        # still keep multiple candidates: a high-overlap segment that collapses
        # to one candidate would violate the preserve-uncertainty requirement.
        if len(candidates) < 2 or len(candidates) < len(sources):
            fallback = generate_high_overlap_candidates(
                candidate_source,
                samples=clip,
                sample_rate=sample_rate,
                language=language,
                speaker_hypotheses=speaker_hypotheses,
                asr_config=asr_config,
            )
            candidates = _merge_candidates(candidates, fallback)
        processed.append({
            **segment,
            "speaker": HIGH_OVERLAP_SPEAKER,
            "text": "",
            "processing_path": HIGH_OVERLAP_PATH,
            "asr_confidence": _aggregate_candidate_confidence(candidates),
            "speaker_confidence": HIGH_OVERLAP_SPEAKER_CONFIDENCE,
            "candidates": candidates,
            "uncertainty_note": "High-overlap segment; speaker attribution is uncertain.",
        })
    return processed


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


def _normalized_text(candidate: dict[str, Any]) -> str:
    """Lowercased, whitespace-collapsed transcript for duplicate detection."""
    return " ".join(str(candidate.get("text", "")).lower().split())


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    """Dedup key combining transcript and speaker hypothesis.

    Same text under a *different* speaker is a distinct interpretation (e.g.
    ``SEPARATED_SOURCE_01`` vs a diarization speaker) and must be preserved to
    keep the speaker uncertainty that the high-overlap path exists to carry.
    """
    return (_normalized_text(candidate), str(candidate.get("speaker", "")).strip())


def _merge_candidates(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append secondary candidates, dropping those that duplicate a primary one.

    A duplicate means same transcript *and* same speaker; identical text under a
    different speaker is kept. Duplicates *within* secondary are also kept:
    fallback/multi-decode hypotheses may legitimately share a transcript while
    differing in decode settings.
    """
    seen = {key for key in (_candidate_key(c) for c in primary) if key[0]}
    merged = list(primary)
    for candidate in secondary:
        key = _candidate_key(candidate)
        if key[0] and key in seen:
            continue
        merged.append(candidate)
    return merged


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
