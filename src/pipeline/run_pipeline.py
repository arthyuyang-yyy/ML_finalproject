"""End-to-end orchestration for one meeting audio file."""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.asr import get_adapter
from src.audio.clipper import write_segment_clips
from src.diarization import diarize_with_pyannote
from src.overlap.router import AUTHORITATIVE_OVERLAP_DETECTORS, route_segment
from src.evidence.builder import build_evidence_segments
from src.high_overlap import process_high_overlap_segments
from src.llm.event_extractor import extract_meeting_events
from src.llm.gemma_client import GemmaClient, create_gemma_client
from src.llm.resolver import resolve_high_overlap_segments
from src.low_overlap import process_low_overlap_segments
from src.memory.episodic_store import build_episodes, upsert_episodes
from src.overlap.detector import estimate_segment_overlap_scores
from src.audio.preprocess import preprocess_audio, segment_waveform
from src.evidence import validate_metadata_segment
from src.speech_separation import get_separation_adapter

from .config import PipelineConfig
from .io import ensure_meeting_dirs, write_json

logger = logging.getLogger("meeting_memory.pipeline")


@contextmanager
def _stage(name: str, timings: dict[str, float], counts: dict[str, int] | None = None) -> Iterator[None]:
    """Time a pipeline stage, log start/end, and persist timings incrementally.

    The accumulated ``timings`` dict is flushed to ``stage_timings.json`` after
    every stage so that long runs survive crashes and ``nohup``-style polling
    can read partial progress without re-running anything.
    """
    logger.info("[stage:%s] start", name)
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        timings[name] = round(dt, 3)
        logger.info("[stage:%s] done in %.2fs", name, dt)
        if counts is not None:
            logger.info("[stage:%s] counts=%s", name, json.dumps(counts, ensure_ascii=False))
        # Best-effort flush; ignore errors so a read-only filesystem doesn't kill the run.
        try:
            stage_path = timings.get("_path")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            stage_path = None


def _flush_stage_timings(out_dir: Path, timings: dict[str, float]) -> None:
    """Persist stage timings so the matrix runner can read partial progress."""
    try:
        (out_dir / "stage_timings.json").write_text(
            json.dumps(timings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to flush stage_timings.json: %s", exc)


def run_meeting_pipeline(
    input_audio_path: str,
    meeting_id: str,
    config: PipelineConfig | None = None,
    llm_client: GemmaClient | None = None,
) -> dict[str, Any]:
    """Run the lightweight meeting pipeline and write per-meeting artifacts."""
    cfg = config or PipelineConfig()
    llm_client = llm_client or create_gemma_client(
        cfg.gemma_backend,
        model=cfg.gemma_model,
        base_url=cfg.gemma_base_url,
    )
    paths = ensure_meeting_dirs(cfg.meeting_dir(meeting_id))
    base_dir = Path(paths["base"]).parent  # outputs/<meeting_id>/../
    stage_timings: dict[str, float] = {}
    logger.info(
        "[pipeline] meeting=%s audio=%s out=%s",
        meeting_id,
        input_audio_path,
        paths["base"],
    )

    with _stage("01_preprocess", stage_timings):
        preprocessed_samples, sample_rate = preprocess_audio(
            input_audio_path,
            str(paths["preprocessed"]),
            target_sample_rate=cfg.target_sample_rate,
            denoise=cfg.enable_denoise,
            denoise_strength=cfg.denoise_strength,
        )
    samples = preprocessed_samples
    _flush_stage_timings(base_dir, stage_timings)

    with _stage("02_vad", stage_timings):
        vad_segments = segment_waveform(
            samples,
            sample_rate,
            meeting_id=meeting_id,
            max_segment_s=cfg.vad_max_segment_s,
            speech_pad_ms=cfg.vad_speech_pad_ms,
            min_silence_ms=cfg.vad_min_silence_ms,
        )
        write_json(paths["vad_segments"], vad_segments)
    _flush_stage_timings(base_dir, stage_timings)
    logger.info("[stage:02_vad] n_segments=%d", len(vad_segments))

    with _stage("03_diarize", stage_timings):
        diarization_turns = diarize_with_pyannote(str(paths["preprocessed"])) or []
        write_json(paths["diarization"], diarization_turns)
    _flush_stage_timings(base_dir, stage_timings)
    logger.info("[stage:03_diarize] n_turns=%d", len(diarization_turns))

    with _stage("04_overlap_score", stage_timings):
        scored_segments = estimate_segment_overlap_scores(
            samples,
            vad_segments,
            sample_rate,
            audio_path=str(paths["preprocessed"]),
            diarization_turns=diarization_turns,
        )
        write_json(paths["overlap"], scored_segments)
    _flush_stage_timings(base_dir, stage_timings)

    with _stage("05_route_split", stage_timings):
        routed_segments = [
            _route_scored_segment(
                segment,
                threshold=cfg.overlap_threshold,
                suspected_threshold=cfg.suspected_overlap_threshold,
            )
            for segment in scored_segments
        ]
        routed_segments = split_segments_by_overlap_regions(
            routed_segments,
            min_segment_seconds=cfg.high_overlap_min_segment_s,
            decode_context_seconds=cfg.high_overlap_decode_context_s,
        )
        write_json(paths["routed_segments"], routed_segments)
    _flush_stage_timings(base_dir, stage_timings)
    n_low = sum(1 for s in routed_segments if s["processing_path"] == "low_overlap_cluster")
    n_high = sum(1 for s in routed_segments if s["processing_path"] == "high_overlap_candidate")
    logger.info("[stage:05_route_split] low=%d high=%d", n_low, n_high)

    low_overlap_input = [
        segment for segment in routed_segments if segment["processing_path"] == "low_overlap_cluster"
    ]
    high_overlap_input = [
        segment for segment in routed_segments if segment["processing_path"] == "high_overlap_candidate"
    ]
    suspected_high_overlap_input = [
        segment for segment in high_overlap_input if segment.get("route_mode") == "suspected_high_overlap"
    ]

    with _stage("06_low_overlap_asr", stage_timings):
        low_overlap_asr_adapter = get_adapter(cfg.low_overlap_asr_model, **_adapter_kwargs(cfg))
        low_overlap_processed = process_low_overlap_segments(
            samples,
            low_overlap_input,
            asr_adapter=low_overlap_asr_adapter,
            sample_rate=sample_rate,
            diarization_turns=diarization_turns,
            asr_context_padding_s=cfg.asr_context_padding_s,
        )
        suspected_baselines = process_low_overlap_segments(
            samples,
            suspected_high_overlap_input,
            asr_adapter=low_overlap_asr_adapter,
            sample_rate=sample_rate,
            diarization_turns=diarization_turns,
            asr_context_padding_s=cfg.asr_context_padding_s,
        )
        suspected_baseline_by_id = {
            str(segment["segment_id"]): segment
            for segment in suspected_baselines
            if str(segment.get("text", "")).strip()
        }
    _flush_stage_timings(base_dir, stage_timings)
    logger.info("[stage:06_low_overlap_asr] n_processed=%d", len(low_overlap_processed))

    with _stage("07_high_overlap", stage_timings):
        high_overlap_processed = process_high_overlap_segments(
            samples,
            high_overlap_input,
            sample_rate=sample_rate,
            language=cfg.language,
            diarization_turns=diarization_turns,
            separation_adapter=get_separation_adapter(
                cfg.speech_separation_backend,
                **_separation_adapter_kwargs(cfg),
            ),
            asr_config=_high_overlap_asr_config(cfg),
        )
        high_overlap_processed = [
            _attach_suspected_baseline(segment, suspected_baseline_by_id)
            for segment in high_overlap_processed
        ]
    _flush_stage_timings(base_dir, stage_timings)
    logger.info("[stage:07_high_overlap] n_processed=%d", len(high_overlap_processed))

    with _stage("08_resolve_llm", stage_timings):
        high_overlap_processed = resolve_high_overlap_segments(
            high_overlap_processed,
            client=llm_client,
            context_segments=low_overlap_processed,
            suspected_min_confidence_gain=cfg.suspected_overlap_min_confidence_gain,
            suspected_max_text_cer=cfg.suspected_overlap_max_text_cer,
        )
    _flush_stage_timings(base_dir, stage_timings)

    with _stage("09_build_evidence", stage_timings):
        evidence_segments = build_evidence_segments(
            low_overlap_processed,
            high_overlap_processed,
            meeting_id=meeting_id,
            source_audio_path=input_audio_path,
            language=cfg.language,
            overlap_threshold=cfg.overlap_threshold,
        )
    _flush_stage_timings(base_dir, stage_timings)
    logger.info("[stage:09_build_evidence] n_evidence=%d", len(evidence_segments))

    with _stage("10_clips_and_validate", stage_timings):
        evidence_segments = write_segment_clips(samples, sample_rate, evidence_segments, paths["clips"])
        evidence_segments = [
            validate_metadata_segment(segment, require_audio_clip=True)
            for segment in evidence_segments
        ]
    _flush_stage_timings(base_dir, stage_timings)

    low_overlap_segments = [
        segment for segment in evidence_segments if segment["processing_path"] == "low_overlap_cluster"
    ]
    high_overlap_segments = [
        segment for segment in evidence_segments if segment["processing_path"] == "high_overlap_candidate"
    ]

    with _stage("11_event_extraction", stage_timings):
        if evidence_segments:
            meeting_events = extract_meeting_events(evidence_segments, client=llm_client)
            episodic_memory = build_episodes(meeting_events, evidence_segments)
        else:
            meeting_events = {"meeting_id": meeting_id, "meeting_summary": "", "events": []}
            episodic_memory = []
    _flush_stage_timings(base_dir, stage_timings)
    logger.info("[stage:11_event_extraction] n_events=%d", len(meeting_events.get("events", [])))

    with _stage("12_persist", stage_timings):
        write_json(paths["low_overlap_segments"], low_overlap_segments)
        write_json(paths["high_overlap_candidates"], high_overlap_segments)
        write_json(paths["evidence_segments"], evidence_segments)
        write_json(paths["meeting_events"], meeting_events)
        write_json(paths["episodic_memory"], episodic_memory)
        long_term_memory = upsert_episodes(
            cfg.episodic_memory_path(),
            episodic_memory,
            meeting_id=meeting_id,
        )
    _flush_stage_timings(base_dir, stage_timings)
    logger.info(
        "[pipeline] done n_evidence=%d n_events=%d",
        len(evidence_segments),
        len(meeting_events.get("events", [])),
    )

    return {
        "meeting_id": meeting_id,
        "output_dir": str(paths["base"]),
        "artifacts": {
            **{name: str(path) for name, path in paths.items() if name != "base"},
            "long_term_episodic_memory": str(cfg.episodic_memory_path()),
        },
        "num_vad_segments": len(vad_segments),
        "num_evidence_segments": len(evidence_segments),
        "num_low_overlap_segments": len(low_overlap_segments),
        "num_high_overlap_segments": len(high_overlap_segments),
        "meeting_events": meeting_events,
        "episodic_memory": episodic_memory,
        "long_term_memory_size": len(long_term_memory),
        "preprocessed_num_samples": int(preprocessed_samples.size),
        "stage_timings_s": stage_timings,
    }


MIN_HIGH_OVERLAP_SEGMENT_SECONDS = 1.0


def _route_scored_segment(
    segment: dict[str, Any],
    *,
    threshold: float,
    suspected_threshold: float,
) -> dict[str, Any]:
    """Route strong and suspected high-overlap segments separately."""
    score = float(segment["overlap_score"])
    routed_path = route_segment(
        score,
        threshold=threshold,
        overlap_detector=segment.get("overlap_detector"),
        overlap_seconds=float(segment.get("overlap_seconds", 0.0)),
    )
    if routed_path == "high_overlap_candidate":
        route_mode = "high_overlap"
        reason = f"overlap_score={score:.3f} routed to high-overlap path"
    elif score >= suspected_threshold and _eligible_for_suspected_overlap(segment):
        routed_path = "high_overlap_candidate"
        route_mode = "suspected_high_overlap"
        reason = (
            f"overlap_score={score:.3f} >= suspected_threshold={suspected_threshold:.3f} "
            f"and < threshold={threshold:.3f}; routed to shadow high-overlap path"
        )
    else:
        route_mode = "low_overlap"
        reason = f"overlap_score={score:.3f} < suspected_threshold={suspected_threshold:.3f}; routed to low-overlap path"
    return {
        **segment,
        "processing_path": routed_path,
        "route_mode": route_mode,
        "route_reason": reason,
    }


def _eligible_for_suspected_overlap(segment: dict[str, Any]) -> bool:
    """Allow recall-oriented shadow routing only when the signal is not pure energy."""
    if segment.get("overlap_detector") in AUTHORITATIVE_OVERLAP_DETECTORS:
        return True
    components = segment.get("overlap_components")
    if not isinstance(components, dict):
        return False
    return any(
        float(components.get(name, 0.0)) > 0.0
        for name in ("diarization_overlap", "speaker_change", "asr_instability")
    )


def _attach_suspected_baseline(
    segment: dict[str, Any],
    baseline_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach low-overlap ASR baseline used to protect CER on suspected overlap."""
    if segment.get("route_mode") != "suspected_high_overlap":
        return segment
    baseline = baseline_by_id.get(str(segment.get("segment_id", "")))
    if not baseline:
        return {
            **segment,
            "uncertainty_note": (
                str(segment.get("uncertainty_note", "")).strip()
                or "Suspected high-overlap segment; no low-overlap baseline was available."
            ),
        }
    return {
        **segment,
        "baseline_text": str(baseline.get("text", "")),
        "baseline_speaker": str(baseline.get("speaker", segment.get("speaker", "UNKNOWN"))),
        "baseline_asr_confidence": float(baseline.get("asr_confidence", 0.0)),
        "baseline_speaker_confidence": float(baseline.get("speaker_confidence", 0.0)),
        "uncertainty_note": (
            str(segment.get("uncertainty_note", "")).strip()
            or "Suspected high-overlap segment; final text is gated against the low-overlap ASR baseline."
        ),
    }


def split_segments_by_overlap_regions(
    segments: list[dict[str, Any]],
    *,
    min_segment_seconds: float = MIN_HIGH_OVERLAP_SEGMENT_SECONDS,
    decode_context_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Split authoritative overlap hits into dedicated high-overlap subsegments."""
    split: list[dict[str, Any]] = []
    for segment in segments:
        if not _should_split_authoritative_overlap(segment):
            split.append(segment)
            continue
        split.extend(
            _split_one_segment_by_overlap_regions(
                segment,
                min_segment_seconds,
                decode_context_seconds,
            )
        )
    return sorted(split, key=lambda item: (float(item["start_time"]), float(item["end_time"]), str(item["segment_id"])))


def _should_split_authoritative_overlap(segment: dict[str, Any]) -> bool:
    return (
        segment.get("processing_path") == "high_overlap_candidate"
        and segment.get("overlap_detector") in AUTHORITATIVE_OVERLAP_DETECTORS
        and bool(segment.get("overlap_regions"))
    )


def _split_one_segment_by_overlap_regions(
    segment: dict[str, Any],
    min_segment_seconds: float,
    decode_context_seconds: float,
) -> list[dict[str, Any]]:
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    regions = _overlap_regions(segment)
    children: list[dict[str, Any]] = []
    cursor = start
    low_index = 1
    high_index = 1
    for overlap_start, overlap_end in regions:
        if overlap_start - cursor >= min_segment_seconds:
            children.append(_child_segment(segment, cursor, overlap_start, "low_overlap_cluster", low_index))
            low_index += 1
        overlap_duration = overlap_end - overlap_start
        decode_start: float | None = None
        decode_end: float | None = None
        if overlap_duration < min_segment_seconds:
            decode_start = max(start, overlap_start - decode_context_seconds)
            decode_end = min(end, overlap_end + decode_context_seconds)
            if decode_end - decode_start < min_segment_seconds:
                extra = (min_segment_seconds - (decode_end - decode_start)) / 2.0
                decode_start = max(start, decode_start - extra)
                decode_end = min(end, decode_end + extra)
        children.append(
            _child_segment(
                segment,
                overlap_start,
                overlap_end,
                "high_overlap_candidate",
                high_index,
                decode_start=decode_start,
                decode_end=decode_end,
                short_overlap_context_decode=overlap_duration < min_segment_seconds,
                min_segment_seconds=min_segment_seconds,
            )
        )
        high_index += 1
        cursor = max(cursor, overlap_end)
    if end - cursor >= min_segment_seconds:
        children.append(_child_segment(segment, cursor, end, "low_overlap_cluster", low_index))
    return children


def _overlap_regions(segment: dict[str, Any]) -> list[tuple[float, float]]:
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    regions: list[tuple[float, float]] = []
    for item in segment.get("overlap_regions", []):
        left = max(start, float(item[0]))
        right = min(end, float(item[1]))
        if right > left:
            regions.append((round(left, 3), round(right, 3)))
    regions.sort()
    merged: list[tuple[float, float]] = []
    for left, right in regions:
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            prev_left, prev_right = merged[-1]
            merged[-1] = (prev_left, max(prev_right, right))
    return merged


def _child_segment(
    parent: dict[str, Any],
    start: float,
    end: float,
    processing_path: str,
    index: int,
    *,
    decode_start: float | None = None,
    decode_end: float | None = None,
    short_overlap_context_decode: bool = False,
    min_segment_seconds: float = MIN_HIGH_OVERLAP_SEGMENT_SECONDS,
) -> dict[str, Any]:
    duration = round(end - start, 3)
    suffix = "ovl" if processing_path == "high_overlap_candidate" else "low"
    overlap_score = 1.0 if processing_path == "high_overlap_candidate" else 0.0
    overlap_seconds = duration if processing_path == "high_overlap_candidate" else 0.0
    parent_id = str(parent["segment_id"])
    if processing_path == "high_overlap_candidate":
        if short_overlap_context_decode:
            route_reason = (
                f"split short {duration:.3f}s pyannote/provided overlap from {parent_id}; "
                f"decoded with local context because it is below {min_segment_seconds:.3f}s"
            )
        else:
            route_reason = f"split {duration:.3f}s pyannote/provided overlap from {parent_id}; routed to {processing_path}"
    else:
        route_reason = f"split non-overlap region from {parent_id}; routed to {processing_path}"
    child = {
        **parent,
        "parent_segment_id": parent_id,
        "segment_id": f"{parent_id}_{suffix}_{index:02d}",
        "start_time": round(start, 3),
        "end_time": round(end, 3),
        "processing_path": processing_path,
        "route_mode": parent.get("route_mode", "high_overlap") if processing_path == "high_overlap_candidate" else "low_overlap",
        "route_reason": route_reason,
        "overlap_score": overlap_score,
        "overlap_seconds": overlap_seconds,
        "overlap_regions": [[round(start, 3), round(end, 3)]] if processing_path == "high_overlap_candidate" else [],
        "parent_overlap_score": parent.get("overlap_score"),
        "parent_overlap_seconds": parent.get("overlap_seconds"),
    }
    if decode_start is not None and decode_end is not None and processing_path == "high_overlap_candidate":
        child.update({
            "decode_start_time": round(decode_start, 3),
            "decode_end_time": round(decode_end, 3),
            "short_overlap_context_decode": short_overlap_context_decode,
        })
    return child


def _adapter_language(language: str) -> str | None:
    """Map project-level unknown language labels to adapter-friendly values."""
    return None if language in {"", "und", "unknown"} else language


def _adapter_kwargs(config: PipelineConfig) -> dict[str, Any]:
    """Return ASR adapter kwargs without breaking dependency-free mock runs.

    Both ``faster-whisper`` and ``funasr`` honour the same
    ``--asr-device`` / ``--asr-compute-type`` knobs so the user can move every
    GPU-capable backend onto the same CUDA device without code changes.
    """
    model = config.low_overlap_asr_model.lower()
    if model == "mock":
        return {"language": config.language}
    kwargs: dict[str, Any] = {"language": _adapter_language(config.language)}
    if model in {"faster-whisper", "funasr"}:
        kwargs["device"] = config.faster_whisper_device
    if model == "faster-whisper":
        kwargs["model_size"] = config.faster_whisper_model_size
        kwargs["compute_type"] = config.faster_whisper_compute_type
    return kwargs


def _high_overlap_asr_config(config: PipelineConfig) -> dict[str, str]:
    """faster-whisper config for separated/multi-decode candidates.

    The high-overlap path always decodes with faster-whisper, so it honours the
    same ``--faster-whisper-model``/``--asr-device``/``--asr-compute-type`` as
    the low-overlap baseline instead of silently using small/cpu/int8.
    """
    return {
        "model": config.faster_whisper_model_size,
        "device": config.faster_whisper_device,
        "compute_type": config.faster_whisper_compute_type,
    }


def _separation_adapter_kwargs(config: PipelineConfig) -> dict[str, Any]:
    if config.speech_separation_backend.lower() not in {"sepformer", "speechbrain-sepformer"}:
        return {}
    return {
        "model_source": config.sepformer_model_source,
        "device": config.speech_separation_device,
    }
