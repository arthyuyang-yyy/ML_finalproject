"""End-to-end orchestration for one meeting audio file."""

from pathlib import Path
from typing import Any

from src.asr import get_adapter
from src.audio.clipper import write_segment_clips
from src.dual_path_router import route_segment
from src.episodic_memory import create_episode_from_segments
from src.high_overlap import process_high_overlap_segments
from src.llm.event_extractor import extract_meeting_events
from src.low_overlap import process_low_overlap_segments
from src.metadata_builder import build_metadata_segment
from src.overlap_detector import estimate_segment_overlap_scores
from src.audio.preprocess import load_audio, preprocess_audio, segment_waveform
from src.schema_validation import validate_metadata_segment

from .config import PipelineConfig
from .io import ensure_meeting_dirs, write_json


def run_meeting_pipeline(
    input_audio_path: str,
    meeting_id: str,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Run the lightweight meeting pipeline and write per-meeting artifacts."""
    cfg = config or PipelineConfig()
    paths = ensure_meeting_dirs(cfg.meeting_dir(meeting_id))

    preprocessed_samples, sample_rate = preprocess_audio(
        input_audio_path,
        str(paths["preprocessed"]),
        target_sample_rate=cfg.target_sample_rate,
    )
    samples, sample_rate = load_audio(str(paths["preprocessed"]), target_sample_rate=cfg.target_sample_rate)

    vad_segments = segment_waveform(samples, sample_rate, meeting_id=meeting_id)
    write_json(paths["vad_segments"], vad_segments)

    scored_segments = estimate_segment_overlap_scores(
        samples,
        vad_segments,
        sample_rate,
        audio_path=str(paths["preprocessed"]),
    )
    write_json(paths["overlap"], scored_segments)

    routed_segments = [
        {
            **segment,
            "processing_path": route_segment(float(segment["overlap_score"]), threshold=cfg.overlap_threshold),
        }
        for segment in scored_segments
    ]
    low_overlap_input = [
        segment for segment in routed_segments if segment["processing_path"] == "low_overlap_cluster"
    ]
    high_overlap_input = [
        segment for segment in routed_segments if segment["processing_path"] == "high_overlap_candidate"
    ]

    low_overlap_processed = process_low_overlap_segments(
        samples,
        low_overlap_input,
        asr_adapter=get_adapter(cfg.low_overlap_asr_model, **_adapter_kwargs(cfg.low_overlap_asr_model, cfg.language)),
        sample_rate=sample_rate,
    )
    high_overlap_processed = process_high_overlap_segments(
        samples,
        high_overlap_input,
        sample_rate=sample_rate,
        language=cfg.language,
    )
    transcribed_segments = low_overlap_processed + high_overlap_processed

    evidence_segments: list[dict[str, Any]] = []
    low_overlap_segments: list[dict[str, Any]] = []
    high_overlap_segments: list[dict[str, Any]] = []
    for segment in transcribed_segments:
        overlap_score = float(segment["overlap_score"])
        processing_path = str(segment.get("processing_path") or route_segment(overlap_score, threshold=cfg.overlap_threshold))
        route_reason = _route_reason(overlap_score, cfg.overlap_threshold, processing_path)
        evidence_id = str(segment["segment_id"])
        candidates = list(segment.get("candidates", []))
        uncertainty_note = str(segment.get("uncertainty_note", ""))
        evidence = build_metadata_segment(
            meeting_id=meeting_id,
            segment_id=str(segment["segment_id"]),
            evidence_id=evidence_id,
            speaker=str(segment["speaker"]),
            start_time=float(segment["start_time"]),
            end_time=float(segment["end_time"]),
            text=str(segment["text"]),
            processing_path=processing_path,
            route_reason=route_reason,
            overlap_score=overlap_score,
            asr_confidence=float(segment["asr_confidence"]),
            speaker_confidence=float(segment["speaker_confidence"]),
            candidates=candidates,
            uncertainty_note=uncertainty_note,
            source_audio_path=str(Path(input_audio_path)),
            language=cfg.language,
        )
        evidence_segments.append(evidence)
        if processing_path == "high_overlap_candidate":
            high_overlap_segments.append(evidence)
        else:
            low_overlap_segments.append(evidence)

    evidence_segments = write_segment_clips(samples, sample_rate, evidence_segments, paths["clips"])
    evidence_segments = [validate_metadata_segment(segment) for segment in evidence_segments]
    low_overlap_segments = [
        segment for segment in evidence_segments if segment["processing_path"] == "low_overlap_cluster"
    ]
    high_overlap_segments = [
        segment for segment in evidence_segments if segment["processing_path"] == "high_overlap_candidate"
    ]

    meeting_events = extract_meeting_events(evidence_segments)
    episodic_memory = [create_episode_from_segments(evidence_segments)] if evidence_segments else []

    write_json(paths["low_overlap_segments"], low_overlap_segments)
    write_json(paths["high_overlap_candidates"], high_overlap_segments)
    write_json(paths["evidence_segments"], evidence_segments)
    write_json(paths["meeting_events"], meeting_events)
    write_json(paths["episodic_memory"], episodic_memory)

    return {
        "meeting_id": meeting_id,
        "output_dir": str(paths["base"]),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "base"},
        "num_vad_segments": len(vad_segments),
        "num_evidence_segments": len(evidence_segments),
        "num_low_overlap_segments": len(low_overlap_segments),
        "num_high_overlap_segments": len(high_overlap_segments),
        "meeting_events": meeting_events,
        "episodic_memory": episodic_memory,
        "preprocessed_num_samples": int(preprocessed_samples.size),
    }


def _route_reason(overlap_score: float, threshold: float, processing_path: str) -> str:
    comparator = ">=" if overlap_score >= threshold else "<"
    return f"overlap_score={overlap_score:.3f} {comparator} threshold={threshold:.3f}; routed to {processing_path}"


def _adapter_language(language: str) -> str | None:
    """Map project-level unknown language labels to adapter-friendly values."""
    return None if language in {"", "und", "unknown"} else language


def _adapter_kwargs(model: str, language: str) -> dict[str, str | None]:
    """Return ASR adapter kwargs without breaking dependency-free mock runs."""
    if model.lower() == "mock":
        return {"language": language}
    return {"language": _adapter_language(language)}
