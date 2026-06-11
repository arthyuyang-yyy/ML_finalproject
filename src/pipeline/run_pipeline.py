"""End-to-end orchestration for one meeting audio file."""

from typing import Any

from src.asr import get_adapter
from src.audio.clipper import write_segment_clips
from src.diarization import diarize_with_pyannote
from src.overlap.router import route_segment
from src.evidence.builder import build_evidence_segments
from src.high_overlap import process_high_overlap_segments
from src.llm.event_extractor import extract_meeting_events
from src.llm.gemma_client import GemmaClient, create_gemma_client
from src.low_overlap import process_low_overlap_segments
from src.memory.episodic_store import build_episodes, upsert_episodes
from src.overlap.detector import estimate_segment_overlap_scores
from src.audio.preprocess import preprocess_audio, segment_waveform
from src.schema_validation import validate_metadata_segment

from .config import PipelineConfig
from .io import ensure_meeting_dirs, write_json


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

    preprocessed_samples, sample_rate = preprocess_audio(
        input_audio_path,
        str(paths["preprocessed"]),
        target_sample_rate=cfg.target_sample_rate,
    )
    samples = preprocessed_samples

    vad_segments = segment_waveform(samples, sample_rate, meeting_id=meeting_id)
    write_json(paths["vad_segments"], vad_segments)

    diarization_turns = diarize_with_pyannote(str(paths["preprocessed"])) or []
    write_json(paths["diarization"], diarization_turns)

    scored_segments = estimate_segment_overlap_scores(
        samples,
        vad_segments,
        sample_rate,
        audio_path=str(paths["preprocessed"]),
        diarization_turns=diarization_turns,
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
        diarization_turns=diarization_turns,
    )
    high_overlap_processed = process_high_overlap_segments(
        samples,
        high_overlap_input,
        sample_rate=sample_rate,
        language=cfg.language,
        diarization_turns=diarization_turns,
    )
    evidence_segments = build_evidence_segments(
        low_overlap_processed,
        high_overlap_processed,
        meeting_id=meeting_id,
        source_audio_path=input_audio_path,
        language=cfg.language,
        overlap_threshold=cfg.overlap_threshold,
    )

    evidence_segments = write_segment_clips(samples, sample_rate, evidence_segments, paths["clips"])
    evidence_segments = [validate_metadata_segment(segment) for segment in evidence_segments]
    low_overlap_segments = [
        segment for segment in evidence_segments if segment["processing_path"] == "low_overlap_cluster"
    ]
    high_overlap_segments = [
        segment for segment in evidence_segments if segment["processing_path"] == "high_overlap_candidate"
    ]

    meeting_events = extract_meeting_events(evidence_segments, client=llm_client)
    episodic_memory = build_episodes(meeting_events, evidence_segments)

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
    }


def _adapter_language(language: str) -> str | None:
    """Map project-level unknown language labels to adapter-friendly values."""
    return None if language in {"", "und", "unknown"} else language


def _adapter_kwargs(model: str, language: str) -> dict[str, str | None]:
    """Return ASR adapter kwargs without breaking dependency-free mock runs."""
    if model.lower() == "mock":
        return {"language": language}
    return {"language": _adapter_language(language)}
