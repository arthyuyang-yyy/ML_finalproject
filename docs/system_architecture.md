# System Architecture

## Design Goal

The architecture separates audio processing, uncertainty representation, LLM reasoning, and memory retrieval. Every downstream conclusion should remain traceable to source segments.

## Data Flow (Actual Implementation)

```text
Audio
  -> preprocess_audio (normalize to 16kHz mono float32 WAV)
  -> load_audio
  -> segment_waveform (energy-based VAD with merge & split)
  -> estimate_segment_overlap_scores
       -> pyannote OSD (if HF_TOKEN set)
       -> explicit overlap regions
       -> energy fallback (conservative, max 0.39)
  -> route_segment (threshold 0.4)
       -> low_overlap_cluster
          -> process_low_overlap_segments
             -> WhisperX/Whisper/Paraformer/Mock ASR
             -> pyannote/WhisperX diarization turns or deterministic fallback
       -> high_overlap_candidate
          -> process_high_overlap_segments
             -> faster-whisper multi-decode candidates or explicit fallback candidates
  -> build_metadata_segment (17-field evidence record)
  -> write_segment_clips (export per-segment WAV)
  -> validate_metadata_segment
  -> extract_meeting_events (LLM or fallback)
  -> create_episode_from_segments
  -> store_episode (JSONL)
  -> write_json (per-meeting JSON artifacts)
```

See [pipeline_walkthrough.md](pipeline_walkthrough.md) for the complete 14-step call chain.

## Module Responsibilities

### Core Pipeline

| Module | File | Responsibility |
| --- | --- | --- |
| Preprocessing | `src/audio/preprocess.py` | Load, mono-convert, polyphase-resample, peak-normalize, VAD-segment, and export float32 WAV |
| Clip export | `src/audio/clipper.py` | Write per-evidence-segment WAV clips to disk |
| Overlap detection | `src/overlap_detector.py` | Score overlap: pyannote OSD adapter (priority), explicit region coverage, or energy fallback (max 0.39) |
| Dual-path router | `src/dual_path_router.py` | Route segments by overlap threshold (default 0.4) |
| Low-overlap path | `src/low_overlap.py` | Produce stable text, speaker, timestamps, ASR confidence, and speaker confidence for low-overlap segments |
| ASR | `src/asr.py` | Pluggable adapters (Mock/WhisperX/Whisper/Paraformer) with calibrated confidence; WhisperX is the preferred heavy backend for low-overlap segments |
| Diarization | `src/diarization.py` | pyannote speaker turns when configured, otherwise deterministic speaker-labeling fallback |
| Speech separation | `src/speech_separation.py` | Separation interface (stub — pending model integration) |
| High-overlap path | `src/high_overlap.py` | Preserve mixed-speaker records with empty main transcript and multiple candidates |
| Candidate generator | `src/candidate_generator.py` | Produce multiple transcript/speaker hypotheses with faster-whisper beam/temperature/language variations, with fallback candidates for lightweight runs |
| Metadata builder | `src/metadata_builder.py` | Normalize outputs into the shared 17-field evidence schema |
| Schema validation | `src/schema_validation.py` | Validates evidence-packet records, candidate structure, and per-meeting lists |
| LLM post-process | `src/llm_postprocess.py` | Build constrained metadata-aware prompts; uncertainty-aware correction interface (correction stub) |

### Pipeline Orchestration

| Module | File | Responsibility |
| --- | --- | --- |
| Run pipeline | `src/pipeline/run_pipeline.py` | `run_meeting_pipeline()` — end-to-end orchestration from audio to artifacts |
| Config | `src/pipeline/config.py` | `PipelineConfig` frozen dataclass (paths, SR, threshold, language) |
| I/O | `src/pipeline/io.py` | `ensure_meeting_dirs()`, `write_json()`, `read_json()` |

### LLM Subsystem

| Module | File | Responsibility |
| --- | --- | --- |
| Event extractor | `src/llm/event_extractor.py` | Extract meeting events from evidence segments (LLM or deterministic fallback) |
| Event validator | `src/llm/event_validator.py` | Validate LLM-extracted events, enforce evidence_id citations |
| Gemma client | `src/llm/gemma_client.py` | Pluggable Gemma JSON-generation interface |
| Prompts | `src/llm/prompts.py` | Build evidence-aware event extraction prompts |

### Memory & QA

| Module | File | Responsibility |
| --- | --- | --- |
| Episodic memory | `src/episodic_memory.py` | Create episodes from segments, persist as JSONL, keyword-based search |
| RAG QA | `src/rag_qa.py` | Retrieve relevant episodes and answer with evidence citations |

### Evaluation & Data

| Module | File | Responsibility |
| --- | --- | --- |
| Evaluation | `src/evaluation.py` | WER, CER, overlap-routing metrics, speaker-attribution accuracy, evidence quality (stub) |
| Data synthesis | `src/data_synthesis.py` | Controlled two-speaker overlap mixtures with SNR and ground-truth annotations |
| Utilities | `src/utils.py` | `validate_score()` utility |

### UI

| Module | File | Responsibility |
| --- | --- | --- |
| Gradio app | `src/ui/gradio_app.py` | Interactive pipeline demo via Gradio |

### Package Facades

| Package | Re-exports |
| --- | --- |
| `src/overlap/` | `detect_overlap_segments`, `estimate_segment_overlap_scores`, `detect_pyannote_overlap_regions`, `DEFAULT_OVERLAP_THRESHOLD` |
| `src/evidence/` | `build_metadata_segment`, `validate_metadata_segment`, `validate_meeting`, `validate_candidate` |
| `src/llm/` | `extract_meeting_events`, `validate_meeting_event` |
| `src/memory/` | `create_episode_from_segments`, `store_episode`, `search_episodes` |
| `src/qa/` | `answer_question_with_evidence`, `retrieve_relevant_memory` |
| `src/candidates/` | `generate_high_overlap_candidates` |

## Key Contracts

- Scores use the range `[0.0, 1.0]`.
- Times are seconds from the beginning of the meeting audio.
- High-overlap records retain candidate lists and uncertainty notes.
- High-overlap main records use `speaker="MIXED"` and `text=""`; transcript content is kept in `candidates` rather than forced into one answer.
- Energy fallback overlap scores are capped at 0.39 (below the default routing threshold of 0.4).
- Low-overlap records are single-hypothesis evidence records: stable `text`, `speaker`, timestamps, `asr_confidence`, `speaker_confidence`, empty `candidates`, and empty `uncertainty_note`.
- Decisions and action items must carry timestamped evidence.
- Storage and retrieval backends remain replaceable during early experiments.

## IO Artifact Paths

Per-meeting outputs are written to `outputs/{meeting_id}/`:

| Artifact | Path | Description |
| --- | --- | --- |
| Preprocessed audio | `preprocessed.wav` | 16kHz mono float32 WAV |
| VAD segments | `vad_segments.json` | Timestamped speech regions |
| Overlap scores | `overlap.json` | VAD segments with overlap scores |
| Low-overlap segments | `low_overlap_segments.json` | Evidence records routed to low-overlap path |
| High-overlap candidates | `high_overlap_candidates.json` | Evidence records routed to high-overlap path |
| Evidence segments | `evidence_segments.json` | All validated evidence records |
| Meeting events | `meeting_events.json` | LLM-extracted meeting events |
| Episodic memory | `episodic_memory.json` | Episode records |
| Audio clips | `clips/{evidence_id}.wav` | Per-segment WAV exports |

## Implementation Status

| Phase | Status |
| --- | --- |
| 1. Validate metadata and annotation contracts | Completed |
| 2. Baseline overlap detection, ASR, and diarization | Completed (pyannote adapters + conservative fallbacks, low-overlap ASR/speaker path, mock defaults for tests) |
| 3. Candidate generation and uncertainty-aware prompts | Completed (multi-decode candidate interface, fallback candidates, LLM event extraction) |
| 4. Local episode storage and retrieval | Completed (JSONL persistence, keyword search) |
| 5. Run ablations and evidence-quality evaluation | Pending (requires annotated evaluation split, heavy-model integration) |
