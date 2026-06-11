# TODO

This file tracks implementation progress. The authoritative requirements,
acceptance criteria, schemas, and final deliverables are defined in
[`Project_task.md`](Project_task.md).

## Phase 1: Interfaces and Data

- [x] Define bilingual research positioning and system architecture.
- [x] Define segment metadata and annotation schemas.
- [x] Create module interfaces without loading heavy models.
- [x] Add schema validation and small fixture datasets.

## Phase 2: Audio and Dual-path Baselines

- [x] Step 1 - Audio preprocessing: mono conversion, 16 kHz resampling,
  normalization, and standard WAV export.
- [x] Step 2 - Energy-based VAD with timestamped segments.
- [x] Step 3 - Per-segment audio clip export with `audio_clip_path`.
- [x] Step 4 - Pluggable ASR adapters and confidence normalization:
  Mock, Whisper, WhisperX, and FunASR.
- [x] Step 5 - Optional pyannote diarization adapter.
- [x] Step 6 - Integrate diarization into the end-to-end pipeline and enforce
  the speaker-assignment rules from `Project_task.md`:
  - use the dominant speaker only when coverage is sufficient;
  - use `MIXED` for heavy overlap;
  - use `UNKNOWN` when no speaker can be assigned reliably.
- [x] Step 7a - Baseline overlap detection with pyannote OSD and an explicitly
  labeled conservative energy fallback.
- [ ] Step 7b - Calibrate the overlap threshold against human labels and report
  the threshold sweep, routing metrics, and cost/quality trade-off.
- [x] Step 8 - Configurable dual-path router with default threshold `0.4`.
- [x] Step 9 - Low-overlap baseline producing speaker, transcript, timestamps,
  confidence values, and an empty candidate list.
- [x] Step 10 - High-overlap baseline preserving multiple ASR candidates and
  an uncertainty note instead of forcing one transcript.
- [ ] Step 11 - Speech separation: removed placeholder stubs (`src/speech_separation.py`,
  `src/candidates/separation_optional.py`). Future work will introduce an independent
  adapter/backend interface with defined input/output protocols and acceptance criteria.
  Pending: model selection, integration path, and evaluation metrics.
- [x] Step 12 - Evidence-segment schema builder and validator.
- [ ] Validate that every emitted `audio_clip_path` exists on disk.

## Phase 3: LLM, Memory, and QA

- [x] Step 13 - Connect an Ollama Gemma-compatible backend while retaining the
  deterministic offline fallback.
- [ ] Step 14 - Complete evidence-only, JSON-only prompt constraints and
  uncertainty rules from `Project_task.md`.
- [x] Step 15 - Add LLM JSON parse, repair, regeneration, and evidence-ID
  validation.
- [ ] Step 16 - Implement real meeting-event extraction for decisions, action
  items, deadlines, open questions, disagreements, and uncertainty.
- [x] Step 17a - Basic JSONL episodic-memory storage.
- [x] Step 17b - Create event-grouped episodes with complete evidence text,
  clip paths, confidence, importance, and event metadata.
- [x] Step 18a - Basic keyword retrieval.
- [x] Step 18b - Add semantic/hybrid retrieval with relevance gating and
  meeting, speaker, and time filters.
- [x] Step 19 - Implement evidence-backed QA that cites evidence IDs and
  timestamps, refuses unsupported answers, and surfaces overlap uncertainty.
- [x] Emit unified per-meeting pipeline artifacts under
  `outputs/<meeting_id>/`.

## Gradio Demo

- [x] Page 1 - Audio upload and Run Pipeline workflow.
- [x] Page 2 - Timeline with speaker, route, overlap score, transcript, and
  uncertainty.
- [x] Page 3 - High-overlap candidate drill-down.
- [x] Page 4 - Structured meeting-memory view.
- [x] Page 5 - Evidence-cited QA window with timestamp traceability.

## Phase 4: Evaluation

- [ ] Build the manually annotated evaluation split.
- [ ] Experiment 1 - Sweep overlap-routing thresholds and report accuracy,
  precision, recall, F1, and cost/quality trade-offs.
- [ ] Experiment 2 - Compare multi-candidate high-overlap processing with a
  forced single transcript.
- [ ] Experiment 3 - Run metadata-aware LLM ablations.
- [ ] Experiment 4 - Compare Episodic Memory QA with summary QA and transcript
  RAG.
- [ ] Finalize and run evidence-support, hallucination, uncertainty-preservation,
  and candidate-usefulness metrics.

## Shared Infrastructure

- [x] Add controlled two-speaker overlap synthesis with SNR control and
  ground-truth labels.
- [x] Implement WER, CER, overlap-routing, and speaker-attribution metrics.
- [x] Add pipeline orchestration, configuration, I/O helpers, and package
  facades.
- [x] Add deterministic LLM event-extraction fallback.
