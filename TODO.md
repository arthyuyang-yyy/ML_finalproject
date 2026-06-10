# TODO

## Phase 1: Interfaces and Data

- [x] Define bilingual research positioning and system architecture.
- [x] Define segment metadata and annotation schemas.
- [x] Create module interfaces without loading heavy models.
- [x] Add schema validation and small fixture datasets.

## Phase 2: Baselines

- [x] Implement audio preprocessing and VAD segmentation.
- [x] Add baseline overlap detector and threshold calibration (energy fallback + pyannote adapter).
- [x] Add ASR and speaker-diarization adapters (mock baselines; WhisperX/Whisper/FunASR and pyannote optional).
- [x] Implement high-overlap candidate generation baseline.
- [x] Implement end-to-end pipeline orchestration (`src/pipeline/run_pipeline.py`).
- [x] Add audio clip export (`src/audio/clipper.py`).
- [x] Add Gradio interactive demo (`src/ui/gradio_app.py`).

## Phase 3: Memory and QA

- [x] Implement persistent episode storage (JSONL).
- [x] Add embedding-based and metadata-filtered retrieval (keyword baseline; vector pending).
- [x] Connect an LLM provider with uncertainty-preserving prompts (deterministic fallback; real LLM pending).
- [x] Build evidence-backed QA and action-item retrieval (baseline implemented).

- [ ] Add vector/semantic retrieval for episodic memory.
- [ ] Connect real LLM backend (Gemma, Ollama, or hosted API).

## Phase 4: Evaluation

- [ ] Build the manually annotated evaluation split.
- [ ] Run routing and candidate-generation experiments.
- [ ] Run metadata-aware LLM ablations.
- [ ] Evaluate Episodic Memory QA, evidence quality, and hallucination.

## Shared Infrastructure

Cross-cutting tooling that supports multiple phases rather than a single one.

- [x] Add a synthetic overlapping-speech generator with ground-truth labels (controlled overlap duration and SNR).
- [x] Implement objective evaluation metrics (WER/CER, overlap-routing classification, speaker-attribution accuracy).
- [x] Implement evidence-support metrics (evidence precision/recall/F1, hit rate, hallucination rate, correct-abstention rate, confidence calibration) now that the traceability design is finalized. Uncertainty-preservation and candidate-usefulness metrics remain deferred.
- [x] Add pipeline orchestration, config system, and I/O helpers.
- [x] Add LLM event extraction with deterministic fallback.
- [x] Add package facades for clean imports (`src/overlap/`, `src/evidence/`, `src/llm/`, `src/memory/`, `src/qa/`, `src/candidates/`).
