# TODO

## Phase 1: Interfaces and Data

- [x] Define bilingual research positioning and system architecture.
- [x] Define segment metadata and annotation schemas.
- [x] Create module interfaces without loading heavy models.
- [x] Add schema validation and small fixture datasets.

## Phase 2: Baselines

- [x] Implement audio preprocessing and VAD segmentation.
- [ ] Add baseline overlap detector and threshold calibration.
- [ ] Add ASR and speaker-diarization adapters.
- [ ] Implement high-overlap candidate generation baseline.

## Phase 3: Memory and QA

- [ ] Implement persistent episode storage.
- [ ] Add embedding-based and metadata-filtered retrieval.
- [ ] Connect an LLM provider with uncertainty-preserving prompts.
- [ ] Build evidence-backed QA and action-item retrieval.

## Phase 4: Evaluation

- [ ] Build the manually annotated evaluation split.
- [ ] Run routing and candidate-generation experiments.
- [ ] Run metadata-aware LLM ablations.
- [ ] Evaluate Episodic Memory QA, evidence quality, and hallucination.

## Shared Infrastructure

Cross-cutting tooling that supports multiple phases rather than a single one.

- [x] Add a synthetic overlapping-speech generator with ground-truth labels (controlled overlap duration and SNR).
- [x] Implement objective evaluation metrics (WER/CER, overlap-routing classification, speaker-attribution accuracy). Evidence/hallucination/uncertainty metrics deferred until the traceability design is finalized.
