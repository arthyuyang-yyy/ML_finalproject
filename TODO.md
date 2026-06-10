# TODO

## Phase 1: Interfaces and Data

- [x] Define bilingual research positioning and system architecture.
- [x] Define segment metadata and annotation schemas.
- [x] Create module interfaces without loading heavy models.
- [x] Add schema validation and small fixture datasets.

## Phase 2: Baselines

- [x] Implement audio preprocessing and VAD segmentation.
- [ ] Add baseline overlap detector and threshold calibration.
  - Output a continuous overlap score in `[0.0, 1.0]` per segment (enforced by schema validation).
  - Return timestamped overlap regions in seconds from the start of the meeting audio.
  - Route segments to `low_overlap_cluster` or `high_overlap_candidate` via a configurable threshold (do not hard-code it).
  - Calibrate the threshold against human overlap labels: sweep thresholds and report accuracy, precision, recall, F1, and the cost/quality trade-off curve (see Experiment 1 in `docs/experiment_plan.md`).
  - Segments routed to `high_overlap_candidate` must carry a candidate list and an uncertainty note downstream (schema-enforced).
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
