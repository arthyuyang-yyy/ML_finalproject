# TODO

Development breakdown follows the low-level framework plan (Project.md, shared by the
infrastructure owner). Steps are numbered to match that plan for easy cross-reference.

## Phase 1: Interfaces and Data

- [x] Define bilingual research positioning and system architecture.
- [x] Define segment metadata and annotation schemas.
- [x] Create module interfaces without loading heavy models.
- [x] Add schema validation and small fixture datasets.

## Phase 2: Baselines (audio layer, below the evidence-segment schema)

- [x] Step 1 — Audio preprocessing (`src/audio/preprocess.py`): mono, 16 kHz resample, normalization.
- [x] Step 2 — VAD segmentation (`src/audio/preprocess.py`): energy-based VAD with timestamped segments. Silero VAD upgrade optional.
- [x] Step 3 — Clip export (`src/audio/clipper.py`): per-segment WAV clips with `audio_clip_path` (requires optional `soundfile`).
- [x] Step 4 — ASR baseline (`src/asr.py`): pluggable adapters with calibrated confidence; mixed zh/en must not crash.
- [ ] Step 5 — Diarization backend (`src/diarization.py`): pyannote adapter returning speaker-labeled time regions. *(interface only; covered by PR #13)*
- [ ] Step 6 — Speaker assignment: map diarization labels onto VAD segments (>=70% coverage -> speaker; heavy overlap -> `MIXED`; no match -> `UNKNOWN`; emit `speaker_confidence` in [0, 1]). *(covered by PR #13)*
- [ ] Step 7 — Overlap detector (`src/overlap_detector.py`) and threshold calibration.
  - Current state: lightweight duration/energy placeholder is merged; the real multi-cue score is pending *(covered by PR #13 via pyannote)*.
  - Combine cues: diarization overlap, ASR decode instability, speaker-change rate, energy complexity.
  - Output a continuous overlap score in `[0.0, 1.0]` per segment (enforced by schema validation).
  - Return timestamped overlap regions in seconds from the start of the meeting audio.
  - Calibrate the threshold against human overlap labels: sweep thresholds and report accuracy, precision, recall, F1, and the cost/quality trade-off curve (see Experiment 1).
- [ ] Step 8 — Router (`src/dual_path_router.py`): route to `low_overlap_cluster` / `high_overlap_candidate` via a configurable threshold (default 0.4, do not hard-code). *(stub; covered by PR #13)*
- [ ] Step 9 — Low-overlap evidence builder: full schema record (speaker, text, timestamps, confidences, empty `candidates`). *(partial in `src/metadata_builder.py`)*
- [ ] Step 10 — High-overlap candidate generation (`src/candidate_generator.py`): multiple ASR decodes (beam size / temperature / language); 1–3 candidates with confidence; never merge into a single forced transcript; `uncertainty_note` required (schema-enforced). *(covered by PR #13)*
- [ ] Step 11 — (Bonus, not on the critical path) Speech separation for high-overlap segments (`src/speech_separation.py`): SepFormer/Demucs, separated tracks fed back as candidates.
- [x] Step 12 — Evidence segment validator (`src/schema_validation.py`): required fields, score ranges, path-specific rules.
  - [ ] Follow-up: also check that `audio_clip_path` exists on disk.

## Phase 3: Memory and QA (above the evidence-segment schema)

- [ ] Step 13 — Gemma client (`src/llm/gemma_client.py`): wire a real quantized Gemma backend (Ollama / llama.cpp); keep the interface model-agnostic with CPU fallback. *(deterministic placeholder merged)*
- [ ] Step 14 — Prompt templates (`src/llm/prompts.py`): evidence-only, JSON-only output, mandatory `evidence_ids`, overlap_score > 0.6 capped at low/medium confidence, uncertain owners marked `uncertain`, no small talk. *(basic template merged)*
- [ ] Step 15 — LLM JSON repair (`src/llm/json_repair.py`): parse/repair/regenerate invalid LLM output; reject events whose `evidence_ids` do not exist.
- [ ] Step 16 — Meeting event extraction (`src/llm/event_extractor.py`): decisions, action items, deadlines, open questions, disagreements, overlap-caused uncertainty. *(rule-based version merged; needs real LLM)*
- [ ] Step 17 — Episodic memory store (`src/episodic_memory.py`): convert meeting events into traceable episodes (evidence_ids, evidence_text, clip paths, confidence, importance). *(basic version merged; event-grouped episodes in PR #11)*
- [ ] Step 18 — Memory retriever: v1 keyword (merged), v2 embedding, v3 hybrid scoring (embedding + keyword + importance + recency − overlap penalty) with meeting/speaker/time filters. *(semantic + filtered retrieval in PR #11)*
- [ ] Step 19 — QA answerer (`src/rag_qa.py`): answer only from retrieved episodes; every claim cites `evidence_id` + timestamp; say "cannot determine" when evidence is insufficient; surface uncertainty for high-overlap evidence. *(keyword baseline merged; LLM answering pending)*
- [ ] Unified pipeline artifacts under `outputs/<meeting_id>/`: `preprocessed.wav`, `vad_segments.json`, `low_overlap_segments.json`, `high_overlap_candidates.json`, `evidence_segments.json`, `meeting_events.json`, `episodic_memory.json`, `clips/`. *(partial in `src/pipeline/run_pipeline.py`)*

## Gradio Demo (`app.py`, `src/ui/gradio_app.py`)

- [ ] Page 1 — Audio upload + Run Pipeline button. *(skeleton merged)*
- [ ] Page 2 — Timeline table: time range, speaker, processing path, overlap score, text, uncertainty note.
- [ ] Page 3 — High-overlap segment drill-down: candidates + uncertainty note (+ clip playback).
- [ ] Page 4 — Meeting memory table: decisions / action items / open questions / uncertainties with evidence and confidence.
- [ ] Page 5 — QA window: evidence-cited answers with timestamps.

## Phase 4: Evaluation

- [ ] Build the manually annotated evaluation split (20–50 segments with overlap labels is enough for Experiment 1).
- [ ] Experiment 1 — Overlap routing: accuracy / precision / recall / F1 across thresholds (0.3 / 0.4 / 0.5 sweep table).
- [ ] Experiment 2 — Multi-candidate vs forced single transcript: human-rated candidate usefulness, uncertainty correctness, speaker safety (1–5 scales).
- [ ] Experiment 3 — Metadata-aware LLM ablation: plain transcript vs +speaker vs full metadata; measure action-item/decision accuracy, evidence citation rate, uncertainty preservation, hallucination rate.
- [ ] Experiment 4 — Episodic Memory QA vs summary QA vs transcript RAG: correctness, evidence hit rate, timestamp citation rate, hallucination rate.

## Shared Infrastructure

Cross-cutting tooling that supports multiple phases rather than a single one.

- [x] Add a synthetic overlapping-speech generator with ground-truth labels (controlled overlap duration and SNR).
- [x] Implement objective evaluation metrics (WER/CER, overlap-routing classification, speaker-attribution accuracy). Evidence/hallucination/uncertainty metrics deferred until the traceability design is finalized.
