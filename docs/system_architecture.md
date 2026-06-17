# System Architecture

This document is the compact architecture reference for the current MVP. The
main project contract lives in `README.md` and `Project_task.md`; this file only
records the stable module boundaries and tradeoffs. Older API references and
historical proposals were removed because they no longer matched the runnable
pipeline.

## Pipeline

```text
input audio
-> preprocess audio
-> VAD segmentation
-> diarization
-> overlap scoring
-> route by overlap_score
   -> low_overlap_cluster: ASR + speaker attribution
   -> high_overlap_candidate: candidate generation + resolver
-> evidence_segments.json
-> meeting_events.json
-> episodic_memory.json
-> evidence-backed QA
```

## Segment Paths

Low-overlap segments are handled by `src/low_overlap.py` and ASR adapters. They
must produce a final `speaker`, `text`, ASR confidence, speaker confidence, and
an empty `candidates` list.

High-overlap segments are handled by `src/high_overlap.py`,
`src/candidates/generator.py`, optional `src/speech_separation.py` adapters, and
`src/llm/resolver.py`. Candidate generation keeps alternate transcripts and
decode settings. The resolver then selects a final text by calling a configured
Gemma/Ollama client, or falls back to the highest-confidence candidate. If no
candidate exists, the segment is marked `unresolved` instead of inventing text.

Speech separation is optional. The default backend is `none`; `mock`, `nmf`, and
`sepformer` can be enabled to add separated-source candidates before resolver
selection.

## Evidence Contract

`src/evidence/` is the boundary between audio processing and downstream memory.
Every segment must expose timestamps, speaker, text, route metadata, confidence
scores, audio paths, candidates, and uncertainty notes. Resolved high-overlap
segments may additionally expose:

- `source`: `llm_resolved`, `fallback_resolved`, or `unresolved`
- `decision_reason`: short explanation of the resolver decision

High-overlap evidence keeps the original `candidates` list after resolution so
QA, review, and future evaluation can inspect alternatives.

## Memory and Retrieval

`src/memory/episodic_store.py` converts meeting events into episodes.
`src/memory/retriever.py` ranks episodes with:

```text
0.70 * embedding_similarity + 0.30 * keyword_score
```

The embedding backend is the custom BLAKE2 character n-gram hash embedding in
`src/fallbacks/embeddings.py`. This is an MVP tradeoff: it is deterministic,
lightweight, and easy to test, but it does not provide transformer-level semantic
matching, recency weighting, or event-importance priors.

## Optional Backends

Large models are loaded only when selected at runtime. Tests and demos can run
with mock/fallback adapters. Model weights, raw audio, generated outputs, and
local environment files should remain outside git.

To add a backend, implement the existing adapter interface for that area
(`ASRAdapter`, diarization adapter, speech-separation adapter, or `LLMBackend`),
keep imports lazy, and add focused tests that can run without downloading model
weights.
