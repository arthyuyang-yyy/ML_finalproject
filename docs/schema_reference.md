# Schema Reference

This document compares the three schema structures used across the project, their fields, and their purposes.

---

## 1. Evidence Segment Schema (17 fields)

Defined by `src/evidence/builder.py` and `src/evidence/schema.py`, then validated by `src/evidence/validator.py`. This is the **canonical in-memory schema** used throughout the pipeline.

The 13 fields in the project task document are the domain-facing core. The implementation deliberately retains four provenance fields required for reliable downstream traceability:

- `evidence_id`
- `route_reason`
- `source_audio_path`
- `language`

`build_evidence_segments()` accepts the low-overlap and high-overlap result lists, normalizes simplified candidate objects, sorts all records by timestamp, rejects duplicate IDs, and emits this complete 17-field representation. `build_evidence_file()` provides the equivalent JSON-file-to-JSON-file workflow.

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `meeting_id` | `str` | Stable meeting identifier |
| 2 | `segment_id` | `str` | Stable segment identifier |
| 3 | `evidence_id` | `str` | Unique evidence record ID (usually mirrors `segment_id`) |
| 4 | `speaker` | `str` | Speaker label or uncertain speaker hypothesis |
| 5 | `start_time` | `float` | Evidence start time in seconds |
| 6 | `end_time` | `float` | Evidence end time in seconds |
| 7 | `text` | `str` | Current transcript |
| 8 | `processing_path` | `str` | `"low_overlap_cluster"` or `"high_overlap_candidate"` |
| 9 | `route_reason` | `str` | Human-readable routing decision explanation |
| 10 | `overlap_score` | `float` | Estimated overlap likelihood [0, 1] |
| 11 | `asr_confidence` | `float` | ASR confidence estimate [0, 1] |
| 12 | `speaker_confidence` | `float` | Speaker-attribution confidence [0, 1] |
| 13 | `audio_clip_path` | `str` | Path to exported audio clip file |
| 14 | `source_audio_path` | `str` | Original input audio path |
| 15 | `language` | `str` | Language code (default `"und"`) |
| 16 | `candidates` | `list[dict]` | Alternative transcript/speaker interpretations |
| 17 | `uncertainty_note` | `str` | Human-readable reason for uncertainty |

### Candidate sub-schema

Each item in `candidates` list:

| Field | Type | Description |
|-------|------|-------------|
| `candidate_id` | `str` | Unique candidate identifier |
| `speaker` | `str` | Speaker hypothesis |
| `text` | `str` | Transcript hypothesis |
| `confidence` | `float` | Candidate confidence [0, 1] |
| `uncertainty_note` | `str` | Reason for uncertainty |
| `decode_config` | `dict` | Optional ASR decode settings used to produce the candidate |

### Validation rules

- `start_time < end_time`
- `processing_path` must be `"low_overlap_cluster"` or `"high_overlap_candidate"`
- All scores (`overlap_score`, `asr_confidence`, `speaker_confidence`, candidate `confidence`) must be in [0, 1]
- IDs must be non-empty and unique within a meeting
- All records in one evidence file must share one `meeting_id`
- Low-overlap segments must contain transcript text and must not contain candidates or uncertainty notes
- High-overlap segments must use `speaker="MIXED"`, keep primary `text` empty, include at least one candidate, and explain uncertainty
- Simplified candidates containing only `speaker`, `text`, and `confidence` are accepted by the builder; stable `candidate_id` and `uncertainty_note` values are generated before validation

---

## 2. Annotation CSV Template

Defined in `data/annotations/annotation_template.csv`. This is the **human annotation schema** for building the evaluation split.

| # | Column | Description |
|---|--------|-------------|
| 1 | `meeting_id` | Meeting identifier |
| 2 | `segment_id` | Segment identifier |
| 3 | `start_time` | Segment start time (s) |
| 4 | `end_time` | Segment end time (s) |
| 5 | `speaker` | Speaker label |
| 6 | `text` | Reference transcript |
| 7 | `is_overlap` | Whether the segment contains overlapping speech |
| 8 | `overlap_type` | `"none"`, `"partial"`, or `"full"` |
| 9 | `topic` | Topic or discussion theme |
| 10 | `decision` | Meeting decision made in this segment |
| 11 | `action_item` | Action item assigned in this segment |

### Mapping to Evidence Schema

| Annotation CSV | Evidence Segment |
|----------------|------------------|
| `meeting_id` | `meeting_id` |
| `segment_id` | `segment_id` (becomes `evidence_id`) |
| `start_time` | `start_time` |
| `end_time` | `end_time` |
| `speaker` | `speaker` |
| `text` | `text` (reference/gold) |
| `is_overlap` + `overlap_type` | → used to derive `overlap_score` (ground truth) |
| `topic` | → stored in episodic memory |
| `decision` | → stored in episodic memory |
| `action_item` | → stored in episodic memory |

The annotation CSV has **extra fields** (`topic`, `decision`, `action_item`) not present in the evidence segment schema. These are used during evaluation to assess decision/action-item extraction quality.

The evidence segment schema has **extra fields** not in the CSV: `evidence_id`, `processing_path`, `route_reason`, `asr_confidence`, `speaker_confidence`, `audio_clip_path`, `source_audio_path`, `language`, `candidates`, `uncertainty_note`. These are pipeline-produced fields.

---

## 3. Data Synthesis Annotation

Defined by `src/data_synthesis.py` — `build_annotation()`. This is the **output schema** of the synthetic data generator, used for controlled-experiment evaluation.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `meeting_id` | `str` | Synthetic meeting identifier |
| `sample_rate` | `int` | Audio sample rate |
| `speakers` | `list[str]` | Speaker labels present |
| `duration_s` | `float` | Total audio duration in seconds |
| `overlap_regions` | `list[list[float]]` | `[[start, end], ...]` overlap intervals |
| `overlap_duration` | `float` | Total overlap seconds |
| `overlap_ratio` | `float` | Overlap duration / total duration |
| `segments` | `list[dict]` | Per-speaker segments |

### Per-segment fields

| Field | Type | Description |
|-------|------|-------------|
| `meeting_id` | `str` | Meeting identifier |
| `segment_id` | `str` | Segment identifier |
| `speaker` | `str` | Speaker label |
| `start_time` | `float` | Segment start (s) |
| `end_time` | `float` | Segment end (s) |
| `text` | `str` | Reference transcript |
| `is_overlap` | `bool` | Whether this segment overlaps with another speaker |
| `overlap_type` | `str` | `"none"`, `"partial"`, or `"full"` |

### Mapping to Evidence Schema

`to_annotation_rows()` in `data_synthesis.py` flattens the synthesis annotation to CSV-style rows matching the annotation template columns.

---

## 4. Episodic Memory Record

Defined and validated by `src/memory/memory_schema.py`, built from structured meeting events by `src/memory/episodic_store.py`, and stored in `memory/episodic_memory.json`.

| Field | Type | Description |
|-------|------|-------------|
| `meeting_id` | `str` | Meeting identifier |
| `episode_id` | `str` | Unique episode identifier |
| `event_type` | `str` | Structured event type; high-overlap evidence is forced to `uncertainty` |
| `topic` | `str` | Concise event topic |
| `content` | `str` | Event content from the validated meeting event |
| `start_time` | `float` | Earliest segment start (s) |
| `end_time` | `float` | Latest segment end (s) |
| `speakers` | `list[str]` | Unique speakers in this episode |
| `evidence_ids` | `list[str]` | Evidence IDs cited |
| `evidence_text` | `str` | Supporting transcript or candidate interpretations |
| `overlap_score` | `float` | Maximum overlap score among cited evidence |
| `confidence` | `str` | `high`, `medium`, or `low` |
| `importance` | `float` | Retrieval importance [0, 1], defaulted by event type |
| `audio_clip_paths` | `list[str]` | Source audio clips for traceability |
| `uncertainty_note` | `str` | Aggregated uncertainty notes; required for uncertainty episodes |
| `memory_timestamp` | `str` | UTC ISO-8601 write time used for recency scoring; optional for legacy records |

High-overlap evidence is never persisted as a confirmed decision or action. The Memory layer independently converts it to `event_type="uncertainty"`, `speakers=["MIXED"]`, and `confidence="low"` even if an upstream caller bypasses the LLM validator.

Long-term persistence uses atomic JSON replacement. Reprocessing a meeting replaces that meeting's previous episodes while preserving episodes from other meetings.

Retrieval indexes `content`, `topic`, `event_type`, `speakers`, and `evidence_text`. It combines normalized BM25, embedding similarity, importance, recency, and an overlap penalty. High-overlap records are penalized only when they are incorrectly represented as a certain event rather than `uncertainty`.

---

## 5. Structured Meeting Events Document

Defined by `src/llm/event_validator.py` and produced by `src/llm/event_extractor.py`.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `meeting_id` | `str` | Meeting identifier; must match the evidence records |
| `meeting_summary` | `str` | Evidence-grounded meeting summary |
| `events` | `list[dict]` | Validated structured events |

### Event fields

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `str` | Unique event identifier |
| `event_type` | `str` | `decision`, `action_item`, `open_question`, `disagreement`, `uncertainty`, `speaker_stance`, or `topic_transition` |
| `content` | `str` | Evidence-backed event content |
| `speakers` | `list[str]` | Speaker labels supported by cited evidence |
| `evidence_ids` | `list[str]` | Evidence ID citations (must exist in evidence segments) |
| `confidence` | `str` | `high`, `medium`, or `low` |
| `uncertainty_note` | `str` | Optional explanation added when confidence is reduced |

Action items additionally require `task` and `owner`; `deadline` is optional. `owner` must be supported by cited evidence or equal to `"uncertain"`.

Validation rejects missing/unknown evidence IDs, duplicate event IDs, unsupported speakers, invalid action items, and invalid event types. Events citing high-overlap evidence cannot remain high confidence; uncertainty events are always low confidence.

---

## 6. QA Answer

Defined by `src/qa/answerer.py` and validated by `src/qa/answer_validator.py`.

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Answer text |
| `episode_ids` | `list[str]` | Retrieved episodes actually used by the answer |
| `evidence_ids` | `list[str]` | Evidence IDs cited in the answer text |
| `citations` | `list[dict]` | Episode ID, evidence IDs, and exact start/end time for each citation |
| `speakers` | `list[str]` | Speaker labels supported by cited episodes |
| `confidence` | `str` | `high`, `medium`, or `low` |
| `uncertainty_note` | `str` | Uncertainty explanation |
| `insufficient_evidence` | `bool` | True when retrieved evidence cannot answer the question |
| `question` | `str` | Original question |

Every supported answer must include its evidence IDs and exact episode time ranges in the rendered answer text. Unknown citations, altered timestamps, unsupported speakers, or high-overlap evidence presented without a low-confidence uncertainty warning are rejected. Empty retrieval returns an explicit cannot-determine response.

---

## 7. ASR Transcript

Defined by `src/asr/core.py` — `ASRAdapter.transcribe_array()`.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Full transcript text |
| `language` | `str` | Detected language |
| `model` | `str` | Model name used |
| `asr_confidence` | `float` | Aggregate confidence [0, 1] |
| `segments` | `list[dict]` | Per-segment `{"start", "end", "text", "confidence"}` |

---

## Schema Relationship Diagram

```text
Annotation CSV (11 cols)
    └── flattened from ── Data Synthesis Annotation
                              └── generated by ── data_synthesis.py

ASR Transcript (5 fields)
    └── input to ── Evidence Segment Builder
                         └── produces ── Evidence Segment (17 fields)
                                              ├── input to ── Event Extractor → Meeting Event (6 fields)
                                              └── input to ── Episode Creator → Episodic Memory (11 fields)
                                                                                      └── input to ── QA → validated QA Answer
```
