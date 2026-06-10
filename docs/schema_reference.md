# Schema Reference

This document compares the three schema structures used across the project, their fields, and their purposes.

---

## 1. Evidence Segment Schema (17 fields)

Defined by `src/metadata_builder.py` — `build_metadata_segment()` and validated by `src/schema_validation.py`. This is the **canonical in-memory schema** used throughout the pipeline.

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

- `start_time <= end_time`
- `processing_path` must be `"low_overlap_cluster"` or `"high_overlap_candidate"`
- All scores (`overlap_score`, `asr_confidence`, `speaker_confidence`, candidate `confidence`) must be in [0, 1]
- High-overlap segments (`processing_path == "high_overlap_candidate"`) must have at least one candidate

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

Defined by `src/episodic_memory.py` — `create_episode_from_segments()`.

| Field | Type | Description |
|-------|------|-------------|
| `meeting_id` | `str` | Meeting identifier |
| `episode_id` | `str` | Unique episode identifier |
| `start_time` | `float` | Earliest segment start (s) |
| `end_time` | `float` | Latest segment end (s) |
| `speakers` | `list[str]` | Unique speakers in this episode |
| `topic` | `str` | Topic label (default `"meeting discussion"`) |
| `summary` | `str` | Concatenated transcript text |
| `evidence_ids` | `list[str]` | Evidence IDs cited |
| `evidence` | `list[dict]` | Original evidence segments |
| `confidence` | `float` | Aggregated confidence [0, 1] |
| `uncertainty_note` | `str` | Aggregated uncertainty notes |

---

## 5. Meeting Event

Defined by `src/llm/event_validator.py` — `validate_meeting_event()`.

| Field | Type | Description |
|-------|------|-------------|
| `meeting_id` | `str` | Meeting identifier |
| `event_id` | `str` | Unique event identifier |
| `summary` | `str` | Event summary text |
| `evidence_ids` | `list[str]` | Evidence ID citations (must exist in evidence segments) |
| `confidence` | `float` | Event confidence [0, 1] |
| `uncertainty_note` | `str` | Human-readable uncertainty note |

---

## 6. QA Answer

Defined by `src/rag_qa.py` — `answer_question_with_evidence()`.

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Answer text |
| `evidence` | `list[dict]` | Supporting evidence segments |
| `speaker` | `str` | Cited speakers |
| `timestamp` | `str` | Time range string |
| `confidence` | `float` | Answer confidence [0, 1] |
| `uncertainty_note` | `str` | Uncertainty explanation |
| `query` | `str` | Original query (echoed) |

---

## 7. ASR Transcript

Defined by `src/asr.py` — `ASRAdapter.transcribe_array()`.

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
    └── input to ── Metadata Builder
                         └── produces ── Evidence Segment (17 fields)
                                              ├── input to ── Event Extractor → Meeting Event (6 fields)
                                              └── input to ── Episode Creator → Episodic Memory (11 fields)
                                                                                      └── input to ── QA → QA Answer (7 fields)
```
