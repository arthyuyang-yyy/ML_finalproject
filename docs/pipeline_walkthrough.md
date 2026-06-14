# Pipeline Walkthrough

This document traces the exact 14-step call chain executed by `run_meeting_pipeline()` in `src/pipeline/run_pipeline.py`. Each step identifies the responsible module, the function called, and the artifact produced.

## Input

- `input_audio_path`: path to a raw meeting audio file (any format supported by `soundfile`)
- `meeting_id`: stable identifier used for output directory naming

## Pre-flight

```python
cfg = PipelineConfig(
    outputs_root=Path("outputs"),
    target_sample_rate=16000,
    overlap_threshold=0.4,
    language="und",
)
paths = ensure_meeting_dirs(cfg.meeting_dir(meeting_id))
# Creates: outputs/{meeting_id}/
#          outputs/{meeting_id}/clips/
```

## Step-by-step

### Step 1: Preprocess Audio
**Module:** `src/audio/preprocess.py` — `preprocess_audio()`

Reads raw audio, averages channels to mono, resamples to 16 kHz (polyphase if SciPy is available, linear interpolation otherwise), peak-normalizes to 0.97, and writes a float32 WAV file.

**Output:** `outputs/{meeting_id}/preprocessed.wav`

### Step 2: Reuse Preprocessed Samples

`preprocess_audio()` returns the normalized mono float32 samples together with the sample rate. The pipeline reuses this array directly and does not perform a redundant disk read.

### Step 3: VAD Segmentation
**Module:** `src/audio/preprocess.py` — `segment_waveform()`

Runs energy-threshold VAD to detect speech regions:
- Frames: 25 ms with 10 ms hop
- Threshold: 0.3 × peak RMS
- Post-processing: bridge short silences (< 300 ms), merge adjacent short regions (gap < 1 s), drop regions < 1 s, pad 50 ms boundaries, split regions > 20 s into ~12 s chunks

**Output:** list of dicts with `meeting_id`, `segment_id`, `start_time`, `end_time`
**Artifact:** `vad_segments.json`

### Step 4: Overlap Scoring
**Module:** `src/overlap/detector.py` — `estimate_segment_overlap_scores()`

Attaches an `overlap_score` [0, 1] to each segment. Three strategies, tried in order:

1. **pyannote OSD** (if `HF_TOKEN` is set): loads the `pyannote/overlapped-speech-detection` model, detects overlapped-speech regions, and computes per-segment overlap coverage ratio. Once configured, loading or inference failures are surfaced instead of silently changing detectors.
2. **Explicit regions** (if `overlap_regions` parameter is provided): computes coverage of provided regions.
3. **Energy fallback**: computes a weak proxy from per-frame RMS high-energy ratio and dynamic range, capped at 0.39 to prevent false routing to the high-overlap path.

Each segment also receives an `overlap_detector` field identifying the strategy used.

**Artifact:** `overlap.json`

### Step 5: Routing
**Module:** `src/overlap/router.py` — `route_segment()`

Routes each segment to one of two paths based on `overlap_score >= threshold` (default threshold: 0.4):
- `"low_overlap_cluster"` — handled by ASR + speaker attribution
- `"high_overlap_candidate"` — preserves multiple interpretation candidates

Each segment receives a `route_reason` string explaining the decision.

### Step 6: Low-Overlap ASR + Speaker Attribution
**Module:** `src/low_overlap.py` — `process_low_overlap_segments()`

For low-overlap segments, the pipeline produces a single stable evidence record:
- `text` via the configured ASR adapter (`WhisperX` is recommended for heavy runs; `MockASRAdapter` remains the default for tests/demo wiring)
- `speaker` and `speaker_confidence` via pyannote/WhisperX-style diarization turns when available, or deterministic fallback labels otherwise
- original `start_time` / `end_time`, `overlap_score`, and empty `candidates`

**Artifact:** `low_overlap_segments.json`

### Step 7: High-Overlap Candidate Generation
**Module:** `src/high_overlap.py` — `process_high_overlap_segments()`

For high-overlap segments, the main evidence record intentionally avoids a forced transcript:
- `speaker` is set to `"MIXED"`
- `text` is kept empty
- `speaker_confidence` is low
- multiple transcript/speaker hypotheses are stored in `candidates`

Candidate generation uses `src/candidates/generator.py` and prefers faster-whisper multi-decode settings:
- `beam_size`: 1 / 5
- `temperature`: 0.0 / 0.4 / 0.8
- `language`: auto / zh / en

If faster-whisper is unavailable, explicit fallback candidates are emitted so downstream uncertainty handling remains testable.

### Step 8: Evidence Segment Construction
**Module:** `src/evidence/builder.py` — `build_evidence_segments()`

Merges low-overlap and high-overlap results into one timestamp-sorted list. It verifies that each record is supplied through the correct route, normalizes simplified high-overlap candidates, rejects duplicate IDs, and builds an evidence record (17 required + 1 optional field) containing timing, routing, confidence, candidate, and provenance data.

Fields: `meeting_id`, `segment_id`, `evidence_id`, `speaker`, `start_time`, `end_time`, `text`, `processing_path`, `route_reason`, `overlap_score`, `asr_confidence`, `speaker_confidence`, `audio_clip_path`, `source_audio_path`, `language`, `candidates`, `uncertainty_note`.

### Step 9: Clip Export
**Module:** `src/audio/clipper.py` — `write_segment_clips()`

Writes each segment's audio slice as a float32 WAV file to `outputs/{meeting_id}/clips/{evidence_id}.wav`. The `audio_clip_path` field is updated accordingly.

### Step 10: Schema Validation
**Module:** `src/evidence/validator.py` — `validate_metadata_segment()`

Validates every evidence record against the canonical schema: required fields, types, score ranges [0, 1], valid processing paths, time ordering, and candidate structure (high-overlap segments must have candidates).

### Step 11: Event Extraction
**Module:** `src/llm/event_extractor.py` — `extract_meeting_events()`

Extracts a structured `{meeting_id, meeting_summary, events}` document. A configured `GemmaClient` must return JSON-only output using the supported event types. Output is parsed/repaired, validated against real evidence IDs, checked for action-item fields and unsupported speakers, and retried once when invalid. Remaining invalid events are discarded; if no valid LLM result remains, a deterministic evidence-only fallback is used. High-overlap evidence cannot support high-confidence events.

**Artifact:** `meeting_events.json`

### Step 12: Episodic Memory
**Module:** `src/memory/episodic_store.py` — `build_episodes()`

Converts each extracted meeting event into a traceable episode containing speakers, timestamps, evidence citations, evidence text, confidence, importance, and audio clip paths. Any event citing high-overlap evidence is independently forced to an uncertainty episode with `MIXED` speakers and low confidence.

**Artifacts:**
- `outputs/{meeting_id}/episodic_memory.json` — per-meeting episodes
- `memory/episodic_memory.json` — long-term memory, atomically upserted by meeting ID

### Step 13: Write Artifacts
**Module:** `src/pipeline/io.py` — `write_json()`

Persists all intermediate and final results as JSON files under `outputs/{meeting_id}/`:

| File | Content |
|------|---------|
| `vad_segments.json` | Raw VAD segments |
| `overlap.json` | Overlap-scored segments |
| `low_overlap_segments.json` | Low-overlap evidence records |
| `high_overlap_candidates.json` | High-overlap evidence records with candidates |
| `evidence_segments.json` | All validated evidence records |
| `meeting_events.json` | Extracted meeting events |
| `episodic_memory.json` | Episode records |

The long-term memory path is returned as `artifacts.long_term_episodic_memory`. Re-running the same meeting replaces its previous episodes instead of appending duplicates.

## Output Summary

`run_meeting_pipeline()` returns a dictionary:

```python
{
    "meeting_id": str,
    "output_dir": str,
    "artifacts": {name: path, ...},
    "num_vad_segments": int,
    "num_evidence_segments": int,
    "num_low_overlap_segments": int,
    "num_high_overlap_segments": int,
    "meeting_events": dict,
    "episodic_memory": list[dict],
    "long_term_memory_size": int,
    "preprocessed_num_samples": int,
}
```

## Running the Pipeline

```bash
# CLI
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001

# Python
from src.pipeline import run_meeting_pipeline
result = run_meeting_pipeline("data/raw_audio/meeting_001.wav", "meeting_001")

# Gradio UI
python app.py
```

The Gradio page is organized into five areas:

1. Upload audio and run the shared pipeline.
2. Inspect the canonical evidence timeline.
3. Select high-overlap segments and inspect candidate interpretations.
4. Review structured episodic memory with evidence citations.
5. Ask questions using hybrid retrieval over the current meeting's episodes only.

The UI stores canonical evidence and episodes in `gr.State`; it does not duplicate pipeline, retrieval, or QA logic. QA also exposes the retrieved episodes and validated structured answer for traceability.
