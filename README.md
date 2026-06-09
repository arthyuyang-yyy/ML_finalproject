# Overlap-aware Dual-path ASR with Episodic Memory for Multi-speaker Meeting Understanding

## Project Title

**Overlap-aware Dual-path ASR with Episodic Memory for Multi-speaker Meeting Understanding**

Chinese title: 面向多人会议理解的重叠感知双路径语音处理与情景记忆系统

## Motivation

Meeting assistants often compress imperfect transcripts into fluent summaries. This can hide speaker-attribution errors, overlapping speech, and unsupported conclusions. This project is not only a meeting summarization system. It is a **verifiable meeting memory system** that retains uncertainty and links later answers, decisions, and action items back to timestamped evidence.

## Difference from the Reference Thesis

The reference thesis already combines ASR, speaker diarization, low-overlap clustering, high-overlap speech separation, LLM correction, and structured meeting summaries. This project extends that foundation instead of reproducing it.

**Reference system**

`ASR -> speaker diarization -> LLM correction -> structured summary`

**Our system**

`ASR + overlap-aware routing -> uncertainty-aware candidate generation -> metadata-aware LLM post-processing -> Episodic Memory -> traceable QA and meeting recall`

The key change is that high-overlap speech is not forced into one confident transcript. Candidate interpretations and confidence metadata remain available to downstream reasoning and retrieval.

## Core Innovations

1. **Overlap-aware routing:** route low-overlap audio to lightweight speaker clustering and high-overlap audio to separation or candidate generation.
2. **Uncertainty-aware candidate generation:** preserve multiple plausible transcripts and speaker hypotheses for ambiguous regions.
3. **Metadata-aware LLM post-processing:** reason over timestamps, confidence, overlap, candidates, and prior memory rather than plain text alone.
4. **Episodic Memory:** store meaningful meeting events with evidence for traceable QA, action-item retrieval, and cross-meeting recall.
5. **Evaluation beyond WER and DER:** measure routing, candidate usefulness, uncertainty preservation, evidence quality, and hallucination.

## System Pipeline

1. Preprocess audio and create timestamped segments.
2. Estimate overlap scores.
3. Route each segment:
   - Low overlap: VAD, speaker embedding, clustering, and ASR.
   - High overlap: speech separation or multiple candidate interpretations.
4. Build a common metadata record for every segment.
5. Use an LLM to correct text, preserve uncertainty, and extract evidence-backed meeting events.
6. Convert related segments into Episodic Memory records.
7. Retrieve episodes to answer questions with speakers, timestamps, confidence, and uncertainty notes.

See [docs/system_architecture.md](docs/system_architecture.md) for the module-level design.

## Repository Structure

```text
.
├── docs/                  # Bilingual research design and experiment plans
├── data/                  # Raw/processed audio and annotation templates
├── outputs/               # Generated artifacts, ignored except placeholders
├── src/                   # Modular pipeline interfaces
├── app.py                 # Future interactive application entry point
├── main.py                # Pipeline entry point
├── README.md
└── README.zh-CN.md
```

## Metadata Schema

Each processed segment uses a shared schema:

| Field | Meaning |
| --- | --- |
| `meeting_id`, `segment_id` | Stable meeting and segment identifiers |
| `speaker` | Speaker label or uncertain speaker hypothesis |
| `start_time`, `end_time` | Evidence timestamp range in seconds |
| `text` | Current transcript |
| `processing_path` | `low_overlap_cluster` or `high_overlap_candidate` |
| `overlap_score` | Estimated overlap likelihood |
| `asr_confidence` | ASR confidence estimate |
| `speaker_confidence` | Speaker-attribution confidence |
| `candidates` | Alternative transcript/speaker interpretations |
| `uncertainty_note` | Human-readable reason for uncertainty |

## Episodic Memory Design

An episode represents a meaningful meeting event or coherent segment group. It stores meeting and episode IDs, timestamp range, speakers, topic, original and corrected transcripts, overlap and confidence information, candidates, decisions, action items, evidence text, and a future embedding vector.

Episodes support:

- evidence-backed meeting QA;
- historical and cross-meeting recall;
- action-item and decision retrieval;
- speaker-specific search;
- traceability from an answer to exact timestamps.

## Planned Experiments

1. Compare predicted overlap routes with manual labels.
2. Compare high-overlap candidate generation with forced single-output transcription.
3. Compare plain-text, speaker-aware, and full-metadata LLM post-processing.
4. Compare summary QA, transcript RAG, and speaker-aware Episodic Memory QA.
5. Measure hallucination rate and timestamped evidence hit rate.

Full details are in [docs/experiment_plan.md](docs/experiment_plan.md).

## Current Status

The repository currently contains the first-stage research design, annotation schema, and clean module interfaces. Heavy models such as Whisper, pyannote, and speech separation models are intentionally not loaded yet.

## How to Run

The current code is an interface-only scaffold:

```bash
python main.py
python app.py
```

Implement the TODOs in `src/` before running real audio experiments. Keep large audio files, model weights, and generated outputs outside Git.
