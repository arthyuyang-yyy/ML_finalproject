# System Architecture

## Design Goal

The architecture separates audio processing, uncertainty representation, LLM reasoning, and memory retrieval. Every downstream conclusion should remain traceable to source segments.

## Data Flow

```text
Audio
  -> preprocessing
  -> overlap detection
  -> dual-path routing
       -> low overlap: diarization/clustering + ASR
       -> high overlap: separation and/or candidate generation
  -> shared metadata segments
  -> uncertainty-aware LLM post-processing
  -> Episodic Memory
  -> retrieval and evidence-backed QA
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `preprocessing.py` | Validate and segment input audio |
| `overlap_detector.py` | Score overlap and return overlap regions |
| `dual_path_router.py` | Choose the low- or high-overlap path |
| `asr.py` | Pluggable ASR adapters (mock/Whisper/Paraformer) with calibrated confidence |
| `diarization.py` | Provide speaker-labeling interface |
| `speech_separation.py` | Provide high-overlap separation interface |
| `candidate_generator.py` | Represent multiple high-overlap interpretations |
| `metadata_builder.py` | Normalize outputs into a shared schema |
| `llm_postprocess.py` | Build constrained prompts and evidence-backed outputs |
| `episodic_memory.py` | Create, store, and search meeting episodes |
| `rag_qa.py` | Retrieve episodes and answer with evidence |
| `evaluation.py` | Define metrics beyond WER and DER |

## Key Contracts

- Scores use the range `[0.0, 1.0]`.
- Times are seconds from the beginning of the meeting audio.
- High-overlap records retain candidate lists and uncertainty notes.
- Decisions and action items must carry timestamped evidence.
- Storage and retrieval backends remain replaceable during early experiments.

## Planned Implementation Phases

1. Validate metadata and annotation contracts with small hand-written examples.
2. Add baseline overlap detection, ASR, and diarization adapters.
3. Implement candidate generation and uncertainty-aware prompts.
4. Add local episode storage and retrieval.
5. Run ablations and evidence-quality evaluation.
