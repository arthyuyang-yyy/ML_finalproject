# Core Innovation Points

## 1. Overlap-aware Routing

The system estimates the overlap degree of every audio segment and chooses a processing path instead of applying the most expensive method everywhere.

**Low-overlap path**

- VAD segmentation
- speaker embedding extraction
- speaker clustering
- ASR transcription

**High-overlap path**

- speech separation or candidate generation
- multiple possible transcript and speaker candidates
- uncertainty preservation
- LLM-assisted reasoning without forcing a single answer

This routing design aims to balance computational cost and recognition quality while making the route itself measurable.

## 2. Uncertainty-aware High-overlap Processing

A high-overlap segment may contain several plausible interpretations. The system can retain candidate transcript A, candidate transcript B, possible speaker assignments, overlap score, confidence score, and an uncertainty note.

Downstream modules must not silently collapse these alternatives. The LLM should explicitly mark uncertain content and avoid presenting unsupported guesses as facts.

## 3. Metadata-aware LLM Post-processing

The LLM receives structured metadata, including speaker label, timestamp, overlap score, ASR confidence, speaker confidence, processing path, candidate interpretations, domain terms, and previous memory context.

It performs transcript correction, speaker-attribution checking, uncertainty preservation, opinion extraction, conflict and consensus detection, decision extraction, action-item extraction, and evidence-based summary generation. Every extracted decision and action item should cite timestamped evidence.

## 4. Episodic Memory

Each meeting episode stores:

- meeting ID and episode ID;
- timestamp range, speakers, and topic;
- original and corrected transcripts;
- overlap and confidence information;
- candidate interpretations;
- decisions, action items, and evidence text;
- an embedding vector when retrieval is implemented.

This supports meeting QA, historical and cross-meeting recall, action-item retrieval, speaker-specific search, and traceable timestamped evidence.

## 5. Evaluation Beyond WER and DER

WER and DER remain useful, but they do not measure whether a meeting assistant is trustworthy. The project also evaluates overlap-routing accuracy, candidate usefulness, speaker-attribution accuracy, uncertainty-preservation quality, action-item extraction accuracy, RAG QA accuracy, evidence hit rate, and hallucination rate.

**Implementation status**: The evidence-support metrics are implemented (`evaluate_evidence_support` in `src/evaluation.py`): evidence precision/recall/F1, evidence hit rate, hallucination rate, correct-abstention rate, and confidence calibration (Brier score); overlap-routing, WER/CER, and speaker-attribution metrics were already in place. Candidate-usefulness and uncertainty-preservation metrics remain to be defined.
