# Experiment Plan

## Common Setup

Create a manually annotated evaluation split with timestamps, speaker labels, overlap labels, transcript text, topics, decisions, and action items. Report aggregate metrics and separate results for low- and high-overlap regions.

## Experiment 1: Overlap Routing

Compare predicted `low_overlap_cluster` and `high_overlap_candidate` routes with manual overlap labels. Report accuracy, precision, recall, F1, and the downstream cost/quality trade-off at different thresholds.

## Experiment 2: High-overlap Candidate Generation

Compare candidate outputs with forced single-output transcription. Evaluate oracle candidate WER, top-1 WER, speaker-hypothesis coverage, human-rated candidate usefulness, and whether useful information survives ambiguous overlap.

## Experiment 3: Metadata-aware LLM Post-processing

Compare:

1. plain transcript + LLM;
2. transcript + speaker labels + LLM;
3. transcript + speaker labels + overlap/confidence metadata + LLM.

Measure correction quality, speaker-attribution accuracy, uncertainty-preservation quality, decision extraction, and action-item extraction.

## Experiment 4: Episodic Memory QA

Compare:

1. normal summary-based QA;
2. RAG over plain transcript;
3. speaker-aware Episodic Memory QA.

Measure QA accuracy, speaker-specific retrieval accuracy, action-item retrieval, cross-meeting recall, and evidence hit rate.

## Experiment 5: Hallucination and Evidence Evaluation

Evaluate whether answers, decisions, and action items are supported by timestamped evidence. Report evidence precision/recall, unsupported-claim rate, hallucination rate, and calibration of confidence and uncertainty notes.
