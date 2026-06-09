# Project Proposal

## Research Question

Can overlap-aware routing, explicit uncertainty representation, and Episodic Memory make multi-speaker meeting understanding more accurate, efficient, and verifiable than a conventional ASR-to-summary pipeline?

## Problem

Overlapping speech creates correlated ASR and speaker-attribution errors. Conventional pipelines often pass one deterministic transcript to an LLM, which can turn uncertain input into confident but unsupported summaries. They also discard the segment-level evidence needed for later meeting QA.

## Proposed Approach

The project routes low- and high-overlap segments through different processing paths. High-overlap segments may produce multiple transcript and speaker candidates. A metadata-aware LLM preserves uncertainty and extracts evidence-backed decisions and action items. These outputs are stored as Episodic Memory records for traceable QA and cross-meeting recall.

## Expected Contributions

- A modular dual-path meeting-audio pipeline.
- A shared uncertainty-aware segment metadata schema.
- An Episodic Memory representation for verifiable meeting understanding.
- An evaluation framework covering routing, candidates, uncertainty, retrieval, evidence, and hallucination.

## Scope

The first stage defines research design and interfaces without loading heavy models. Later stages will implement replaceable baselines, create a manually annotated test set, and run controlled ablation experiments.
