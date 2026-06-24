"""Tests for high-overlap candidate generation."""

import unittest

import numpy as np

from src.candidates.generator import generate_high_overlap_candidates
from src.high_overlap import HIGH_OVERLAP_PATH, HIGH_OVERLAP_SPEAKER, _slice_segment, process_high_overlap_segments
from src.llm.gemma_client import GemmaClient
from src.llm.resolver import resolve_high_overlap_segment


class CandidateGeneratorTests(unittest.TestCase):
    def test_fallback_candidates_preserve_decode_configs(self) -> None:
        candidates = generate_high_overlap_candidates(
            {"segment_id": "m1_seg_009", "text": ""},
            samples=np.ones(16000, dtype=np.float32),
            sample_rate=16000,
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["candidate_id"], "m1_seg_009_c1")
        self.assertEqual(candidates[0]["speaker"], "UNKNOWN")
        self.assertIn("decode_config", candidates[0])
        self.assertIn("High-overlap segment", candidates[0]["uncertainty_note"])

    def test_resolver_accepts_generated_candidate_shape(self) -> None:
        segment = {
            "segment_id": "m1_seg_009",
            "speaker": "MIXED",
            "text": "",
            "candidates": generate_high_overlap_candidates(
                {"segment_id": "m1_seg_009", "text": ""},
                samples=np.ones(16000, dtype=np.float32),
                sample_rate=16000,
            ),
        }

        resolved = resolve_high_overlap_segment(segment)

        self.assertEqual(resolved["speaker"], "MIXED")
        self.assertTrue(resolved["text"].strip())
        self.assertEqual(resolved["source"], "fallback_resolved")
        self.assertIn("decision_reason", resolved)


class HighOverlapPathTests(unittest.TestCase):
    def _high_overlap_segment(self) -> dict:
        return {
            "meeting_id": "meeting_001",
            "segment_id": "m1_seg_009",
            "speaker": "MIXED",
            "start_time": 0.0,
            "end_time": 6.0,
            "text": "",
            "overlap_score": 0.78,
            "processing_path": HIGH_OVERLAP_PATH,
            "asr_confidence": 0.54,
            "speaker_confidence": 0.35,
            "candidates": [
                {
                    "candidate_id": "m1_seg_009_c1",
                    "speaker": "SPEAKER_01",
                    "text": "We should test WhisperX first.",
                    "confidence": 0.62,
                    "uncertainty_note": "High-overlap segment.",
                },
                {
                    "candidate_id": "m1_seg_009_c2",
                    "speaker": "SPEAKER_02",
                    "text": "Let's test the baseline first.",
                    "confidence": 0.51,
                    "uncertainty_note": "High-overlap segment.",
                },
            ],
            "uncertainty_note": "High-overlap segment; speaker attribution is uncertain.",
        }

    def test_high_overlap_record_keeps_main_transcript_empty(self) -> None:
        samples = np.ones(16000 * 6, dtype=np.float32) * 0.1
        segments = [
            {
                "meeting_id": "meeting_001",
                "segment_id": "m1_seg_009",
                "start_time": 0.0,
                "end_time": 6.0,
                "overlap_score": 0.78,
                "processing_path": HIGH_OVERLAP_PATH,
            }
        ]

        processed = process_high_overlap_segments(samples, segments, sample_rate=16000)

        self.assertEqual(len(processed), 1)
        segment = processed[0]
        self.assertEqual(segment["speaker"], HIGH_OVERLAP_SPEAKER)
        self.assertEqual(segment["text"], "")
        self.assertEqual(segment["processing_path"], HIGH_OVERLAP_PATH)
        self.assertEqual(segment["speaker_confidence"], 0.35)
        self.assertGreaterEqual(len(segment["candidates"]), 2)
        self.assertIn("speaker attribution is uncertain", segment["uncertainty_note"])

    def test_slice_segment_uses_decode_context_window(self) -> None:
        samples = np.arange(16000 * 5, dtype=np.float32)
        segment = {
            "start_time": 2.0,
            "end_time": 2.1,
            "decode_start_time": 1.0,
            "decode_end_time": 3.0,
        }

        clip = _slice_segment(samples, segment, sample_rate=16000)

        self.assertEqual(clip.size, 16000 * 2)
        self.assertEqual(float(clip[0]), 16000.0)

    def test_resolver_fills_high_overlap_transcript_from_llm(self) -> None:
        segment = self._high_overlap_segment()
        client = GemmaClient(lambda _prompt: {
            "speaker": "SPEAKER_01",
            "text": "We should test WhisperX first.",
            "confidence": 0.72,
            "resolution_mode": "selected",
            "source_candidate_ids": ["m1_seg_009_c1"],
            "decision_reason": "Candidate 1 best matches the local context.",
        })

        resolved = resolve_high_overlap_segment(segment, client=client)

        self.assertEqual(resolved["speaker"], "SPEAKER_01")
        self.assertEqual(resolved["text"], "We should test WhisperX first.")
        self.assertEqual(resolved["source"], "llm_resolved")
        self.assertEqual(resolved["resolution_mode"], "selected")
        self.assertEqual(resolved["source_candidate_ids"], ["m1_seg_009_c1"])
        self.assertIn("Candidate 1", resolved["decision_reason"])
        self.assertEqual(len(resolved["candidates"]), 2)

    def test_resolver_prompt_includes_context_and_accepts_merged_output(self) -> None:
        segment = self._high_overlap_segment()
        captured: dict[str, str] = {}

        def generate(prompt: str) -> dict:
            captured["prompt"] = prompt
            return {
                "speaker": "SPEAKER_01",
                "text": "We should test the WhisperX baseline first.",
                "confidence": 0.68,
                "resolution_mode": "merged",
                "source_candidate_ids": ["m1_seg_009_c1", "m1_seg_009_c2"],
                "decision_reason": "Merged the stable action from candidate 1 with the baseline wording from candidate 2.",
            }

        context = [
            {
                "speaker": "SPEAKER_00",
                "start_time": -4.0,
                "end_time": -1.0,
                "text": "We need one baseline before the alignment test.",
            }
        ]
        resolved = resolve_high_overlap_segment(segment, client=GemmaClient(generate), context_segments=context)

        self.assertIn("Adjacent reliable context", captured["prompt"])
        self.assertIn("one baseline", captured["prompt"])
        self.assertEqual(resolved["text"], "We should test the WhisperX baseline first.")
        self.assertEqual(resolved["resolution_mode"], "merged")
        self.assertEqual(resolved["source_candidate_ids"], ["m1_seg_009_c1", "m1_seg_009_c2"])

    def test_resolver_allows_llm_to_mark_unresolved(self) -> None:
        segment = self._high_overlap_segment()
        client = GemmaClient(lambda _prompt: {
            "speaker": "MIXED",
            "text": "",
            "confidence": 0.0,
            "resolution_mode": "unresolved",
            "source_candidate_ids": [],
            "decision_reason": "Candidates conflict and context does not support either wording.",
        })

        resolved = resolve_high_overlap_segment(segment, client=client)

        self.assertEqual(resolved["speaker"], "MIXED")
        self.assertEqual(resolved["text"], "")
        self.assertEqual(resolved["source"], "unresolved")
        self.assertEqual(resolved["resolution_mode"], "unresolved")

    def test_resolver_falls_back_when_llm_output_is_invalid(self) -> None:
        segment = self._high_overlap_segment()
        client = GemmaClient(lambda _prompt: {
            "speaker": "SPEAKER_02",
            "text": "This invalid confidence should not be accepted.",
            "confidence": 1.2,
            "decision_reason": "Out of range confidence.",
        })

        resolved = resolve_high_overlap_segment(segment, client=client)

        self.assertEqual(resolved["speaker"], "SPEAKER_01")
        self.assertEqual(resolved["text"], "We should test WhisperX first.")
        self.assertEqual(resolved["asr_confidence"], 0.62)
        self.assertEqual(resolved["source"], "fallback_resolved")
        self.assertEqual(resolved["resolution_mode"], "selected")
        self.assertEqual(resolved["source_candidate_ids"], ["m1_seg_009_c1"])
        self.assertIn("highest-confidence candidate", resolved["decision_reason"])

    def test_resolver_preserves_segment_speaker_when_best_candidate_is_unknown(self) -> None:
        segment = self._high_overlap_segment()
        segment["speaker"] = "MIXED"
        segment["candidates"] = [{
            "candidate_id": "m1_seg_009_c1",
            "speaker": "UNKNOWN",
            "text": "The speaker is unclear.",
            "confidence": 0.82,
            "uncertainty_note": "High-overlap segment.",
        }]

        resolved = resolve_high_overlap_segment(segment)

        self.assertEqual(resolved["speaker"], "MIXED")
        self.assertEqual(resolved["text"], "The speaker is unclear.")
        self.assertEqual(resolved["source"], "fallback_resolved")

    def test_resolver_tie_breaks_equal_confidence_by_candidate_order(self) -> None:
        segment = self._high_overlap_segment()
        segment["candidates"] = [
            {
                "candidate_id": "m1_seg_009_c1",
                "speaker": "SPEAKER_01",
                "text": "First equal-confidence candidate.",
                "confidence": 0.7,
                "uncertainty_note": "High-overlap segment.",
            },
            {
                "candidate_id": "m1_seg_009_c2",
                "speaker": "SPEAKER_02",
                "text": "Second equal-confidence candidate.",
                "confidence": 0.7,
                "uncertainty_note": "High-overlap segment.",
            },
        ]

        resolved = resolve_high_overlap_segment(segment)

        self.assertEqual(resolved["speaker"], "SPEAKER_01")
        self.assertEqual(resolved["text"], "First equal-confidence candidate.")
        self.assertEqual(resolved["source"], "fallback_resolved")

    def test_suspected_overlap_preserves_baseline_when_candidate_diverges(self) -> None:
        segment = self._high_overlap_segment()
        segment.update({
            "route_mode": "suspected_high_overlap",
            "baseline_text": "We should test WhisperX first.",
            "baseline_speaker": "SPEAKER_00",
            "baseline_asr_confidence": 0.74,
            "baseline_speaker_confidence": 0.9,
        })
        segment["candidates"] = [{
            "candidate_id": "m1_seg_009_c1",
            "speaker": "SPEAKER_01",
            "text": "Completely different unsupported wording.",
            "confidence": 0.95,
            "uncertainty_note": "High-overlap segment.",
        }]

        resolved = resolve_high_overlap_segment(segment)

        self.assertEqual(resolved["speaker"], "SPEAKER_00")
        self.assertEqual(resolved["text"], "We should test WhisperX first.")
        self.assertEqual(resolved["source"], "baseline_preserved")
        self.assertEqual(resolved["resolution_mode"], "baseline_preserved")
        self.assertEqual(resolved["asr_confidence"], 0.74)

    def test_suspected_overlap_replaces_baseline_when_candidate_is_low_risk(self) -> None:
        segment = self._high_overlap_segment()
        segment.update({
            "route_mode": "suspected_high_overlap",
            "baseline_text": "We should test WhisperX first.",
            "baseline_speaker": "SPEAKER_00",
            "baseline_asr_confidence": 0.6,
            "baseline_speaker_confidence": 0.9,
        })
        segment["candidates"] = [{
            "candidate_id": "m1_seg_009_c1",
            "speaker": "SPEAKER_01",
            "text": "We should test WhisperX first",
            "confidence": 0.82,
            "uncertainty_note": "High-overlap segment.",
        }]

        resolved = resolve_high_overlap_segment(segment)

        self.assertEqual(resolved["speaker"], "SPEAKER_01")
        self.assertEqual(resolved["text"], "We should test WhisperX first")
        self.assertEqual(resolved["source"], "fallback_resolved")
        self.assertEqual(resolved["resolution_mode"], "selected")

    def test_resolver_marks_segment_unresolved_without_candidates(self) -> None:
        segment = self._high_overlap_segment()
        segment["candidates"] = []

        resolved = resolve_high_overlap_segment(segment)

        self.assertEqual(resolved["speaker"], "MIXED")
        self.assertEqual(resolved["text"], "")
        self.assertEqual(resolved["source"], "unresolved")
        self.assertIn("No usable transcript candidates", resolved["decision_reason"])

    def test_resolver_does_not_promote_placeholder_candidate_text(self) -> None:
        segment = self._high_overlap_segment()
        segment["candidates"] = [{
            "candidate_id": "m1_seg_009_c1",
            "speaker": "UNKNOWN",
            "text": "[high-overlap transcript candidate pending ASR decode]",
            "confidence": 0.0,
            "uncertainty_note": "High-overlap segment.",
        }]

        resolved = resolve_high_overlap_segment(segment)

        self.assertEqual(resolved["text"], "")
        self.assertEqual(resolved["source"], "unresolved")
        self.assertIn("No usable transcript candidates", resolved["decision_reason"])


if __name__ == "__main__":
    unittest.main()
