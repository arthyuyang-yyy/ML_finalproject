"""Tests covering every fallback path when external model backends are absent.

These tests guarantee that the lightweight / no-model pipeline produces valid,
schema-compliant output without whisper, pyannote, faster-whisper, Gemma, or
an external embedding model installed.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.asr import MockASRAdapter, get_adapter
from src.candidates.generator import generate_high_overlap_candidates
from src.diarization.core import cluster_speakers
from src.evidence import validate_evidence_segments
from src.llm.event_extractor import extract_meeting_events
from src.low_overlap import LOW_OVERLAP_PATH, process_low_overlap_segments
from src.memory.retriever import _default_embedding_backend
from src.qa.answerer import answer_question
from src.evidence.validator import validate_meeting
from src.fallbacks import (
    HashingEmbeddingBackend,
    energy_overlap_proxy,
    estimate_with_energy_fallback,
    fallback_candidates,
    fallback_event_document,
    fallback_answer,
)


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tone(duration_s: float) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _vad_segment(**overrides) -> dict:
    base = {
        "meeting_id": "meeting_fallback",
        "segment_id": "mfb_000",
        "start_time": 0.0,
        "end_time": 1.0,
        "overlap_score": 0.05,
    }
    base.update(overrides)
    return base


def _low_evidence(**overrides) -> dict:
    base = {
        "meeting_id": "meeting_fallback",
        "segment_id": "mfb_000",
        "evidence_id": "mfb_000",
        "speaker": "SPEAKER_00",
        "start_time": 0.0,
        "end_time": 1.5,
        "text": "hello world",
        "processing_path": "low_overlap_cluster",
        "route_reason": "low overlap (0.05 < 0.400)",
        "overlap_score": 0.05,
        "asr_confidence": 0.85,
        "speaker_confidence": 0.80,
        "audio_clip_path": "",
        "source_audio_path": "",
        "language": "und",
        "candidates": [],
        "uncertainty_note": "",
    }
    base.update(overrides)
    return base


def _episode(**overrides) -> dict:
    base = {
        "episode_id": "ep_001",
        "meeting_id": "meeting_fallback",
        "event_type": "action_item",
        "content": "SPEAKER_00 will test the mock pipeline.",
        "speakers": ["SPEAKER_00"],
        "start_time": 0.0,
        "end_time": 1.5,
        "evidence_ids": ["mfb_000"],
        "evidence_text": "Test the mock pipeline.",
        "overlap_score": 0.05,
        "confidence": "high",
        "importance": 0.9,
        "audio_clip_paths": [],
        "uncertainty_note": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ASR fallback tests
# ---------------------------------------------------------------------------


class MockASRFallbackTests(unittest.TestCase):
    def test_auto_selects_mock_when_nothing_installed(self) -> None:
        with patch("src.fallbacks.asr.importlib.util.find_spec", return_value=None):
            adapter = get_adapter("auto")
        self.assertEqual(adapter.name, "mock")

    def test_auto_selects_funasr_with_shared_language_configuration(self) -> None:
        with patch("src.fallbacks.resolve_asr_backend", return_value="funasr"):
            adapter = get_adapter("auto", language=None)
        self.assertEqual(adapter.name, "funasr")

    def test_mock_adapter_produces_valid_transcript_shape(self) -> None:
        adapter = MockASRAdapter(confidence=0.8, language="en")
        result = adapter.transcribe_array(_tone(1.0), SAMPLE_RATE)
        self.assertEqual(result["model"], "mock")
        self.assertEqual(result["language"], "en")
        self.assertIn("text", result)
        self.assertIn("segments", result)
        self.assertEqual(len(result["segments"]), 1)
        self.assertAlmostEqual(result["asr_confidence"], 0.8)

    def test_empty_audio_gets_placeholder_text(self) -> None:
        result = MockASRAdapter().transcribe_array(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
        self.assertIn("mock transcript", result["text"])
        self.assertEqual(result["segments"][0]["end_time"], 0.0)


# ---------------------------------------------------------------------------
# candidate generation fallback tests
# ---------------------------------------------------------------------------


class CandidateFallbackTests(unittest.TestCase):
    def testfallback_candidates_use_segment_text(self) -> None:
        segment = _vad_segment()
        candidates = fallback_candidates(
            segment,
            decode_configs=[
                {"beam_size": 2, "temperature": 1.0},
                {"beam_size": 5, "temperature": 2.0},
            ],
            speaker_hypotheses=[],
        )
        self.assertGreaterEqual(len(candidates), 1)
        for c in candidates:
            self.assertIn("candidate_id", c)
            self.assertIn("text", c)
            self.assertIn("confidence", c)
            self.assertEqual(c["speaker"], "UNKNOWN")

    def test_empty_text_uses_placeholder(self) -> None:
        segment = _vad_segment(text="")
        candidates = fallback_candidates(segment, [{"beam_size": 2, "temperature": 1.0}], [])
        self.assertIn("[high-overlap transcript", candidates[0]["text"])

    def test_speaker_hypotheses_distributed_across_candidates(self) -> None:
        segment = _vad_segment()
        candidates = fallback_candidates(
            segment,
            [{"beam_size": 2, "temperature": 1.0}, {"beam_size": 5, "temperature": 1.5}],
            speaker_hypotheses=["A", "B"],
        )
        speakers = {c["speaker"] for c in candidates}
        self.assertIn("A", speakers)

    def test_high_overlap_generation_uses_only_fallback_without_samples(self) -> None:
        result = generate_high_overlap_candidates(
            _vad_segment(), samples=None,
        )
        self.assertGreaterEqual(len(result), 1)

    def test_zero_sized_samples_skip_faster_whisper_and_fallback(self) -> None:
        result = generate_high_overlap_candidates(
            _vad_segment(), samples=np.array([], dtype=np.float32),
        )
        self.assertGreaterEqual(len(result), 1)


# ---------------------------------------------------------------------------
# diarization fallback tests
# ---------------------------------------------------------------------------


class DiarizationFallbackTests(unittest.TestCase):
    def test_cluster_speakers_returns_unknown_when_empty(self) -> None:
        result = cluster_speakers([])
        self.assertEqual(result, [])

    def test_cluster_speakers_assigns_unknown_to_all_segments(self) -> None:
        segments = [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 1.0, "end_time": 2.0},
        ]
        result = cluster_speakers(segments)
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertEqual(item["speaker"], "UNKNOWN")
            self.assertIn("speaker_confidence", item)


# ---------------------------------------------------------------------------
# overlap detection fallback tests
# ---------------------------------------------------------------------------


class OverlapFallbackTests(unittest.TestCase):
    def test_energy_proxy_returns_zero_for_empty_clip(self) -> None:
        self.assertEqual(energy_overlap_proxy(np.array([]), SAMPLE_RATE), 0.0)

    def test_energy_proxy_capped_below_forty_percent(self) -> None:
        clip = np.random.randn(SAMPLE_RATE).astype(np.float32)
        score = energy_overlap_proxy(clip, SAMPLE_RATE)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 0.39)

    def test_energy_fallback_adds_detector_tag(self) -> None:
        samples = _tone(2.0)
        segments = [_vad_segment(), _vad_segment(segment_id="mfb_001", start_time=1.0, end_time=2.0)]
        scored = estimate_with_energy_fallback(samples, segments, SAMPLE_RATE)
        self.assertEqual(len(scored), 2)
        for item in scored:
            self.assertEqual(item["overlap_detector"], "energy_fallback")
            self.assertIn("overlap_score", item)

    def test_empty_segments_produce_empty_output(self) -> None:
        result = estimate_with_energy_fallback(_tone(1.0), [], SAMPLE_RATE)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# event extraction fallback tests
# ---------------------------------------------------------------------------


class EventExtractionFallbackTests(unittest.TestCase):
    def test_null_client_goes_straight_to_fallback(self) -> None:
        evidence = [_low_evidence()]
        result = extract_meeting_events(evidence, client=None)
        self.assertEqual(result["meeting_id"], "meeting_fallback")
        self.assertIn("meeting_summary", result)
        self.assertIn("events", result)

    def test_empty_evidence_returns_empty_document(self) -> None:
        result = extract_meeting_events([])
        self.assertEqual(result["meeting_id"], "")
        self.assertEqual(result["events"], [])

    def test_fallback_document_includes_low_overlap_as_speaker_stance(self) -> None:
        evidence = [_low_evidence()]
        result = fallback_event_document(evidence, event_index=1)
        events = result["events"]
        self.assertEqual(events[0]["event_type"], "speaker_stance")
        self.assertIn("hello world", events[0]["content"])

    def test_fallback_document_includes_high_overlap_as_uncertainty(self) -> None:
        high = _low_evidence(
            segment_id="mfb_001",
            evidence_id="mfb_001",
            processing_path="high_overlap_candidate",
            speaker="MIXED",
            text="",
            overlap_score=0.82,
            uncertainty_note="Conflicting speakers",
            candidates=[
                {"candidate_id": "c1", "speaker": "A", "text": "maybe A", "confidence": 0.5, "uncertainty_note": "overlap"},
            ],
        )
        result = fallback_event_document([_low_evidence(), high], event_index=1)
        events = result["events"]
        types = {e["event_type"] for e in events}
        self.assertIn("uncertainty", types)
        self.assertIn("speaker_stance", types)

    def test_fallback_validates_output_against_schema(self) -> None:
        evidence = [_low_evidence()]
        result = fallback_event_document(evidence, event_index=1)
        processed = validate_meeting([_low_evidence()])
        for ev in result["events"]:
            if ev["evidence_ids"]:
                self.assertIn(ev["evidence_ids"][0], {s["evidence_id"] for s in processed})


# ---------------------------------------------------------------------------
# QA fallback tests
# ---------------------------------------------------------------------------


class QAFallbackTests(unittest.TestCase):
    def test_no_retrieved_episodes_returns_insufficient(self) -> None:
        result = answer_question("Who did what?", [])
        self.assertTrue(result["insufficient_evidence"])

    def test_insufficient_answer_has_all_required_fields(self) -> None:
        for field in ("answer", "evidence_ids", "speakers", "confidence", "insufficient_evidence"):
            result = answer_question("Q", [])
            self.assertIn(field, result)

    def test_null_client_falls_back_to_deterministic_answer(self) -> None:
        result = answer_question("Who will test?", [_episode()])
        self.assertIn("mfb_000", result["answer"])
        self.assertEqual(result["speaker"], "SPEAKER_00")

    def test_fallback_answer_cites_evidence_ids(self) -> None:
        episode = _episode(evidence_ids=["ev_003", "ev_004"])
        result = fallback_answer("what?", [episode])
        self.assertIn("ev_003", result["answer"])
        self.assertIn("ev_004", result["answer"])

    def test_fallback_answer_formats_timestamp(self) -> None:
        episode = _episode(start_time=10.5, end_time=25.7)
        result = fallback_answer("what?", [episode])
        self.assertIn("10.500-25.700s", result["answer"])

    def test_low_confidence_marks_uncertainty(self) -> None:
        episode = _episode(confidence="low", uncertainty_note="Speaker is unsure.")
        result = fallback_answer("what?", [episode])
        self.assertEqual(result["confidence"], "low")
        self.assertIn("不确定", result["answer"])

    def test_empty_content_falls_back_to_evidence_text(self) -> None:
        episode = _episode(content="", evidence_text="Evidence-only content.")
        result = fallback_answer("what?", [episode])
        self.assertIn("Evidence-only content", result["answer"])

    def test_insufficient_answer_when_no_content_nor_evidence(self) -> None:
        episode = _episode(content="", evidence_text="")
        result = fallback_answer("what?", [episode])
        self.assertTrue(result["insufficient_evidence"])


# ---------------------------------------------------------------------------
# embedding fallback tests
# ---------------------------------------------------------------------------


class EmbeddingFallbackTests(unittest.TestCase):
    def test_hashing_backend_produces_unit_vectors(self) -> None:
        backend = HashingEmbeddingBackend(dimensions=64)
        vectors = backend.encode(["hello world", "你好世界"])
        self.assertEqual(len(vectors), 2)
        for v in vectors:
            self.assertEqual(len(v), 64)
            norm = sum(x * x for x in v)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_default_backend_falls_back_to_hashing(self) -> None:
        with patch(
            "src.memory.retriever.SentenceTransformerEmbeddingBackend",
            side_effect=ImportError("unavailable"),
        ):
            backend = _default_embedding_backend()
        self.assertIsInstance(backend, HashingEmbeddingBackend)
        vectors = backend.encode(["test"])
        self.assertEqual(len(vectors), 1)
        self.assertGreater(len(vectors[0]), 0)


# ---------------------------------------------------------------------------
# low-overlap path with mock ASR
# ---------------------------------------------------------------------------


class LowOverlapMockTests(unittest.TestCase):
    def test_mock_asr_produces_low_overlap_segments(self) -> None:
        samples = _tone(3.0)
        segments = [
            {
                "meeting_id": "t",
                "segment_id": "t_000",
                "start_time": 0.0,
                "end_time": 1.0,
                "overlap_score": 0.1,
            },
            {
                "meeting_id": "t",
                "segment_id": "t_001",
                "start_time": 1.0,
                "end_time": 3.0,
                "overlap_score": 0.05,
            },
        ]
        adapter = MockASRAdapter(confidence=0.75, language="und")
        processed = process_low_overlap_segments(samples, segments, adapter)
        self.assertEqual(len(processed), 2)
        for item in processed:
            self.assertEqual(item["processing_path"], LOW_OVERLAP_PATH)
            self.assertIn("text", item)
            self.assertIn("asr_confidence", item)
            self.assertEqual(item["candidates"], [])
            self.assertIn("speaker_confidence", item)

    def test_no_segments_returns_empty_list(self) -> None:
        result = process_low_overlap_segments(
            _tone(1.0), [], MockASRAdapter(),
        )
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# end-to-end lightweight pipeline invariants
# ---------------------------------------------------------------------------


class LightweightPipelineInvariants(unittest.TestCase):
    def test_mock_pipeline_output_is_schema_valid(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        from src.pipeline.config import PipelineConfig
        from src.pipeline.io import read_json
        from src.pipeline.run_pipeline import run_meeting_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 16000
            t = np.arange(sr) / sr
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)

            config = PipelineConfig(
                outputs_root=root / "outputs",
                low_overlap_asr_model="mock",
                gemma_backend="none",
                overlap_threshold=0.4,
                language="und",
            )
            result = run_meeting_pipeline(str(input_path), "meeting_fb", config=config)

            evidence = read_json(Path(result["output_dir"]) / "evidence_segments.json")
            errors = validate_evidence_segments(evidence)
            self.assertEqual(errors, [], f"fallback evidence is valid; got {errors}")

            meeting_events = read_json(Path(result["output_dir"]) / "meeting_events.json")
            self.assertIn("events", meeting_events)

            memory_path = Path(result["artifacts"]["long_term_episodic_memory"])
            self.assertTrue(memory_path.exists())

            for item in evidence:
                self.assertIn("evidence_id", item)
                self.assertIn("processing_path", item)


if __name__ == "__main__":
    unittest.main()
