"""Tests for the pluggable ASR adapters and confidence mapping.

These exercise the dependency-free mock recogniser, the confidence calibration,
the VAD-segment wiring, and the raw-result normalizers, so they run without
``whisper``/``funasr`` or any model download.

Run with::

    python -m unittest tests.test_asr
"""

import unittest

import numpy as np

from src.asr import (
    MockASRAdapter,
    get_adapter,
    logprob_to_confidence,
    transcribe_segments,
    _aggregate_confidence,
    _from_funasr_result,
    _from_whisper_result,
    _from_whisperx_result,
)

SAMPLE_RATE = 16000


def _tone(duration_s: float) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


class ConfidenceTests(unittest.TestCase):
    def test_high_logprob_maps_near_one(self) -> None:
        self.assertAlmostEqual(logprob_to_confidence(0.0), 1.0, places=5)

    def test_low_logprob_maps_lower(self) -> None:
        self.assertLess(logprob_to_confidence(-1.0), logprob_to_confidence(-0.1))

    def test_no_speech_prob_discounts_confidence(self) -> None:
        self.assertAlmostEqual(logprob_to_confidence(0.0, no_speech_prob=0.5), 0.5, places=5)

    def test_result_stays_in_unit_range(self) -> None:
        for lp in (-5.0, -1.0, 0.0):
            for ns in (0.0, 0.5, 1.0):
                value = logprob_to_confidence(lp, ns)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class MockAdapterTests(unittest.TestCase):
    def test_transcribe_array_shape(self) -> None:
        result = MockASRAdapter(confidence=0.8).transcribe_array(_tone(1.0), SAMPLE_RATE)
        self.assertEqual(set(result), {"text", "language", "model", "asr_confidence", "segments"})
        self.assertEqual(result["model"], "mock")
        self.assertEqual(result["asr_confidence"], 0.8)
        self.assertEqual(len(result["segments"]), 1)

    def test_rejects_out_of_range_confidence(self) -> None:
        with self.assertRaises(ValueError):
            MockASRAdapter(confidence=1.5)

    def test_negative_confidence_raises(self) -> None:
        with self.assertRaises(ValueError):
            MockASRAdapter(confidence=-0.1)

    def test_transcribe_array_with_duration(self) -> None:
        adapter = MockASRAdapter(confidence=0.8, language="en")
        self.assertEqual(adapter.language, "en")
        self.assertEqual(adapter.confidence, 0.8)
        result = adapter.transcribe_array(_tone(2.5), SAMPLE_RATE)
        self.assertEqual(result["language"], "en")
        self.assertGreaterEqual(result["segments"][0]["end_time"], 2.0)


class FactoryTests(unittest.TestCase):
    def test_builds_known_adapters(self) -> None:
        self.assertIsInstance(get_adapter("mock"), MockASRAdapter)
        self.assertEqual(get_adapter("whisperx").name, "whisperx")
        # paraformer is an alias for the FunASR adapter (not instantiated/loaded here)
        self.assertEqual(get_adapter("paraformer").name, "funasr")

    def test_rejects_unknown_adapter(self) -> None:
        with self.assertRaises(ValueError):
            get_adapter("does-not-exist")


class TranscribeSegmentsTests(unittest.TestCase):
    def test_enriches_vad_segments_in_place(self) -> None:
        samples = _tone(3.0)
        segments = [
            {"meeting_id": "m", "segment_id": "m-0000", "start_time": 0.0, "end_time": 1.0},
            {"meeting_id": "m", "segment_id": "m-0001", "start_time": 2.0, "end_time": 3.0},
        ]
        enriched = transcribe_segments(samples, segments, MockASRAdapter(confidence=0.7))
        self.assertEqual(len(enriched), 2)
        for original, out in zip(segments, enriched):
            self.assertEqual(out["segment_id"], original["segment_id"])  # original keys kept
            self.assertIn("text", out)
            self.assertEqual(out["asr_confidence"], 0.7)

    def test_empty_clip_yields_zero_confidence(self) -> None:
        segments = [{"meeting_id": "m", "segment_id": "m-0000", "start_time": 5.0, "end_time": 5.0}]
        enriched = transcribe_segments(_tone(1.0), segments, MockASRAdapter())
        self.assertEqual(enriched[0]["text"], "")
        self.assertEqual(enriched[0]["asr_confidence"], 0.0)


class NormalizerTests(unittest.TestCase):
    def test_from_whisper_result_maps_segments(self) -> None:
        raw = {
            "text": " hello world ",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " hello", "avg_logprob": 0.0, "no_speech_prob": 0.0},
                {"start": 1.0, "end": 2.0, "text": " world", "avg_logprob": -1.0, "no_speech_prob": 0.0},
            ],
        }
        out = _from_whisper_result(raw, "whisper")
        self.assertEqual(out["text"], "hello world")
        self.assertEqual(out["language"], "en")
        self.assertEqual(len(out["segments"]), 2)
        self.assertAlmostEqual(out["segments"][0]["asr_confidence"], 1.0, places=5)
        self.assertLess(out["segments"][1]["asr_confidence"], 1.0)

    def test_from_funasr_result_without_sentences(self) -> None:
        out = _from_funasr_result({"text": "你好"}, "funasr", default_confidence=0.6, duration=2.0)
        self.assertEqual(out["text"], "你好")
        self.assertEqual(out["language"], "zh")
        self.assertEqual(len(out["segments"]), 1)
        self.assertEqual(out["segments"][0]["asr_confidence"], 0.6)

    def test_from_funasr_result_with_sentence_timestamps(self) -> None:
        raw = {"text": "你好世界", "sentences": [
            {"text": "你好", "start": 0, "end": 1000},
            {"text": "世界", "start": 1000, "end": 2000},
        ]}
        out = _from_funasr_result(raw, "funasr", default_confidence=0.6, duration=2.0)
        self.assertEqual(len(out["segments"]), 2)
        self.assertAlmostEqual(out["segments"][1]["start_time"], 1.0, places=5)
        self.assertAlmostEqual(out["segments"][1]["end_time"], 2.0, places=5)

    def test_from_whisperx_result_uses_word_scores(self) -> None:
        raw = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "hello",
                    "words": [{"word": "hello", "score": 0.8}, {"word": "there", "score": 0.6}],
                }
            ],
        }
        out = _from_whisperx_result(raw, "whisperx", default_confidence=0.5, duration=1.5)
        self.assertEqual(out["text"], "hello")
        self.assertEqual(out["language"], "en")
        self.assertAlmostEqual(out["segments"][0]["asr_confidence"], 0.7, places=5)

    def test_aggregate_confidence_is_duration_weighted(self) -> None:
        segments = [
            {"start_time": 0.0, "end_time": 1.0, "asr_confidence": 1.0},
            {"start_time": 1.0, "end_time": 3.0, "asr_confidence": 0.0},  # twice as long
        ]
        # weighted mean = (1*1.0 + 2*0.0) / 3 = 0.333...
        self.assertAlmostEqual(_aggregate_confidence(segments), 1.0 / 3.0, places=5)

    def test_aggregate_confidence_empty_is_zero(self) -> None:
        self.assertEqual(_aggregate_confidence([]), 0.0)


if __name__ == "__main__":
    unittest.main()
