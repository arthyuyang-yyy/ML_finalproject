"""Tests for merging low/high-overlap results into canonical evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from src.evidence.builder import build_evidence_file, build_evidence_segments
from src.evidence.schema import EVIDENCE_SEGMENT_FIELDS


def _low_segment() -> dict:
    return {
        "meeting_id": "meeting_001",
        "segment_id": "m1_seg_012",
        "speaker": "SPEAKER_01",
        "start_time": 60.2,
        "end_time": 68.4,
        "text": "我来测试 WhisperX 和 pyannote 的对齐。",
        "processing_path": "low_overlap_cluster",
        "overlap_score": 0.08,
        "asr_confidence": 0.90,
        "speaker_confidence": 0.86,
        "candidates": [],
        "uncertainty_note": "",
        "audio_clip_path": "outputs/meeting_001/clips/m1_seg_012.wav",
    }


def _high_segment() -> dict:
    return {
        "meeting_id": "meeting_001",
        "segment_id": "m1_seg_013",
        "speaker": "MIXED",
        "start_time": 68.4,
        "end_time": 73.0,
        "text": "",
        "processing_path": "high_overlap_candidate",
        "overlap_score": 0.82,
        "asr_confidence": 0.48,
        "speaker_confidence": 0.30,
        "candidates": [
            {
                "speaker": "SPEAKER_00",
                "text": "We can use Gemma for post-processing.",
                "confidence": 0.61,
            },
            {
                "speaker": "SPEAKER_01",
                "text": "But not directly for full ASR.",
                "confidence": 0.56,
            },
        ],
        "uncertainty_note": (
            "Multiple speakers overlap; do not force a single speaker attribution."
        ),
        "audio_clip_path": "outputs/meeting_001/clips/m1_seg_013.wav",
    }


class EvidenceBuilderTests(unittest.TestCase):
    def test_merges_sorts_and_fills_canonical_fields(self) -> None:
        evidence = build_evidence_segments(
            [_low_segment()],
            [_high_segment()],
            source_audio_path="data/raw/meeting_001.wav",
            language="und",
        )

        self.assertEqual([item["segment_id"] for item in evidence], ["m1_seg_012", "m1_seg_013"])
        self.assertEqual(set(evidence[0]), set(EVIDENCE_SEGMENT_FIELDS))
        self.assertEqual(evidence[0]["evidence_id"], "m1_seg_012")
        self.assertEqual(evidence[0]["source_audio_path"], "data/raw/meeting_001.wav")
        self.assertIn("threshold=0.400", evidence[0]["route_reason"])

    def test_normalizes_document_candidate_shape(self) -> None:
        evidence = build_evidence_segments([], [_high_segment()])
        candidates = evidence[0]["candidates"]

        self.assertEqual(candidates[0]["candidate_id"], "m1_seg_013_c1")
        self.assertEqual(candidates[1]["candidate_id"], "m1_seg_013_c2")
        self.assertTrue(candidates[0]["uncertainty_note"])

    def test_rejects_segment_in_wrong_input_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "belongs to"):
            build_evidence_segments([_high_segment()], [])

    def test_rejects_route_score_mismatch(self) -> None:
        high = _high_segment()
        high["overlap_score"] = 0.2
        with self.assertRaisesRegex(ValueError, "routes to low_overlap_cluster"):
            build_evidence_segments([], [high])

    def test_rejects_duplicate_ids_across_paths(self) -> None:
        high = _high_segment()
        high["segment_id"] = "m1_seg_012"
        with self.assertRaisesRegex(ValueError, "duplicate segment_id"):
            build_evidence_segments([_low_segment()], [high])

    def test_builds_evidence_file_from_two_json_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low_path = root / "low.json"
            high_path = root / "high.json"
            output_path = root / "evidence_segments.json"
            low_path.write_text(json.dumps([_low_segment()], ensure_ascii=False), encoding="utf-8")
            high_path.write_text(json.dumps([_high_segment()], ensure_ascii=False), encoding="utf-8")

            result = build_evidence_file(low_path, high_path, output_path)

            self.assertEqual(len(result), 2)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), result)


class NoneCoercionTests(unittest.TestCase):
    def test_text_none_coerces_to_empty_on_high_overlap(self) -> None:
        segment = _high_segment()
        segment["text"] = None
        evidence = build_evidence_segments([], [segment])
        self.assertEqual(evidence[0]["text"], "")
        self.assertNotEqual(evidence[0]["text"], "None")

    def test_segment_id_none_raises(self) -> None:
        segment = _low_segment()
        segment["segment_id"] = None
        with self.assertRaisesRegex(ValueError, "segment_id must be a non-empty string"):
            build_evidence_segments([segment], [])

    def test_audio_clip_path_none_coerces_to_empty(self) -> None:
        segment = _low_segment()
        segment["audio_clip_path"] = None
        evidence = build_evidence_segments([segment], [])
        self.assertEqual(evidence[0]["audio_clip_path"], "")
        self.assertNotEqual(evidence[0]["audio_clip_path"], "None")
        self.assertFalse(evidence[0]["audio_clip_path"])

    def test_source_audio_path_none_coerces_to_empty(self) -> None:
        segment = _low_segment()
        segment["source_audio_path"] = None
        evidence = build_evidence_segments([segment], [], source_audio_path=None)
        self.assertEqual(evidence[0]["source_audio_path"], "")
        self.assertNotEqual(evidence[0]["source_audio_path"], "None")

    def test_empty_source_audio_and_language_fallback_to_parameter(self) -> None:
        """Empty string in segment dict means 'not provided' and falls back."""
        segment = _low_segment()
        segment["source_audio_path"] = ""
        segment["language"] = ""
        evidence = build_evidence_segments(
            [segment], [], source_audio_path="data/m1.wav", language="zh"
        )
        self.assertEqual(evidence[0]["source_audio_path"], "data/m1.wav")
        self.assertEqual(evidence[0]["language"], "zh")

    def test_distribution_label_none_raises(self) -> None:
        segment = _low_segment()
        segment["cluster_similarity_distribution"] = {None: 1.0}
        with self.assertRaisesRegex(ValueError, "cluster_similarity_distribution keys must be strings"):
            build_evidence_segments([segment], [])

    def test_distribution_numeric_key_raises(self) -> None:
        segment = _low_segment()
        segment["cluster_similarity_distribution"] = {123: 1.0}
        with self.assertRaisesRegex(ValueError, "cluster_similarity_distribution keys must be strings"):
            build_evidence_segments([segment], [])

    def test_speaker_none_raises(self) -> None:
        segment = _low_segment()
        segment["speaker"] = None
        with self.assertRaisesRegex(ValueError, "speaker must be a non-empty string"):
            build_evidence_segments([segment], [])

    def test_language_none_coerces_to_default(self) -> None:
        segment = _low_segment()
        evidence = build_evidence_segments([segment], [], language=None)
        self.assertEqual(evidence[0]["language"], "und")
        self.assertNotEqual(evidence[0]["language"], "None")

    def test_text_none_raises_on_low_overlap(self) -> None:
        segment = _low_segment()
        segment["text"] = None
        with self.assertRaisesRegex(ValueError, "must contain transcript text"):
            build_evidence_segments([segment], [])

    def test_missing_segment_id_raises(self) -> None:
        segment = _low_segment()
        del segment["segment_id"]
        with self.assertRaisesRegex(ValueError, "segment_id must be a non-empty string"):
            build_evidence_segments([segment], [])

    def test_missing_speaker_raises(self) -> None:
        segment = _low_segment()
        del segment["speaker"]
        with self.assertRaisesRegex(ValueError, "speaker must be a non-empty string"):
            build_evidence_segments([segment], [])

    def test_none_handling_consistent_across_overlap_paths(self) -> None:
        low = _low_segment()
        low["text"] = None
        high = _high_segment()
        high["text"] = None
        with self.subTest(path="low"):
            with self.assertRaises(ValueError):
                build_evidence_segments([low], [])
        with self.subTest(path="high"):
            evidence = build_evidence_segments([], [high])
            self.assertEqual(evidence[0]["text"], "")
        with self.subTest(path="direct_call"):
            from src.evidence.builder import _build_from_processed_segment

            segment = {
                "meeting_id": "m1",
                "segment_id": "s1",
                "speaker": "MIXED",
                "start_time": 0.0,
                "end_time": 1.0,
                "text": None,
                "overlap_score": 0.8,
                "asr_confidence": 0.5,
                "speaker_confidence": 0.3,
                "candidates": [{"speaker": "SP", "text": "x", "confidence": 0.5}],
                "uncertainty_note": "overlap",
            }
            result = _build_from_processed_segment(
                segment,
                expected_path="high_overlap_candidate",
                meeting_id=None,
                source_audio_path=None,
                language=None,
                overlap_threshold=0.4,
            )
            self.assertEqual(result["text"], "")

    def test_segment_id_null_vs_missing_same_error(self) -> None:
        """Missing key and explicit None must both be rejected with the same message."""
        null_seg = _low_segment()
        null_seg["segment_id"] = None
        missing_seg = _low_segment()
        del missing_seg["segment_id"]

        with self.subTest(scenario="null"):
            with self.assertRaisesRegex(ValueError, "segment_id must be a non-empty string"):
                build_evidence_segments([null_seg], [])
        with self.subTest(scenario="missing"):
            with self.assertRaisesRegex(ValueError, "segment_id must be a non-empty string"):
                build_evidence_segments([missing_seg], [])

    def test_speaker_null_vs_missing_same_error(self) -> None:
        """Missing key and explicit None must both be rejected with the same message."""
        null_seg = _low_segment()
        null_seg["speaker"] = None
        missing_seg = _low_segment()
        del missing_seg["speaker"]

        with self.subTest(scenario="null"):
            with self.assertRaisesRegex(ValueError, "speaker must be a non-empty string"):
                build_evidence_segments([null_seg], [])
        with self.subTest(scenario="missing"):
            with self.assertRaisesRegex(ValueError, "speaker must be a non-empty string"):
                build_evidence_segments([missing_seg], [])

    def test_build_from_processed_segment_direct_call_missing_segment_id(self) -> None:
        """Direct helper call must reject missing segment_id with a clear message."""
        from src.evidence.builder import _build_from_processed_segment

        segment = {
            "meeting_id": "m1",
            "speaker": "SP",
            "start_time": 0.0,
            "end_time": 1.0,
            "text": "hello",
            "overlap_score": 0.1,
            "asr_confidence": 0.5,
            "speaker_confidence": 0.3,
            "candidates": [],
            "uncertainty_note": "",
        }
        with self.assertRaisesRegex(ValueError, "segment_id must be a non-empty string"):
            _build_from_processed_segment(
                segment,
                expected_path="low_overlap_cluster",
                meeting_id=None,
                source_audio_path=None,
                language="und",
                overlap_threshold=0.4,
            )

    def test_build_from_processed_segment_direct_call_missing_speaker(self) -> None:
        """Direct helper call must reject missing speaker with a clear message."""
        from src.evidence.builder import _build_from_processed_segment

        segment = {
            "meeting_id": "m1",
            "segment_id": "s1",
            "start_time": 0.0,
            "end_time": 1.0,
            "text": "hello",
            "overlap_score": 0.1,
            "asr_confidence": 0.5,
            "speaker_confidence": 0.3,
            "candidates": [],
            "uncertainty_note": "",
        }
        with self.assertRaisesRegex(ValueError, "speaker must be a non-empty string"):
            _build_from_processed_segment(
                segment,
                expected_path="low_overlap_cluster",
                meeting_id=None,
                source_audio_path=None,
                language="und",
                overlap_threshold=0.4,
            )

    def test_build_from_processed_segment_direct_call_with_nulls(self) -> None:
        """Directly calling the internal helper must also coerce None correctly."""
        from src.evidence.builder import _build_from_processed_segment

        segment = {
            "meeting_id": "m1",
            "segment_id": "s1",
            "speaker": "MIXED",
            "start_time": 0.0,
            "end_time": 1.0,
            "text": None,
            "overlap_score": 0.8,
            "asr_confidence": 0.5,
            "speaker_confidence": 0.3,
            "candidates": [
                {"speaker": "SP", "text": "x", "confidence": 0.5},
            ],
            "uncertainty_note": "overlap",
        }
        result = _build_from_processed_segment(
            segment,
            expected_path="high_overlap_candidate",
            meeting_id=None,
            source_audio_path=None,
            language=None,
            overlap_threshold=0.4,
        )
        self.assertEqual(result["text"], "")
        self.assertEqual(result["source_audio_path"], "")
        self.assertEqual(result["language"], "und")
        self.assertNotEqual(result["text"], "None")
        self.assertNotEqual(result["source_audio_path"], "None")

    def test_json_with_null_fields(self) -> None:
        """JSON deserialisation may produce null values; they must not become 'None' strings."""
        raw = json.dumps([{
            "meeting_id": "m1",
            "segment_id": "s1",
            "speaker": "MIXED",
            "start_time": 0.0,
            "end_time": 1.0,
            "text": None,
            "processing_path": "high_overlap_candidate",
            "overlap_score": 0.82,
            "asr_confidence": 0.5,
            "speaker_confidence": 0.3,
            "candidates": [
                {"speaker": "SP", "text": "x", "confidence": 0.5},
            ],
            "uncertainty_note": "overlap",
            "audio_clip_path": None,
            "source_audio_path": None,
            "language": None,
        }], ensure_ascii=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low_path = root / "low.json"
            high_path = root / "high.json"
            low_path.write_text("[]", encoding="utf-8")
            high_path.write_text(raw, encoding="utf-8")

            result = build_evidence_file(low_path, high_path, root / "out.json")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["text"], "")
            self.assertEqual(result[0]["audio_clip_path"], "")
            self.assertEqual(result[0]["source_audio_path"], "")
            self.assertEqual(result[0]["language"], "und")
            self.assertNotEqual(result[0]["text"], "None")
            self.assertNotEqual(result[0]["audio_clip_path"], "None")


if __name__ == "__main__":
    unittest.main()
