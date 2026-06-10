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


if __name__ == "__main__":
    unittest.main()
