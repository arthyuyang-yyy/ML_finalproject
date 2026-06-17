"""Tests for metadata (evidence-packet) schema validation.

Run with the standard library only::

    python -m unittest tests.test_schema_validation
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.metadata_builder import build_metadata_segment
from src.schema_validation import (
    validate_candidate,
    validate_meeting,
    validate_metadata_segment,
)

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "sample_meeting.json"


def _valid_low_overlap() -> dict:
    return build_metadata_segment(
        meeting_id="m1",
        segment_id="m1-1",
        speaker="SPEAKER_00",
        start_time=0.0,
        end_time=1.5,
        text="hello",
        processing_path="low_overlap_cluster",
        overlap_score=0.05,
        asr_confidence=0.9,
        speaker_confidence=0.85,
    )


class SampleMeetingFixtureTests(unittest.TestCase):
    def test_fixture_passes_validation(self) -> None:
        segments = json.loads(FIXTURE.read_text(encoding="utf-8"))
        validated = validate_meeting(segments)
        self.assertEqual(len(validated), 3)

    def test_fixture_has_a_high_overlap_segment_with_candidates(self) -> None:
        segments = json.loads(FIXTURE.read_text(encoding="utf-8"))
        high = [s for s in segments if s["processing_path"] == "high_overlap_candidate"]
        self.assertTrue(high)
        self.assertGreaterEqual(len(high[0]["candidates"]), 1)


class SegmentValidationTests(unittest.TestCase):
    def test_builder_output_is_valid(self) -> None:
        self.assertIs(validate_metadata_segment(_valid_low_overlap()).__class__, dict)

    def test_missing_field_raises(self) -> None:
        record = _valid_low_overlap()
        del record["speaker"]
        with self.assertRaisesRegex(ValueError, "speaker"):
            validate_metadata_segment(record)

    def test_score_out_of_range_raises(self) -> None:
        record = _valid_low_overlap()
        record["overlap_score"] = 1.4
        with self.assertRaisesRegex(ValueError, "overlap_score"):
            validate_metadata_segment(record)

    def test_reversed_time_raises(self) -> None:
        record = _valid_low_overlap()
        record["start_time"], record["end_time"] = 2.0, 1.0
        with self.assertRaisesRegex(ValueError, "end_time"):
            validate_metadata_segment(record)

    def test_unknown_processing_path_raises(self) -> None:
        record = _valid_low_overlap()
        record["processing_path"] = "magic_path"
        with self.assertRaisesRegex(ValueError, "processing_path"):
            validate_metadata_segment(record)

    def test_bool_is_not_accepted_as_number(self) -> None:
        record = _valid_low_overlap()
        record["asr_confidence"] = True
        with self.assertRaisesRegex(ValueError, "asr_confidence"):
            validate_metadata_segment(record)

    def test_high_overlap_without_candidates_raises(self) -> None:
        record = _valid_low_overlap()
        record["processing_path"] = "high_overlap_candidate"
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            validate_metadata_segment(record)

    def test_validate_meeting_empty_list_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_meeting([])

    def test_validate_meeting_rejects_non_list(self) -> None:
        with self.assertRaises(ValueError):
            validate_meeting("not a list")  # type: ignore[arg-type]

    def test_validate_metadata_segment_rejects_non_dict(self) -> None:
        with self.assertRaises(ValueError):
            validate_metadata_segment(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_strict_audio_clip_validation_accepts_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.wav"
            clip_path.touch()
            record = _valid_low_overlap()
            record["audio_clip_path"] = str(clip_path)

            self.assertEqual(
                validate_metadata_segment(record, require_audio_clip=True),
                record,
            )

    def test_strict_audio_clip_validation_rejects_missing_file(self) -> None:
        record = _valid_low_overlap()
        record["audio_clip_path"] = "missing/clip.wav"

        with self.assertRaisesRegex(ValueError, "does not exist or is not a file"):
            validate_metadata_segment(record, require_audio_clip=True)

    def test_strict_audio_clip_validation_rejects_empty_path(self) -> None:
        record = _valid_low_overlap()

        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            validate_metadata_segment(record, require_audio_clip=True)

    def test_strict_audio_clip_validation_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = _valid_low_overlap()
            record["audio_clip_path"] = tmp

            with self.assertRaisesRegex(ValueError, "does not exist or is not a file"):
                validate_metadata_segment(record, require_audio_clip=True)

    def test_low_overlap_requires_text(self) -> None:
        record = _valid_low_overlap()
        record["text"] = ""
        with self.assertRaisesRegex(ValueError, "must contain transcript text"):
            validate_metadata_segment(record)

    def test_low_overlap_rejects_candidates(self) -> None:
        record = _valid_low_overlap()
        record["candidates"] = [{
            "candidate_id": "c1",
            "speaker": "SPEAKER_00",
            "text": "alternate",
            "confidence": 0.5,
            "uncertainty_note": "not expected",
        }]
        with self.assertRaisesRegex(ValueError, "must not contain candidates"):
            validate_metadata_segment(record)

    def test_resolved_high_overlap_allows_speaker_and_text(self) -> None:
        record = _valid_low_overlap()
        record.update({
            "processing_path": "high_overlap_candidate",
            "speaker": "SPEAKER_00",
            "text": "resolved transcript",
            "candidates": [{
                "candidate_id": "c1",
                "speaker": "SPEAKER_00",
                "text": "alternate",
                "confidence": 0.5,
                "uncertainty_note": "overlap",
            }],
            "uncertainty_note": "overlap",
            "source": "llm_resolved",
            "decision_reason": "Candidate 1 is most consistent.",
        })
        self.assertEqual(validate_metadata_segment(record), record)

    def test_meeting_rejects_duplicate_evidence_ids(self) -> None:
        first = _valid_low_overlap()
        second = _valid_low_overlap()
        second["segment_id"] = "m1-2"
        with self.assertRaisesRegex(ValueError, "duplicate evidence_id"):
            validate_meeting([first, second])


class CandidateValidationTests(unittest.TestCase):
    def test_valid_candidate_passes(self) -> None:
        candidate = {
            "candidate_id": "m1-1_c1",
            "speaker": "SPEAKER_01",
            "text": "keep it",
            "confidence": 0.4,
            "uncertainty_note": "stream B",
        }
        self.assertEqual(validate_candidate(candidate), candidate)

    def test_candidate_missing_field_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "uncertainty_note"):
            validate_candidate({
                "candidate_id": "m1-1_c1",
                "speaker": "SPEAKER_01",
                "text": "keep it",
                "confidence": 0.4,
            })

    def test_candidate_confidence_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_candidate({
                "candidate_id": "m1-1_c1",
                "speaker": "X",
                "text": "x",
                "confidence": 2.0,
                "uncertainty_note": "",
            })


if __name__ == "__main__":
    unittest.main()
