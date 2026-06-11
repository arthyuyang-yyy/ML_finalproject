"""Tests for structured Gemma event extraction and validation."""

import json
import tempfile
import unittest
from pathlib import Path

from src.llm.event_extractor import extract_meeting_events, extract_meeting_events_file
from src.llm.event_validator import validate_meeting_event, validate_meeting_events_document
from src.llm.gemma_client import GemmaClient
from src.llm.prompts import build_event_extraction_prompt


def _make_evidence_segments(meeting_id: str = "m1", include_high: bool = False) -> list[dict]:
    segments = [
        {
            "meeting_id": meeting_id,
            "segment_id": "m1_seg_001",
            "evidence_id": "m1_seg_001",
            "speaker": "SPEAKER_00",
            "start_time": 0.0,
            "end_time": 3.0,
            "text": "Use WhisperX as the baseline.",
            "processing_path": "low_overlap_cluster",
            "route_reason": "",
            "overlap_score": 0.04,
            "asr_confidence": 0.95,
            "speaker_confidence": 0.91,
            "audio_clip_path": "",
            "source_audio_path": "",
            "language": "en",
            "candidates": [],
            "uncertainty_note": "",
        },
        {
            "meeting_id": meeting_id,
            "segment_id": "m1_seg_002",
            "evidence_id": "m1_seg_002",
            "speaker": "SPEAKER_01",
            "start_time": 3.0,
            "end_time": 6.0,
            "text": "I will test the alignment this Friday.",
            "processing_path": "low_overlap_cluster",
            "route_reason": "",
            "overlap_score": 0.09,
            "asr_confidence": 0.92,
            "speaker_confidence": 0.88,
            "audio_clip_path": "",
            "source_audio_path": "",
            "language": "en",
            "candidates": [],
            "uncertainty_note": "",
        },
    ]
    if include_high:
        segments.append({
            "meeting_id": meeting_id,
            "segment_id": "m1_seg_003",
            "evidence_id": "m1_seg_003",
            "speaker": "MIXED",
            "start_time": 6.0,
            "end_time": 8.0,
            "text": "",
            "processing_path": "high_overlap_candidate",
            "route_reason": "",
            "overlap_score": 0.82,
            "asr_confidence": 0.55,
            "speaker_confidence": 0.3,
            "audio_clip_path": "",
            "source_audio_path": "",
            "language": "en",
            "candidates": [{
                "candidate_id": "m1_seg_003_c1",
                "speaker": "SPEAKER_00",
                "text": "Use Gemma for post-processing.",
                "confidence": 0.55,
                "uncertainty_note": "overlap",
            }],
            "uncertainty_note": "Multiple speakers overlap.",
        })
    return segments


def _valid_document() -> dict:
    return {
        "meeting_id": "m1",
        "meeting_summary": "The team selected an ASR baseline and assigned alignment testing.",
        "events": [
            {
                "event_id": "ev_001",
                "event_type": "decision",
                "content": "Use WhisperX as the baseline.",
                "speakers": ["SPEAKER_00"],
                "evidence_ids": ["m1_seg_001"],
                "confidence": "high",
            },
            {
                "event_id": "ev_002",
                "event_type": "action_item",
                "content": "Test the alignment this Friday.",
                "task": "Test the alignment.",
                "owner": "SPEAKER_01",
                "deadline": "this Friday",
                "speakers": ["SPEAKER_01"],
                "evidence_ids": ["m1_seg_002"],
                "confidence": "high",
            },
        ],
    }


class StaticClient(GemmaClient):
    def __init__(self, outputs: list[dict | str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate_json(self, prompt: str) -> dict | str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class EventExtractionTests(unittest.TestCase):
    def test_empty_segments_returns_empty_document(self) -> None:
        self.assertEqual(
            extract_meeting_events([]),
            {"meeting_id": "", "meeting_summary": "", "events": []},
        )

    def test_fallback_creates_structured_document(self) -> None:
        document = extract_meeting_events(_make_evidence_segments())
        self.assertEqual(document["meeting_id"], "m1")
        self.assertIn("WhisperX", document["meeting_summary"])
        self.assertEqual(document["events"][0]["event_id"], "ev_001")
        self.assertEqual(document["events"][0]["event_type"], "speaker_stance")

    def test_fallback_preserves_high_overlap_as_low_confidence_uncertainty(self) -> None:
        document = extract_meeting_events(_make_evidence_segments(include_high=True))
        uncertainty = [event for event in document["events"] if event["event_type"] == "uncertainty"]
        self.assertEqual(len(uncertainty), 1)
        self.assertEqual(uncertainty[0]["confidence"], "low")
        self.assertEqual(uncertainty[0]["evidence_ids"], ["m1_seg_003"])

    def test_valid_gemma_document_is_used(self) -> None:
        document = extract_meeting_events(
            _make_evidence_segments(),
            client=StaticClient([_valid_document()]),
        )
        self.assertEqual(document["events"][0]["event_type"], "decision")
        self.assertEqual(document["events"][1]["owner"], "SPEAKER_01")

    def test_json_string_is_repaired_and_validated(self) -> None:
        raw = "```json\n" + json.dumps(_valid_document())[:-1] + ",}\n```"
        document = extract_meeting_events(
            _make_evidence_segments(),
            client=StaticClient([raw]),
        )
        self.assertEqual(len(document["events"]), 2)

    def test_invalid_first_response_triggers_repair_attempt(self) -> None:
        client = StaticClient(["not json", _valid_document()])
        document = extract_meeting_events(_make_evidence_segments(), client=client)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("previous output was invalid", client.prompts[1])
        self.assertEqual(document["events"][0]["event_type"], "decision")

    def test_final_invalid_event_is_deleted_when_other_events_are_valid(self) -> None:
        invalid = _valid_document()
        invalid["events"].append({
            "event_id": "ev_003",
            "event_type": "decision",
            "content": "Unsupported decision.",
            "speakers": [],
            "evidence_ids": ["fake_id"],
            "confidence": "high",
        })
        client = StaticClient([invalid, invalid])
        document = extract_meeting_events(_make_evidence_segments(), client=client)
        self.assertEqual(len(document["events"]), 2)
        self.assertIn("validation_warnings", document)

    def test_file_entry_point_writes_meeting_events_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_path = root / "evidence_segments.json"
            output_path = root / "meeting_events.json"
            evidence_path.write_text(json.dumps(_make_evidence_segments()), encoding="utf-8")
            document = extract_meeting_events_file(evidence_path, output_path)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), document)


class EventValidationTests(unittest.TestCase):
    def test_valid_event_passes(self) -> None:
        event = _valid_document()["events"][0]
        validated = validate_meeting_event(event, known_evidence_ids={"m1_seg_001"})
        self.assertEqual(validated, event)

    def test_missing_event_id_raises(self) -> None:
        event = _valid_document()["events"][0]
        del event["event_id"]
        with self.assertRaisesRegex(ValueError, "event_id"):
            validate_meeting_event(event)

    def test_event_without_evidence_is_rejected(self) -> None:
        event = _valid_document()["events"][0]
        event["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_meeting_event(event)

    def test_unknown_evidence_id_raises(self) -> None:
        event = _valid_document()["events"][0]
        event["evidence_ids"] = ["fake_id"]
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            validate_meeting_event(event, known_evidence_ids={"m1_seg_001"})

    def test_action_item_requires_task_and_owner(self) -> None:
        event = _valid_document()["events"][1]
        del event["owner"]
        with self.assertRaisesRegex(ValueError, "action_item.owner"):
            validate_meeting_event(event)

    def test_action_item_owner_must_be_supported_or_uncertain(self) -> None:
        document = _valid_document()
        document["events"][1]["owner"] = "SPEAKER_99"
        with self.assertRaisesRegex(ValueError, "owner is not supported"):
            validate_meeting_events_document(document, _make_evidence_segments())

        document["events"][1]["owner"] = "uncertain"
        validated = validate_meeting_events_document(document, _make_evidence_segments())
        self.assertEqual(validated["events"][1]["owner"], "uncertain")

    def test_event_speaker_must_be_supported_by_cited_evidence(self) -> None:
        document = _valid_document()
        document["events"][0]["speakers"] = ["SPEAKER_99"]
        with self.assertRaisesRegex(ValueError, "unsupported speakers"):
            validate_meeting_events_document(document, _make_evidence_segments())

    def test_high_overlap_high_confidence_is_reduced(self) -> None:
        evidence = _make_evidence_segments(include_high=True)
        document = {
            "meeting_id": "m1",
            "meeting_summary": "Gemma usage was discussed with uncertainty.",
            "events": [{
                "event_id": "ev_003",
                "event_type": "uncertainty",
                "content": "Gemma usage is unclear.",
                "speakers": [],
                "evidence_ids": ["m1_seg_003"],
                "confidence": "high",
            }],
        }
        validated = validate_meeting_events_document(document, evidence)
        self.assertEqual(validated["events"][0]["confidence"], "low")
        self.assertIn("Confidence reduced", validated["events"][0]["uncertainty_note"])

    def test_document_rejects_wrong_meeting_id(self) -> None:
        document = _valid_document()
        document["meeting_id"] = "other"
        with self.assertRaisesRegex(ValueError, "meeting_id"):
            validate_meeting_events_document(document, _make_evidence_segments())


class GemmaClientTests(unittest.TestCase):
    def test_generate_json_returns_placeholder(self) -> None:
        result = GemmaClient().generate_json("test prompt")
        self.assertIsInstance(result, dict)
        self.assertIn("events", result)
        self.assertEqual(result["events"], [])

    def test_configured_generator_is_used(self) -> None:
        client = GemmaClient(generator=lambda prompt: {"prompt": prompt})
        self.assertEqual(client.generate_json("hello"), {"prompt": "hello"})


class PromptTests(unittest.TestCase):
    def test_prompt_includes_schema_and_evidence(self) -> None:
        prompt = build_event_extraction_prompt(_make_evidence_segments())
        self.assertIn("Allowed event_type values", prompt)
        self.assertIn("action_item", prompt)
        self.assertIn("evidence_ids", prompt)
        self.assertIn("Use WhisperX as the baseline", prompt)

    def test_prompt_forbids_high_confidence_overlap_claims(self) -> None:
        prompt = build_event_extraction_prompt(_make_evidence_segments(include_high=True))
        self.assertIn("Never mark an event \"high\"", prompt)
        self.assertIn("owner", prompt)


if __name__ == "__main__":
    unittest.main()
