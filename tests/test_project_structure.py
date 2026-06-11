"""Contract tests for the documented project package structure."""

import unittest

from src.asr import MockASRAdapter, get_adapter
from src.audio.vad import segment_waveform
from src.candidates.generator import generate_high_overlap_candidates
from src.diarization import assign_speakers_to_segments
from src.evaluation.qa_metrics import citation_rate, timestamp_citation_rate
from src.evidence import build_metadata_segment, validate_evidence_segments
from src.llm.json_repair import parse_or_repair_json
from src.memory.episodic_store import build_episodes
from src.memory.memory_schema import Episode
from src.memory.retriever import retrieve_episodes
from src.overlap.router import route_segment
from src.qa.answerer import answer_question


def _evidence() -> dict:
    return build_metadata_segment(
        meeting_id="meeting_001",
        segment_id="m1_seg_001",
        speaker="SPEAKER_00",
        start_time=1.0,
        end_time=3.0,
        text="SPEAKER_00 will test WhisperX.",
        processing_path="low_overlap_cluster",
        overlap_score=0.1,
        asr_confidence=0.9,
        speaker_confidence=0.8,
        audio_clip_path="outputs/meeting_001/clips/m1_seg_001.wav",
    )


class ProjectStructureTests(unittest.TestCase):
    def test_core_package_entry_points_are_importable(self) -> None:
        self.assertIsInstance(get_adapter("mock"), MockASRAdapter)
        self.assertEqual(route_segment(0.2), "low_overlap_cluster")
        self.assertTrue(callable(segment_waveform))
        self.assertTrue(callable(generate_high_overlap_candidates))
        self.assertTrue(callable(assign_speakers_to_segments))

    def test_json_repair_handles_fences_and_trailing_comma(self) -> None:
        payload = parse_or_repair_json('```json\n{"meeting_id": "m1", "events": [],}\n```')
        self.assertEqual(payload["meeting_id"], "m1")
        self.assertEqual(payload["events"], [])

    def test_evidence_to_memory_to_qa_contract(self) -> None:
        evidence = _evidence()
        self.assertEqual(validate_evidence_segments([evidence]), [])

        events = {
            "meeting_id": "meeting_001",
            "meeting_summary": "SPEAKER_00 will test WhisperX.",
            "events": [{
                "event_id": "meeting_001_event_0001",
                "event_type": "action_item",
                "content": "SPEAKER_00 will test WhisperX.",
                "task": "Test WhisperX.",
                "owner": "SPEAKER_00",
                "speakers": ["SPEAKER_00"],
                "evidence_ids": ["m1_seg_001"],
                "confidence": "high",
            }],
        }
        episodes = build_episodes(events, [evidence])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["event_type"], "action_item")

        retrieved = retrieve_episodes("谁负责测试 WhisperX", episodes=episodes)
        answer = answer_question("谁负责测试 WhisperX", retrieved)
        self.assertIn("m1_seg_001", answer["answer"])
        self.assertIn("1.000-3.000s", answer["answer"])

    def test_episode_schema_matches_stored_shape(self) -> None:
        self.assertIn("content", Episode.__annotations__)
        self.assertIn("evidence_text", Episode.__annotations__)

    def test_build_episodes_rejects_missing_evidence_id(self) -> None:
        event = {"meeting_id": "meeting_001", "events": []}
        with self.assertRaisesRegex(ValueError, "evidence_id must be a non-empty string"):
            build_episodes(event, [{"meeting_id": "meeting_001"}])

    def test_build_episodes_rejects_duplicate_evidence_id(self) -> None:
        evidence = _evidence()
        with self.assertRaisesRegex(ValueError, "duplicate evidence ID"):
            build_episodes({"meeting_id": "meeting_001", "events": []}, [evidence, dict(evidence)])

    def test_build_episodes_rejects_null_event_evidence_id(self) -> None:
        event = {
            "meeting_id": "meeting_001",
            "events": [{
                "event_type": "decision",
                "content": "test",
                "evidence_ids": [None],
                "confidence": "low",
            }],
        }
        with self.assertRaisesRegex(ValueError, "evidence ID must be a non-empty string"):
            build_episodes(event, [_evidence()])

    def test_qa_metrics(self) -> None:
        answers = [{"evidence_ids": ["e1"], "timestamp": "0-1s"}, {}]
        self.assertEqual(citation_rate(answers), 0.5)
        self.assertEqual(timestamp_citation_rate(answers), 0.5)


if __name__ == "__main__":
    unittest.main()
