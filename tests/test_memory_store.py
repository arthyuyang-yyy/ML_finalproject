"""Tests for event-level episodic-memory construction and persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from src.memory.episodic_store import (
    build_episodes,
    build_episodes_file,
    read_episodes,
    upsert_episodes,
)
from src.memory.memory_schema import validate_episode
from src.memory.retriever import retrieve_episodes


def _low_evidence(meeting_id: str = "meeting_001") -> dict:
    return {
        "meeting_id": meeting_id,
        "segment_id": "m1_seg_012",
        "evidence_id": "m1_seg_012",
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


def _high_evidence(meeting_id: str = "meeting_001") -> dict:
    return {
        "meeting_id": meeting_id,
        "segment_id": "m1_seg_013",
        "evidence_id": "m1_seg_013",
        "speaker": "MIXED",
        "start_time": 68.4,
        "end_time": 73.0,
        "text": "",
        "processing_path": "high_overlap_candidate",
        "overlap_score": 0.82,
        "asr_confidence": 0.48,
        "speaker_confidence": 0.30,
        "candidates": [{
            "candidate_id": "m1_seg_013_c1",
            "speaker": "SPEAKER_00",
            "text": "Use Gemma for post-processing.",
            "confidence": 0.61,
            "uncertainty_note": "overlap",
        }],
        "uncertainty_note": "High-overlap segment with conflicting candidates.",
        "audio_clip_path": "outputs/meeting_001/clips/m1_seg_013.wav",
    }


def _events(meeting_id: str = "meeting_001") -> dict:
    return {
        "meeting_id": meeting_id,
        "meeting_summary": "The team discussed the ASR baseline and Gemma usage.",
        "events": [
            {
                "event_id": "ev_001",
                "event_type": "action_item",
                "topic": "ASR baseline",
                "content": "SPEAKER_01 will test WhisperX and pyannote alignment.",
                "task": "Test WhisperX and pyannote alignment.",
                "owner": "SPEAKER_01",
                "speakers": ["SPEAKER_01"],
                "evidence_ids": ["m1_seg_012"],
                "confidence": "high",
                "keywords": ["WhisperX", "pyannote", "对齐"],
            },
            {
                "event_id": "ev_002",
                "event_type": "decision",
                "topic": "Gemma usage",
                "content": "Use Gemma for post-processing.",
                "speakers": ["SPEAKER_00"],
                "evidence_ids": ["m1_seg_013"],
                "confidence": "high",
                "importance": 0.95,
            },
        ],
    }


class EpisodeBuildTests(unittest.TestCase):
    def test_builds_traceable_action_item_episode(self) -> None:
        episode = build_episodes(_events(), [_low_evidence(), _high_evidence()])[0]
        self.assertEqual(episode["episode_id"], "meeting_001_ep_001")
        self.assertEqual(episode["event_type"], "action_item")
        self.assertEqual(episode["topic"], "ASR baseline")
        self.assertEqual(episode["speakers"], ["SPEAKER_01"])
        self.assertEqual(episode["evidence_text"], "我来测试 WhisperX 和 pyannote 的对齐。")
        self.assertEqual(episode["confidence"], "high")
        self.assertEqual(episode["importance"], 0.90)
        self.assertEqual(episode["audio_clip_paths"], ["outputs/meeting_001/clips/m1_seg_012.wav"])
        self.assertEqual(episode["event_id"], "ev_001")
        self.assertEqual(episode["keywords"], ["WhisperX", "pyannote", "对齐"])
        self.assertEqual(episode["quality_notes"], [])
        self.assertEqual(episode["validation_warnings"], [])
        self.assertEqual(
            episode["evidence_quality"],
            {
                "min_asr_confidence": 0.9,
                "min_speaker_confidence": 0.86,
                "max_overlap_score": 0.08,
                "has_high_overlap": False,
                "has_unresolved_evidence": False,
                "evidence_count": 1,
            },
        )

    def test_high_overlap_is_forced_to_uncertainty(self) -> None:
        episode = build_episodes(_events(), [_low_evidence(), _high_evidence()])[1]
        self.assertEqual(episode["event_type"], "uncertainty")
        self.assertEqual(episode["speakers"], ["MIXED"])
        self.assertEqual(episode["confidence"], "low")
        self.assertEqual(episode["importance"], 0.60)
        self.assertIn("Uncertain interpretation", episode["content"])
        self.assertIn("Candidate interpretations", episode["evidence_text"])
        self.assertIn("High-overlap evidence", episode["uncertainty_note"])

    def test_low_quality_evidence_caps_episode_confidence(self) -> None:
        evidence = _low_evidence()
        evidence["asr_confidence"] = 0.60
        episode = build_episodes(_events(), [evidence, _high_evidence()])[0]
        self.assertEqual(episode["confidence"], "medium")
        self.assertEqual(episode["importance"], 0.70)
        self.assertIn("minimum ASR confidence is 0.60", episode["uncertainty_note"])
        self.assertTrue(any("minimum ASR confidence is 0.60" in note for note in episode["quality_notes"]))
        self.assertEqual(episode["evidence_quality"]["min_asr_confidence"], 0.6)

    def test_very_low_speaker_confidence_caps_episode_to_low(self) -> None:
        evidence = _low_evidence()
        evidence["speaker_confidence"] = 0.45
        episode = build_episodes(_events(), [evidence, _high_evidence()])[0]
        self.assertEqual(episode["confidence"], "low")
        self.assertEqual(episode["importance"], 0.60)
        self.assertIn("minimum speaker confidence is 0.45", episode["uncertainty_note"])

    def test_document_validation_warnings_are_preserved_in_episode_notes(self) -> None:
        events = _events()
        events["validation_warnings"] = ["events[2]: dropped unsupported action item"]
        episode = build_episodes(events, [_low_evidence(), _high_evidence()])[0]
        self.assertIn("dropped unsupported action item", episode["uncertainty_note"])

    def test_event_indexed_validation_warnings_attach_to_matching_episode(self) -> None:
        events = _events()
        events["validation_warnings"] = ["events[1]: event cites unsupported speakers"]
        episodes = build_episodes(events, [_low_evidence(), _high_evidence()])
        self.assertNotIn("unsupported speakers", episodes[0]["uncertainty_note"])
        self.assertIn("unsupported speakers", episodes[1]["uncertainty_note"])
        self.assertEqual(episodes[1]["validation_warnings"], [
            "Event validation warning for meeting_001: events[1]: event cites unsupported speakers"
        ])

    def test_rejects_unknown_evidence_id(self) -> None:
        events = _events()
        events["events"][0]["evidence_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            build_episodes(events, [_low_evidence(), _high_evidence()])

    def test_episode_validator_rejects_certain_uncertainty(self) -> None:
        episode = build_episodes(_events(), [_low_evidence(), _high_evidence()])[1]
        episode["confidence"] = "high"
        with self.assertRaisesRegex(ValueError, "must have low confidence"):
            validate_episode(episode)


class EpisodePersistenceTests(unittest.TestCase):
    def test_upsert_replaces_same_meeting_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory" / "episodic_memory.json"
            first = build_episodes(_events(), [_low_evidence(), _high_evidence()])
            upsert_episodes(path, first, meeting_id="meeting_001")
            second = [dict(first[0], content="Updated task content.")]
            merged = upsert_episodes(path, second, meeting_id="meeting_001")
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["content"], "Updated task content.")

    def test_upsert_preserves_other_meetings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_memory.json"
            first = build_episodes(_events(), [_low_evidence(), _high_evidence()])[:1]
            other = dict(first[0], episode_id="meeting_002_ep_001", meeting_id="meeting_002")
            upsert_episodes(path, first, meeting_id="meeting_001")
            merged = upsert_episodes(path, [other], meeting_id="meeting_002")
            self.assertEqual({item["meeting_id"] for item in merged}, {"meeting_001", "meeting_002"})

    def test_build_episodes_file_updates_long_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "meeting_events.json"
            evidence_path = root / "evidence_segments.json"
            memory_path = root / "memory" / "episodic_memory.json"
            events_path.write_text(json.dumps(_events(), ensure_ascii=False), encoding="utf-8")
            evidence_path.write_text(
                json.dumps([_low_evidence(), _high_evidence()], ensure_ascii=False),
                encoding="utf-8",
            )
            result = build_episodes_file(events_path, evidence_path, memory_path)
            self.assertEqual(read_episodes(memory_path), result)
            retrieved = retrieve_episodes("谁测试 WhisperX", path=memory_path)
            self.assertEqual(retrieved[0]["event_type"], "action_item")


if __name__ == "__main__":
    unittest.main()
