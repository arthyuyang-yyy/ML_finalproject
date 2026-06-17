"""Tests for hash-embedding episodic-memory retrieval."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from src.memory.retriever import retrieve_episodes


def _episode(
    episode_id: str,
    *,
    event_type: str,
    content: str,
    topic: str,
    speakers: list[str],
    evidence_text: str,
    importance: float = 0.8,
    overlap_score: float = 0.1,
    memory_timestamp: str = "2026-01-01T00:00:00Z",
) -> dict:
    return {
        "episode_id": episode_id,
        "meeting_id": "meeting_001",
        "event_type": event_type,
        "topic": topic,
        "content": content,
        "speakers": speakers,
        "start_time": 1.0,
        "end_time": 2.0,
        "evidence_ids": [f"{episode_id}_evidence"],
        "evidence_text": evidence_text,
        "overlap_score": overlap_score,
        "confidence": "low" if event_type == "uncertainty" else "high",
        "importance": importance,
        "audio_clip_paths": [],
        "uncertainty_note": "Overlapping speech." if event_type == "uncertainty" else "",
        "memory_timestamp": memory_timestamp,
    }


class FixedEmbeddingBackend:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]


class MemoryRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = [
            _episode(
                "ep_action",
                event_type="action_item",
                content="SPEAKER_01 will test WhisperX and pyannote alignment.",
                topic="ASR baseline",
                speakers=["SPEAKER_01"],
                evidence_text="我来测试 WhisperX 和 pyannote 的对齐。",
                memory_timestamp="2026-01-01T00:00:00Z",
            ),
            _episode(
                "ep_uncertain",
                event_type="uncertainty",
                content="Gemma may be used for post-processing.",
                topic="Gemma usage",
                speakers=["MIXED"],
                evidence_text="Conflicting Gemma candidates.",
                overlap_score=0.9,
                importance=0.6,
                memory_timestamp="2026-02-01T00:00:00Z",
            ),
            _episode(
                "ep_decision",
                event_type="decision",
                content="Use WhisperX and pyannote as the front-end baseline.",
                topic="ASR baseline",
                speakers=["SPEAKER_00"],
                evidence_text="We decided on the front-end baseline.",
                memory_timestamp="2026-03-01T00:00:00Z",
            ),
        ]

    def test_chinese_owner_query_ranks_action_item_first(self) -> None:
        results = retrieve_episodes("谁负责测试 WhisperX？", episodes=self.episodes)
        self.assertEqual(results[0]["episode_id"], "ep_action")
        self.assertIn("retrieval", results[0])

    def test_uncertainty_query_ranks_uncertainty_first(self) -> None:
        results = retrieve_episodes("有哪些不确定的重叠片段？", episodes=self.episodes)
        self.assertEqual(results[0]["episode_id"], "ep_uncertain")

    def test_decision_query_uses_event_type_text(self) -> None:
        results = retrieve_episodes("decision baseline", episodes=self.episodes)
        self.assertEqual(results[0]["episode_id"], "ep_decision")

    def test_score_uses_documented_hash_and_keyword_weights(self) -> None:
        episodes = [
            _episode(
                "ep_match",
                event_type="decision",
                content="alpha",
                topic="alpha",
                speakers=["S1"],
                evidence_text="alpha",
                importance=0.8,
                overlap_score=0.8,
            ),
            _episode(
                "ep_other",
                event_type="decision",
                content="beta",
                topic="beta",
                speakers=["S2"],
                evidence_text="beta",
                importance=0.2,
                memory_timestamp="2026-02-01T00:00:00Z",
            ),
        ]
        result = retrieve_episodes(
            "alpha",
            episodes=episodes,
            embedding_backend=FixedEmbeddingBackend(),
        )[0]
        components = result["retrieval"]
        expected = (
            0.70 * components["embedding_similarity"]
            + 0.30 * components["keyword_score"]
        )
        self.assertAlmostEqual(components["final_score"], expected, places=6)
        self.assertEqual(components["embedding_backend"], "FixedEmbeddingBackend")

    def test_default_backend_is_custom_hash_embedding(self) -> None:
        results = retrieve_episodes("WhisperX", episodes=self.episodes)
        self.assertEqual(results[0]["retrieval"]["embedding_backend"], "HashingEmbeddingBackend")

    def test_retrieval_does_not_mutate_memory(self) -> None:
        original = deepcopy(self.episodes)
        retrieve_episodes("WhisperX", episodes=self.episodes)
        self.assertEqual(self.episodes, original)

    def test_reads_json_memory_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_memory.json"
            path.write_text(json.dumps(self.episodes), encoding="utf-8")
            results = retrieve_episodes("谁负责测试 WhisperX？", path=path, top_k=1)
            self.assertEqual(results[0]["episode_id"], "ep_action")

    def test_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            retrieve_episodes(" ", episodes=self.episodes)

    def test_filters_by_meeting_speaker_and_time(self) -> None:
        other = _episode(
            "ep_other_meeting",
            event_type="decision",
            content="Other meeting decision",
            topic="Other",
            speakers=["SPEAKER_99"],
            evidence_text="Other",
        )
        other["meeting_id"] = "meeting_002"
        other["start_time"] = 100.0
        other["end_time"] = 110.0
        results = retrieve_episodes(
            "decision",
            episodes=[*self.episodes, other],
            meeting_id="meeting_002",
            speaker="SPEAKER_99",
            start_time=105.0,
            end_time=120.0,
        )
        self.assertEqual([item["episode_id"] for item in results], ["ep_other_meeting"])

    def test_rejects_inverted_time_filter(self) -> None:
        with self.assertRaisesRegex(ValueError, "end_time filter"):
            retrieve_episodes("decision", episodes=self.episodes, start_time=10.0, end_time=1.0)


if __name__ == "__main__":
    unittest.main()
