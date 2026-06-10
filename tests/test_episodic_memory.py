"""Tests for episodic memory: event grouping, storage, and retrieval.

Run with the standard library only::

    python -m unittest tests.test_episodic_memory

The semantic embedding backend is disabled so the tests exercise the
dependency-free lexical retrieval path deterministically.
"""

import os
import tempfile
import unittest
from pathlib import Path

import src.episodic_memory as episodic_memory
from src.episodic_memory import (
    create_episode_from_segments,
    create_episodes_from_segments,
    search_episodes,
    store_episodes,
)
from src.metadata_builder import build_metadata_segment


def _segment(segment_id: str, speaker: str, start: float, end: float, text: str) -> dict:
    return build_metadata_segment(
        meeting_id="m1",
        segment_id=segment_id,
        speaker=speaker,
        start_time=start,
        end_time=end,
        text=text,
        processing_path="low_overlap_cluster",
        overlap_score=0.05,
        asr_confidence=0.9,
        speaker_confidence=0.8,
    )


class EpisodeCreationTests(unittest.TestCase):
    def test_single_episode_aggregates_evidence_and_confidence(self) -> None:
        segments = [
            _segment("m1-1", "SPEAKER_00", 0.0, 1.0, "budget review"),
            _segment("m1-2", "SPEAKER_01", 1.0, 2.0, "timeline update"),
        ]
        episode = create_episode_from_segments(segments)

        self.assertEqual(episode["evidence_ids"], ["m1-1", "m1-2"])
        self.assertEqual(episode["speakers"], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(episode["start_time"], 0.0)
        self.assertEqual(episode["end_time"], 2.0)
        self.assertAlmostEqual(episode["confidence"], round(0.9 * 0.8, 3))

    def test_episode_inherits_event_metadata(self) -> None:
        episode = create_episode_from_segments(
            [_segment("m1-1", "SPEAKER_00", 0.0, 1.0, "hello")],
            episode_id="m1_event_0042",
            topic="kickoff",
            event_type="decision",
            importance=0.9,
        )
        self.assertEqual(episode["episode_id"], "m1_event_0042")
        self.assertEqual(episode["topic"], "kickoff")
        self.assertEqual(episode["event_type"], "decision")
        self.assertAlmostEqual(episode["importance"], 0.9)

    def test_episode_default_metadata_fields(self) -> None:
        episode = create_episode_from_segments(
            [_segment("m1-1", "SPEAKER_00", 0.0, 1.0, "hello")]
        )
        # event_type defaults to discussion; importance proxies confidence;
        # overlap_score is the mean of the evidence segments' overlap scores.
        self.assertEqual(episode["event_type"], "discussion")
        self.assertAlmostEqual(episode["importance"], episode["confidence"])
        self.assertAlmostEqual(episode["overlap_score"], 0.05)

    def test_empty_segments_raise(self) -> None:
        with self.assertRaises(ValueError):
            create_episode_from_segments([])


class EpisodeGroupingTests(unittest.TestCase):
    def test_groups_one_episode_per_event(self) -> None:
        segments = [
            _segment("m1-1", "SPEAKER_00", 0.0, 1.0, "budget"),
            _segment("m1-2", "SPEAKER_01", 1.0, 2.0, "schedule"),
        ]
        events = [
            {"event_id": "m1_event_0001", "summary": "budget", "evidence_ids": ["m1-1"]},
            {"event_id": "m1_event_0002", "summary": "schedule", "evidence_ids": ["m1-2"]},
        ]
        episodes = create_episodes_from_segments(segments, events)

        self.assertEqual(len(episodes), 2)
        self.assertEqual([ep["episode_id"] for ep in episodes], ["m1_event_0001", "m1_event_0002"])
        self.assertEqual([ep["topic"] for ep in episodes], ["budget", "schedule"])

    def test_uncovered_segments_fall_back_to_time_gap_episodes(self) -> None:
        segments = [
            _segment("m1-1", "SPEAKER_00", 0.0, 1.0, "covered"),
            _segment("m1-2", "SPEAKER_01", 100.0, 101.0, "uncovered"),
        ]
        events = [{"event_id": "m1_event_0001", "summary": "covered", "evidence_ids": ["m1-1"]}]
        episodes = create_episodes_from_segments(segments, events)

        self.assertEqual(len(episodes), 2)
        evidence_ids = {eid for ep in episodes for eid in ep["evidence_ids"]}
        self.assertEqual(evidence_ids, {"m1-1", "m1-2"})

    def test_time_gap_grouping_without_events(self) -> None:
        segments = [
            _segment("m1-1", "SPEAKER_00", 0.0, 1.0, "first"),
            _segment("m1-2", "SPEAKER_00", 1.5, 2.0, "still first"),
            _segment("m1-3", "SPEAKER_01", 90.0, 91.0, "much later"),
        ]
        episodes = create_episodes_from_segments(segments)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["evidence_ids"], ["m1-1", "m1-2"])
        self.assertEqual(episodes[1]["evidence_ids"], ["m1-3"])

    def test_no_segments_returns_empty(self) -> None:
        self.assertEqual(create_episodes_from_segments([]), [])


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        # Force the deterministic lexical path and reset the cached backend.
        self._prev = os.environ.get("EPISODIC_DISABLE_SEMANTIC")
        os.environ["EPISODIC_DISABLE_SEMANTIC"] = "1"
        episodic_memory._EMBEDDER = None

        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "memory.jsonl"
        segments = [
            _segment("m1-1", "SPEAKER_00", 0.0, 1.0, "quarterly budget approval"),
            _segment("m1-2", "SPEAKER_01", 60.0, 61.0, "release schedule slipped"),
        ]
        events = [
            {"event_id": "m1_event_0001", "summary": "quarterly budget approval", "evidence_ids": ["m1-1"]},
            {"event_id": "m1_event_0002", "summary": "release schedule slipped", "evidence_ids": ["m1-2"]},
        ]
        store_episodes(create_episodes_from_segments(segments, events), self.path)

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("EPISODIC_DISABLE_SEMANTIC", None)
        else:
            os.environ["EPISODIC_DISABLE_SEMANTIC"] = self._prev
        episodic_memory._EMBEDDER = None
        self._tmp.cleanup()

    def test_missing_memory_file_returns_empty(self) -> None:
        self.assertEqual(search_episodes("anything", path=self.path.parent / "missing.jsonl"), [])

    def test_lexical_ranking_prefers_matching_episode(self) -> None:
        results = search_episodes("budget", path=self.path)
        self.assertTrue(results)
        self.assertEqual(results[0]["episode_id"], "m1_event_0001")
        self.assertEqual(results[0]["retrieval_method"], "lexical")
        self.assertGreater(results[0]["retrieval_score"], 0.0)

    def test_speaker_filter(self) -> None:
        results = search_episodes("schedule", path=self.path, speaker="SPEAKER_01")
        self.assertEqual({ep["episode_id"] for ep in results}, {"m1_event_0002"})

    def test_meeting_filter_excludes_other_meetings(self) -> None:
        self.assertEqual(search_episodes("budget", path=self.path, meeting_id="other"), [])

    def test_time_range_filter(self) -> None:
        results = search_episodes("budget schedule", path=self.path, time_range=(0.0, 10.0))
        self.assertEqual({ep["episode_id"] for ep in results}, {"m1_event_0001"})

    def test_high_overlap_episode_is_penalized(self) -> None:
        # Two episodes match the query equally; the high-overlap one should rank
        # lower because uncertain memories are penalized.
        path = Path(self._tmp.name) / "overlap.jsonl"
        clean = build_metadata_segment(
            meeting_id="m2", segment_id="m2-1", speaker="SPEAKER_00",
            start_time=0.0, end_time=1.0, text="alpha topic",
            processing_path="low_overlap_cluster", overlap_score=0.05,
            asr_confidence=0.9, speaker_confidence=0.9,
        )
        noisy = build_metadata_segment(
            meeting_id="m2", segment_id="m2-2", speaker="SPEAKER_01",
            start_time=0.0, end_time=1.0, text="alpha topic",
            processing_path="low_overlap_cluster", overlap_score=0.9,
            asr_confidence=0.9, speaker_confidence=0.9,
        )
        store_episodes(
            [
                create_episode_from_segments([clean], episode_id="clean"),
                create_episode_from_segments([noisy], episode_id="noisy"),
            ],
            path,
        )
        results = search_episodes("alpha", path=path)
        self.assertEqual(results[0]["episode_id"], "clean")
        self.assertGreater(results[0]["retrieval_score"], results[1]["retrieval_score"])


if __name__ == "__main__":
    unittest.main()
