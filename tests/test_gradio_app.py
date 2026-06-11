"""Tests for Gradio demo data adapters and event handlers."""

import unittest
from unittest.mock import patch

from src.ui.gradio_app import (
    answer_demo_question,
    build_app,
    build_memory_rows,
    build_timeline_rows,
    candidate_audio_path,
    candidate_detail,
    prepare_demo_data,
    run_demo_pipeline,
)


def _low_segment() -> dict:
    return {
        "segment_id": "m1_seg_012",
        "speaker": "SPEAKER_01",
        "start_time": 60.2,
        "end_time": 68.4,
        "text": "我来测试 WhisperX。",
        "processing_path": "low_overlap_cluster",
        "overlap_score": 0.08,
        "candidates": [],
        "uncertainty_note": "",
    }


def _high_segment() -> dict:
    return {
        "segment_id": "m1_seg_013",
        "speaker": "MIXED",
        "start_time": 68.4,
        "end_time": 73.0,
        "text": "",
        "processing_path": "high_overlap_candidate",
        "overlap_score": 0.82,
        "candidates": [
            {"speaker": "SPEAKER_00", "text": "Use Gemma for post-processing."},
            {"speaker": "SPEAKER_01", "text": "Not directly for full ASR."},
        ],
        "uncertainty_note": "Speaker attribution is uncertain.",
        "audio_clip_path": "outputs/meeting_001/clips/m1_seg_013.wav",
    }


def _episode() -> dict:
    return {
        "episode_id": "m1_ep_001",
        "meeting_id": "meeting_001",
        "event_type": "action_item",
        "topic": "WhisperX testing",
        "content": "SPEAKER_01 负责测试 WhisperX。",
        "speakers": ["SPEAKER_01"],
        "start_time": 60.2,
        "end_time": 68.4,
        "evidence_ids": ["m1_seg_012"],
        "evidence_text": "我来测试 WhisperX。",
        "overlap_score": 0.08,
        "confidence": "high",
        "importance": 0.9,
        "audio_clip_paths": [],
        "uncertainty_note": "",
        "memory_timestamp": "2026-06-11T00:00:00Z",
    }


class GradioAdapterTests(unittest.TestCase):
    def test_timeline_formats_paths_and_candidate_placeholder(self) -> None:
        rows = build_timeline_rows([_low_segment(), _high_segment()])
        self.assertEqual(rows[0][:3], ["01:00.2–01:08.4", "SPEAKER_01", "low_overlap"])
        self.assertEqual(rows[1][2], "high_overlap")
        self.assertEqual(rows[1][4], "candidates available")

    def test_candidate_detail_uses_presentation_schema(self) -> None:
        state = {"evidence_segments": [_low_segment(), _high_segment()]}
        detail = candidate_detail("m1_seg_013", state)
        self.assertEqual(detail["overlap_score"], 0.82)
        self.assertEqual(
            detail["candidates"][0],
            "SPEAKER_00: Use Gemma for post-processing.",
        )

    def test_candidate_audio_path_returns_traceable_clip(self) -> None:
        state = {"evidence_segments": [_low_segment(), _high_segment()]}
        self.assertEqual(
            candidate_audio_path("m1_seg_013", state),
            "outputs/meeting_001/clips/m1_seg_013.wav",
        )

    def test_memory_table_contains_traceable_fields(self) -> None:
        self.assertEqual(
            build_memory_rows([_episode()]),
            [["action_item", "SPEAKER_01 负责测试 WhisperX。", "m1_seg_012", "high"]],
        )

    def test_prepare_demo_data_populates_all_areas(self) -> None:
        result = {
            "meeting_id": "meeting_001",
            "evidence_segments": [_low_segment(), _high_segment()],
            "episodic_memory": [_episode()],
        }
        view = prepare_demo_data(result)
        self.assertEqual(len(view["timeline"]), 2)
        self.assertEqual(view["selected_candidate"], "m1_seg_013")
        self.assertEqual(len(view["memory"]), 1)

    def test_qa_retrieves_only_current_meeting_state(self) -> None:
        answer, retrieval_rows, result = answer_demo_question(
            "谁负责测试 WhisperX？",
            {"episodic_memory": [_episode()]},
        )
        self.assertIn("m1_seg_012", answer)
        self.assertEqual(retrieval_rows[0][0], "m1_ep_001")
        self.assertFalse(result["insufficient_evidence"])

    def test_qa_without_pipeline_state_is_insufficient(self) -> None:
        answer, rows, result = answer_demo_question("谁负责？", {})
        self.assertIn("无法确定", answer)
        self.assertEqual(rows, [])
        self.assertTrue(result["insufficient_evidence"])

    def test_unrelated_question_is_insufficient(self) -> None:
        answer, rows, result = answer_demo_question(
            "who discovered Neptune?",
            {"episodic_memory": [_episode()]},
        )
        self.assertIn("无法确定", answer)
        self.assertEqual(rows, [])
        self.assertTrue(result["insufficient_evidence"])

    @patch("src.ui.gradio_app.answer_question")
    def test_qa_passes_configured_gemma_client(self, mocked_answer) -> None:
        mocked_answer.return_value = {"answer": "ok", "insufficient_evidence": False}
        client = object()
        answer_demo_question("test", {"episodic_memory": [_episode()]}, client=client)
        self.assertIs(mocked_answer.call_args.kwargs["client"], client)

    def test_run_pipeline_rejects_unsafe_meeting_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "Meeting ID"):
            run_demo_pipeline("meeting.wav", "../escape")

    @patch("src.ui.gradio_app.run_meeting_pipeline")
    def test_run_pipeline_calls_shared_pipeline(self, mocked_run) -> None:
        mocked_run.return_value = {"meeting_id": "meeting_001"}
        result = run_demo_pipeline("meeting.wav", "meeting_001")
        mocked_run.assert_called_once()
        self.assertEqual(mocked_run.call_args.args, ("meeting.wav", "meeting_001"))
        self.assertEqual(mocked_run.call_args.kwargs["config"].low_overlap_asr_model, "auto")
        self.assertEqual(result["meeting_id"], "meeting_001")

    def test_build_app_has_five_area_components(self) -> None:
        app = build_app()
        config = app.get_config_file()
        labels = {component.get("props", {}).get("label") for component in config["components"]}
        self.assertIn("Meeting audio", labels)
        self.assertIn("High-overlap segment", labels)
        self.assertIn("High-overlap audio", labels)
        self.assertIn("Candidate detail", labels)
        self.assertIn("Question", labels)


if __name__ == "__main__":
    unittest.main()
