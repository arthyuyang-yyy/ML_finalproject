"""Tests for the lightweight end-to-end meeting pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.pipeline.config import PipelineConfig
from src.pipeline.io import read_json
from src.pipeline.run_pipeline import run_meeting_pipeline, split_segments_by_overlap_regions
from src.audio.preprocess import load_audio
from src.llm.gemma_client import GemmaClient


def _stub_vad_segments(samples: np.ndarray, sample_rate: int, meeting_id: str = "meeting", **_kwargs):
    """Deterministic VAD stub: one segment covering the first second of audio.

    silero VAD is a learned model and will not fire on a synthetic sine tone, so
    pipeline integration tests stub the segment source to exercise orchestration
    rather than the detector.
    """
    duration = len(samples) / sample_rate
    end = round(min(1.0, duration), 3)
    return [{
        "meeting_id": meeting_id,
        "segment_id": f"{meeting_id}_seg_001",
        "start_time": 0.0,
        "end_time": end,
    }]


class PipelineTests(unittest.TestCase):
    def test_split_segments_by_authoritative_overlap_regions(self) -> None:
        segments = [{
            "meeting_id": "meeting_split",
            "segment_id": "meeting_split_seg_001",
            "start_time": 0.0,
            "end_time": 10.0,
            "overlap_score": 0.2,
            "overlap_seconds": 1.0,
            "overlap_regions": [[2.0, 3.0]],
            "overlap_detector": "pyannote",
            "processing_path": "high_overlap_candidate",
        }]

        split = split_segments_by_overlap_regions(segments)

        self.assertEqual(
            [(item["segment_id"], item["processing_path"], item["start_time"], item["end_time"]) for item in split],
            [
                ("meeting_split_seg_001_low_01", "low_overlap_cluster", 0.0, 2.0),
                ("meeting_split_seg_001_ovl_01", "high_overlap_candidate", 2.0, 3.0),
                ("meeting_split_seg_001_low_02", "low_overlap_cluster", 3.0, 10.0),
            ],
        )
        self.assertEqual(split[1]["parent_segment_id"], "meeting_split_seg_001")
        self.assertEqual(split[1]["overlap_score"], 1.0)
        self.assertEqual(split[0]["overlap_score"], 0.0)

    def test_split_segments_decodes_tiny_overlap_with_context(self) -> None:
        segments = [{
            "meeting_id": "meeting_split",
            "segment_id": "meeting_split_seg_001",
            "start_time": 0.0,
            "end_time": 10.0,
            "overlap_score": 0.2,
            "overlap_seconds": 0.01,
            "overlap_regions": [[2.0, 2.01]],
            "overlap_detector": "pyannote",
            "processing_path": "high_overlap_candidate",
        }]

        split = split_segments_by_overlap_regions(
            segments,
            min_segment_seconds=2.0,
            decode_context_seconds=1.0,
        )

        high = [item for item in split if item["processing_path"] == "high_overlap_candidate"]
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0]["start_time"], 2.0)
        self.assertEqual(high[0]["end_time"], 2.01)
        self.assertEqual(high[0]["decode_start_time"], 1.0)
        self.assertEqual(high[0]["decode_end_time"], 3.01)
        self.assertTrue(high[0]["short_overlap_context_decode"])

    @patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments)
    def test_pipeline_passes_vad_boundary_settings(self, mocked_segment_waveform) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            run_meeting_pipeline(
                str(input_path),
                "meeting_vad_cfg",
                PipelineConfig(
                    outputs_root=root / "outputs",
                    vad_max_segment_s=30.0,
                    vad_speech_pad_ms=400,
                    vad_min_silence_ms=500,
                ),
            )

        kwargs = mocked_segment_waveform.call_args.kwargs
        self.assertEqual(kwargs["max_segment_s"], 30.0)
        self.assertEqual(kwargs["speech_pad_ms"], 400)
        self.assertEqual(kwargs["min_silence_ms"], 500)

    def test_pipeline_accepts_m4a_and_keeps_asr_input_standardized(self) -> None:
        try:
            import av
        except ImportError:
            self.skipTest("PyAV is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            samples = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sample_rate * 2) / sample_rate)).astype(np.float32)
            input_path = root / "input.m4a"
            output = av.open(str(input_path), "w")
            stream = output.add_stream("aac", rate=sample_rate)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="fltp", layout="mono")
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode(None):
                output.mux(packet)
            output.close()

            with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                result = run_meeting_pipeline(
                    str(input_path),
                    "meeting_m4a",
                    PipelineConfig(outputs_root=root / "outputs", low_overlap_asr_model="mock"),
                )

            evidence = read_json(Path(result["output_dir"]) / "evidence_segments.json")
            self.assertGreaterEqual(len(evidence), 1)
            self.assertTrue(all(segment["text"].startswith("[mock transcript") for segment in evidence))
            preprocessed, rate = load_audio(result["artifacts"]["preprocessed"])
            self.assertEqual(rate, 16000)
            self.assertEqual(preprocessed.ndim, 1)

    @patch("src.pipeline.run_pipeline.write_segment_clips")
    def test_pipeline_rejects_missing_exported_audio_clip(self, mocked_write_clips) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        def missing_clip_paths(samples, sample_rate, segments, output_dir):
            return [
                {**segment, "audio_clip_path": str(Path(output_dir) / "missing.wav")}
                for segment in segments
            ]

        mocked_write_clips.side_effect = missing_clip_paths
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            with self.assertRaisesRegex(ValueError, "does not exist or is not a file"):
                with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                    run_meeting_pipeline(
                        str(input_path),
                        "meeting_missing_clip",
                        PipelineConfig(outputs_root=root / "outputs"),
                    )

    def test_pipeline_accepts_structured_gemma_client(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            client = GemmaClient(generator=lambda prompt: {
                "meeting_id": "meeting_gemma",
                "meeting_summary": "The meeting selected the mock baseline.",
                "events": [{
                    "event_id": "ev_001",
                    "event_type": "decision",
                    "content": "Use the mock baseline for pipeline testing.",
                    # Single-speaker tone -> one cluster -> honest UNKNOWN label.
                    "speakers": ["UNKNOWN"],
                    "evidence_ids": ["meeting_gemma_seg_001"],
                    "confidence": "high",
                }],
            })
            with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                result = run_meeting_pipeline(
                    str(input_path),
                    "meeting_gemma",
                    PipelineConfig(outputs_root=root / "outputs"),
                    llm_client=client,
                )

            self.assertEqual(result["meeting_events"]["events"][0]["event_type"], "decision")
            stored = read_json(Path(result["output_dir"]) / "meeting_events.json")
            self.assertEqual(stored["meeting_summary"], "The meeting selected the mock baseline.")
            memory_path = Path(result["artifacts"]["long_term_episodic_memory"])
            self.assertTrue(memory_path.exists())
            memory = read_json(memory_path)
            self.assertEqual(memory[0]["event_type"], "decision")
            self.assertEqual(memory[0]["evidence_ids"], ["meeting_gemma_seg_001"])

            with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                repeated = run_meeting_pipeline(
                    str(input_path),
                    "meeting_gemma",
                    PipelineConfig(outputs_root=root / "outputs"),
                    llm_client=client,
                )
            self.assertEqual(repeated["long_term_memory_size"], 1)
            self.assertEqual(len(read_json(memory_path)), 1)

    def test_pipeline_does_not_reload_preprocessed_audio(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            with patch("src.audio.preprocess.load_audio", wraps=load_audio) as mocked_load:
                with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                    run_meeting_pipeline(
                        str(input_path),
                        "meeting_single_read",
                        PipelineConfig(outputs_root=root / "outputs"),
                    )

            self.assertEqual(mocked_load.call_count, 1)

    def test_run_meeting_pipeline_writes_per_meeting_artifacts(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            input_path = root / "input.wav"
            sf.write(input_path, audio, sample_rate)

            with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                result = run_meeting_pipeline(
                    str(input_path),
                    "meeting_test",
                    PipelineConfig(outputs_root=root / "outputs", language="en"),
                )

            output_dir = Path(result["output_dir"])
            self.assertEqual(output_dir.name, "meeting_test")
            self.assertTrue((output_dir / "preprocessed.wav").exists())
            self.assertTrue((output_dir / "vad_segments.json").exists())
            self.assertTrue((output_dir / "routed_segments.json").exists())
            self.assertTrue((output_dir / "diarization.json").exists())
            self.assertTrue((output_dir / "low_overlap_segments.json").exists())
            self.assertTrue((output_dir / "evidence_segments.json").exists())
            self.assertTrue((output_dir / "clips").exists())

            evidence = read_json(output_dir / "evidence_segments.json")
            self.assertGreaterEqual(len(evidence), 1)
            first = evidence[0]
            self.assertIn("evidence_id", first)
            self.assertIn("audio_clip_path", first)
            self.assertIn("route_reason", first)
            self.assertTrue(Path(first["audio_clip_path"]).exists())

            low_overlap = read_json(output_dir / "low_overlap_segments.json")
            self.assertGreaterEqual(len(low_overlap), 1)
            low = low_overlap[0]
            self.assertEqual(low["processing_path"], "low_overlap_cluster")
            self.assertIn("text", low)
            self.assertIn("speaker", low)
            self.assertIn("asr_confidence", low)
            self.assertIn("speaker_confidence", low)
            self.assertEqual(low["candidates"], [])

            meeting_events = read_json(output_dir / "meeting_events.json")
            self.assertEqual(meeting_events["meeting_id"], "meeting_test")
            self.assertIn("meeting_summary", meeting_events)
            self.assertIsInstance(meeting_events["events"], list)

    @patch("src.pipeline.run_pipeline.diarize_with_pyannote")
    def test_pipeline_passes_pyannote_turns_to_low_overlap_path(self, mocked_diarize) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        mocked_diarize.return_value = [
            {
                "speaker": "PYANNOTE_SPEAKER",
                "start_time": 0.0,
                "end_time": 1.0,
                "speaker_confidence": 1.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                result = run_meeting_pipeline(
                    str(input_path),
                    "meeting_diarization",
                    PipelineConfig(outputs_root=root / "outputs"),
                )

            output_dir = Path(result["output_dir"])
            mocked_diarize.assert_called_once_with(str(output_dir / "preprocessed.wav"))
            self.assertEqual(read_json(output_dir / "diarization.json"), mocked_diarize.return_value)
            low_overlap = read_json(output_dir / "low_overlap_segments.json")
            self.assertGreaterEqual(len(low_overlap), 1)
            self.assertEqual(low_overlap[0]["speaker"], "PYANNOTE_SPEAKER")

    def test_high_overlap_pipeline_output_preserves_candidates(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate * 2) / sample_rate
            audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            input_path = root / "input.wav"
            sf.write(input_path, audio, sample_rate)

            with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                result = run_meeting_pipeline(
                    str(input_path),
                    "meeting_high",
                    PipelineConfig(outputs_root=root / "outputs", language="en", overlap_threshold=0.0),
                )

            high_overlap = read_json(Path(result["output_dir"]) / "high_overlap_candidates.json")
            self.assertGreaterEqual(len(high_overlap), 1)
            segment = high_overlap[0]
            self.assertEqual(segment["speaker"], "MIXED")
            self.assertTrue(segment["text"].strip())
            self.assertEqual(segment["processing_path"], "high_overlap_candidate")
            self.assertEqual(segment["source"], "fallback_resolved")
            self.assertTrue(segment["decision_reason"].strip())
            self.assertGreaterEqual(len(segment["candidates"]), 1)
            self.assertIn("candidate_id", segment["candidates"][0])
            self.assertIn("decode_config", segment["candidates"][0])

    @patch("src.pipeline.run_pipeline.process_high_overlap_segments")
    @patch("src.pipeline.run_pipeline.estimate_segment_overlap_scores")
    def test_suspected_overlap_routes_high_but_preserves_baseline_text(
        self,
        mocked_overlap,
        mocked_high_overlap,
    ) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        def score_segments(samples, segments, sample_rate, **_kwargs):
            return [
                {
                    **segments[0],
                    "overlap_score": 0.25,
                    "overlap_seconds": 0.0,
                    "overlap_regions": [],
                    "overlap_detector": "pyannote",
                    "overlap_components": {
                        "osd_or_energy": 0.25,
                        "diarization_overlap": 0.0,
                        "speaker_change": 0.0,
                        "asr_instability": 0.0,
                    },
                }
            ]

        def high_overlap_segments(samples, segments, **_kwargs):
            return [
                {
                    **segments[0],
                    "speaker": "MIXED",
                    "text": "",
                    "processing_path": "high_overlap_candidate",
                    "asr_confidence": 0.95,
                    "speaker_confidence": 0.35,
                    "candidates": [{
                        "candidate_id": f"{segments[0]['segment_id']}_c1",
                        "speaker": "SPEAKER_01",
                        "text": "A divergent high-overlap hypothesis.",
                        "confidence": 0.95,
                        "uncertainty_note": "High-overlap segment.",
                    }],
                    "uncertainty_note": "Suspected high-overlap segment.",
                }
            ]

        mocked_overlap.side_effect = score_segments
        mocked_high_overlap.side_effect = high_overlap_segments

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            t = np.arange(sample_rate) / sample_rate
            input_path = root / "input.wav"
            sf.write(input_path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

            with patch("src.pipeline.run_pipeline.segment_waveform", side_effect=_stub_vad_segments):
                result = run_meeting_pipeline(
                    str(input_path),
                    "meeting_suspected",
                    PipelineConfig(
                        outputs_root=root / "outputs",
                        low_overlap_asr_model="mock",
                        overlap_threshold=0.4,
                        suspected_overlap_threshold=0.2,
                    ),
                )

            output_dir = Path(result["output_dir"])
            routed = read_json(output_dir / "routed_segments.json")
            self.assertEqual(routed[0]["processing_path"], "high_overlap_candidate")
            self.assertEqual(routed[0]["route_mode"], "suspected_high_overlap")

            high_overlap = read_json(output_dir / "high_overlap_candidates.json")
            self.assertEqual(len(high_overlap), 1)
            segment = high_overlap[0]
            self.assertEqual(segment["processing_path"], "high_overlap_candidate")
            self.assertEqual(segment["source"], "baseline_preserved")
            self.assertEqual(segment["resolution_mode"], "baseline_preserved")
            self.assertTrue(segment["text"].startswith("[mock transcript"))
            self.assertEqual(len(segment["candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
