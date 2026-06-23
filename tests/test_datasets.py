"""Tests for local dataset manifests, discovery, and resumable execution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_dataset import run_manifest
from src.datasets.discovery import discover_audio_records
from src.datasets.manifest import read_manifest, write_manifest
from src.datasets.schema import expand_local_path, validate_dataset_record
from src.datasets.text_benchmarks import prepare_qmsum, prepare_vcsum


def _record(audio_path: Path, meeting_id: str = "meeting_001") -> dict:
    return {
        "dataset": "testset",
        "split": "eval",
        "meeting_id": meeting_id,
        "audio_path": str(audio_path),
        "language": "zh",
    }


class DatasetManifestTests(unittest.TestCase):
    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            audio.touch()
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, [_record(audio)])
            self.assertEqual(read_manifest(manifest, require_files=True)[0]["meeting_id"], "meeting_001")

    def test_manifest_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "meeting.wav"
            audio.touch()
            with self.assertRaisesRegex(ValueError, "unique meeting_id"):
                write_manifest(Path(tmp) / "manifest.jsonl", [_record(audio), _record(audio)])

    def test_record_rejects_unsafe_meeting_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "meeting_id"):
            validate_dataset_record(_record(Path("meeting.wav"), "../escape"))

    def test_expand_path_rejects_unresolved_environment_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "unresolved environment variable"):
            expand_local_path("${MISSING_MEETING_DATA_ROOT}/meeting.wav")

    def test_discovery_finds_audio_and_sibling_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "session one" / "meeting.wav"
            audio.parent.mkdir()
            audio.touch()
            annotation = audio.with_suffix(".rttm")
            annotation.touch()
            records = discover_audio_records(root, dataset="ami", split="eval", language="en")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["meeting_id"], "session_one__meeting")
            self.assertEqual(Path(records[0]["annotation_path"]), annotation.resolve())

    def test_discovery_matches_alimeeting_separate_annotation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_root = root / "audio"
            annotation_root = root / "annotations"
            audio_root.mkdir()
            annotation_root.mkdir()
            audio = audio_root / "R8001_M8004_MS801.wav"
            annotation = annotation_root / "R8001_M8004.TextGrid"
            audio.touch()
            annotation.touch()
            records = discover_audio_records(
                audio_root,
                dataset="alimeeting",
                split="eval",
                language="zh",
                annotation_root=annotation_root,
            )
            self.assertEqual(Path(records[0]["annotation_path"]), annotation.resolve())


class DatasetRunnerTests(unittest.TestCase):
    @patch("scripts.run_dataset.run_meeting_pipeline")
    def test_runner_isolates_outputs_and_resumes_completed_meetings(self, mocked_run) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            audio.touch()
            manifest = root / "manifest.jsonl"
            output_root = root / "experiment"
            write_manifest(manifest, [_record(audio)])
            mocked_run.return_value = {
                "output_dir": str(output_root / "outputs" / "meeting_001"),
                "num_evidence_segments": 2,
            }

            first = run_manifest(manifest, output_root)
            second = run_manifest(manifest, output_root)

            self.assertEqual(first, {"completed": 1, "failed": 0, "skipped": 0})
            self.assertEqual(second, {"completed": 0, "failed": 0, "skipped": 1})
            self.assertEqual(mocked_run.call_count, 1)
            config = mocked_run.call_args.kwargs["config"]
            self.assertEqual(config.outputs_root, output_root.resolve() / "outputs")
            self.assertEqual(config.memory_root, output_root.resolve() / "memory")
            status = json.loads((output_root / "status" / "meeting_001.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")

    @patch("scripts.run_dataset.run_meeting_pipeline", side_effect=RuntimeError("backend failed"))
    def test_runner_records_failure_and_requires_explicit_retry(self, mocked_run) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            audio.touch()
            manifest = root / "manifest.jsonl"
            output_root = root / "experiment"
            write_manifest(manifest, [_record(audio)])

            failed = run_manifest(manifest, output_root)
            skipped = run_manifest(manifest, output_root)
            retried = run_manifest(manifest, output_root, retry_failed=True)

            self.assertEqual(failed["failed"], 1)
            self.assertEqual(skipped["skipped"], 1)
            self.assertEqual(retried["failed"], 1)
            self.assertEqual(mocked_run.call_count, 2)
            status = json.loads((output_root / "status" / "meeting_001.json").read_text(encoding="utf-8"))
            self.assertIn("RuntimeError: backend failed", status["error"])


class TextBenchmarkTests(unittest.TestCase):
    def test_prepare_qmsum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data" / "ALL" / "test"
            source.mkdir(parents=True)
            (source / "meeting.json").write_text(json.dumps({
                "meeting_transcripts": [{"speaker": "A", "content": "Hello"}],
                "general_query_list": [{"query": "Summary?", "answer": "Hello."}],
                "specific_query_list": [{
                    "query": "Who spoke?",
                    "answer": "A.",
                    "relevant_text_span": [["0", "0"]],
                }],
                "topic_list": [],
            }), encoding="utf-8")
            output = root / "qmsum.jsonl"
            self.assertEqual(prepare_qmsum(root, output), 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["meeting_id"], "meeting")
            self.assertEqual(len(record["queries"]), 2)

    def test_prepare_vcsum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "vcsum_data"
            source.mkdir()
            (source / "long_test.txt").write_text(json.dumps({
                "id": "1",
                "av_num": 10,
                "context": [["你好"]],
                "speaker": [1],
                "summary": "摘要",
                "eos_index": [0],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            output = root / "vcsum.jsonl"
            self.assertEqual(prepare_vcsum(root, output), 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["reference_summary"], "摘要")


if __name__ == "__main__":
    unittest.main()
