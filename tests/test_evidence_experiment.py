"""Tests for the formal evidence-evaluation experiment runner.

Run with::

    python -m unittest tests.test_evidence_experiment
"""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.evidence_eval.run_experiment import format_markdown, run

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = REPO_ROOT / "experiments" / "evidence_eval" / "annotations.json"


class EvidenceExperimentTests(unittest.TestCase):
    def test_run_produces_all_metric_groups(self) -> None:
        results = run(ANNOTATIONS)
        self.assertIn("evidence_support", results)
        self.assertIn("uncertainty_preservation", results)
        self.assertIn("candidate_usefulness", results)

    def test_seed_dataset_values(self) -> None:
        results = run(ANNOTATIONS)
        evidence = results["evidence_support"]
        self.assertEqual(evidence["support"], 7)
        self.assertEqual(evidence["claims"], 6)
        self.assertAlmostEqual(evidence["evidence_recall"], 0.8)
        # Only the ev_999 citation is outside the source universe.
        self.assertAlmostEqual(evidence["hallucination_rate"], 1 / 6)

        uncertainty = results["uncertainty_preservation"]
        self.assertEqual(uncertainty["uncertain_total"], 4)
        self.assertAlmostEqual(uncertainty["collapse_rate"], 0.5)

        candidate = results["candidate_usefulness"]
        self.assertEqual(candidate["evaluated"], 2)
        self.assertGreater(candidate["wer_reduction"], 0.0)

    def test_markdown_report_has_tables(self) -> None:
        report = format_markdown(run(ANNOTATIONS))
        self.assertIn("Evidence support", report)
        self.assertIn("hallucination_rate", report)
        self.assertIn("| Metric | Value |", report)

    def test_runner_writes_results_files(self) -> None:
        results = run(ANNOTATIONS)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
            (out / "results.md").write_text(format_markdown(results), encoding="utf-8")
            self.assertTrue((out / "results.json").exists())
            self.assertTrue((out / "results.md").exists())


if __name__ == "__main__":
    unittest.main()
