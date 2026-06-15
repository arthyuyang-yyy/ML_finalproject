"""Formal evidence-support, hallucination, uncertainty, and candidate experiment.

Loads a manually-annotated evaluation set (see ``annotations.json``), runs the
implemented evaluation metrics over the paired predictions and gold references,
and writes a results table.

Usage::

    python experiments/evidence_eval/run_experiment.py
    python experiments/evidence_eval/run_experiment.py --annotations path.json --out-dir dir

The bundled annotation set is a small hand-authored seed; replace or extend it
with real human annotations of pipeline/QA outputs before reporting final paper
numbers. The metric implementations live in ``src/evaluation/core.py``.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (  # noqa: E402
    evaluate_candidate_usefulness,
    evaluate_evidence_support,
    evaluate_uncertainty_preservation,
)

DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent / "annotations.json"


def run(annotations_path: str | Path = DEFAULT_ANNOTATIONS) -> dict:
    """Compute all evidence-related metrics for one annotation file."""
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))

    qa_items = data.get("qa_items", [])
    predictions = [item["prediction"] for item in qa_items]
    references = [item["reference"] for item in qa_items]
    source_evidence_ids = data.get("source_evidence_ids")
    evidence_text_map = data.get("evidence_texts")

    candidate_segments = data.get("candidate_segments", [])
    segments = [item["segment"] for item in candidate_segments]
    candidate_refs = [item["reference"] for item in candidate_segments]

    results: dict = {}
    if predictions:
        results["evidence_support"] = evaluate_evidence_support(
            predictions,
            references,
            source_evidence_ids=source_evidence_ids,
            evidence_text_map=evidence_text_map,
        )
        results["uncertainty_preservation"] = evaluate_uncertainty_preservation(
            predictions, references
        )
    if segments:
        results["candidate_usefulness"] = evaluate_candidate_usefulness(
            segments, candidate_refs
        )
    return results


def format_markdown(results: dict) -> str:
    """Render the results as a Markdown report with one table per metric group."""
    titles = {
        "evidence_support": "Evidence support & hallucination",
        "uncertainty_preservation": "Uncertainty preservation",
        "candidate_usefulness": "Candidate usefulness",
    }
    lines = ["# Evidence Evaluation Results", ""]
    for key, title in titles.items():
        group = results.get(key)
        if not group:
            continue
        lines += [f"## {title}", "", "| Metric | Value |", "| --- | --- |"]
        for name, value in group.items():
            shown = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {name} | {shown} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for results.json and results.md.",
    )
    args = parser.parse_args()

    results = run(args.annotations)
    report = format_markdown(results)
    print(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "results.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
