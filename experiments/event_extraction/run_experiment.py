"""Rule-based structured-event extraction baseline experiment (Step 16a).

Loads an annotated meeting (see ``annotations.json``), runs the deterministic
rule-based extractor over the evidence segments, scores the predicted events
against the gold event labels with per-type precision/recall/F1, and writes a
results table.

Usage::

    python experiments/event_extraction/run_experiment.py
    python experiments/event_extraction/run_experiment.py --annotations path.json --out-dir dir

The bundled annotation set is a small hand-authored seed; replace or extend it
with real human annotations before reporting final paper numbers. This baseline
is the rule arm that Step 16b compares against plain and evidence-constrained
LLM extraction. The metric implementation lives in ``src/evaluation/core.py``.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_event_extraction  # noqa: E402
from src.events import extract_rule_based_events  # noqa: E402

HERE = Path(__file__).resolve().parent


def run(annotations_path: Path) -> dict:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    evidence_segments = data["evidence_segments"]
    gold_events = data["gold_events"]
    document = extract_rule_based_events(evidence_segments)
    metrics = evaluate_event_extraction(document["events"], gold_events)
    return {"document": document, "metrics": metrics}


def _render_markdown(metrics: dict) -> str:
    lines = [
        "# Rule-based Event Extraction Baseline (Step 16a)",
        "",
        f"- predicted events: {metrics['support_predicted']}",
        f"- gold events: {metrics['support_gold']}",
        f"- matched: {metrics['matched']}",
        f"- overall precision/recall/F1: "
        f"{metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f}",
        "",
        "| event_type | precision | recall | f1 |",
        "| --- | --- | --- | --- |",
    ]
    for event_type, scores in metrics["per_type"].items():
        lines.append(
            f"| {event_type} | {scores['precision']:.3f} | "
            f"{scores['recall']:.3f} | {scores['f1']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(HERE / "annotations.json"))
    parser.add_argument("--out-dir", default=str(HERE))
    args = parser.parse_args()

    result = run(Path(args.annotations))
    metrics = result["metrics"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "results.md").write_text(_render_markdown(metrics), encoding="utf-8")
    print(_render_markdown(metrics))


if __name__ == "__main__":
    main()
