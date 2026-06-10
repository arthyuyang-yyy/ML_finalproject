"""Input/output helpers for per-meeting pipeline artifacts."""

import json
from pathlib import Path
from typing import Any


def ensure_meeting_dirs(meeting_dir: str | Path) -> dict[str, Path]:
    """Create and return the standard output paths for one meeting."""
    base = Path(meeting_dir)
    clips = base / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "clips": clips,
        "preprocessed": base / "preprocessed.wav",
        "vad_segments": base / "vad_segments.json",
        "overlap": base / "overlap.json",
        "low_overlap_segments": base / "low_overlap_segments.json",
        "high_overlap_candidates": base / "high_overlap_candidates.json",
        "evidence_segments": base / "evidence_segments.json",
        "meeting_events": base / "meeting_events.json",
        "episodic_memory": base / "episodic_memory.json",
    }


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
