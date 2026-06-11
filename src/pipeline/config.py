"""Pipeline configuration defaults."""

from dataclasses import dataclass
from pathlib import Path

from src.overlap.detector import DEFAULT_OVERLAP_THRESHOLD


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for one meeting pipeline run."""

    outputs_root: Path = Path("outputs")
    target_sample_rate: int = 16000
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    language: str = "und"
    low_overlap_asr_model: str = "auto"
    gemma_backend: str = "none"
    gemma_model: str = "gemma3:4b"
    gemma_base_url: str = "http://127.0.0.1:11434"
    memory_root: Path | None = None

    def meeting_dir(self, meeting_id: str) -> Path:
        """Return the per-meeting output directory."""
        return self.outputs_root / meeting_id

    def episodic_memory_path(self) -> Path:
        """Return the long-term memory path associated with this run."""
        root = self.memory_root or (self.outputs_root.parent / "memory")
        return root / "episodic_memory.json"
