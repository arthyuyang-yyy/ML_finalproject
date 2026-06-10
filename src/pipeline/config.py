"""Pipeline configuration defaults."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for one meeting pipeline run."""

    outputs_root: Path = Path("outputs")
    target_sample_rate: int = 16000
    overlap_threshold: float = 0.5
    language: str = "und"

    def meeting_dir(self, meeting_id: str) -> Path:
        """Return the per-meeting output directory."""
        return self.outputs_root / meeting_id
