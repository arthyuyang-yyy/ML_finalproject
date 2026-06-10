"""Pipeline configuration defaults."""

from dataclasses import dataclass
from pathlib import Path

from src.overlap_detector import DEFAULT_OVERLAP_THRESHOLD


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for one meeting pipeline run."""

    outputs_root: Path = Path("outputs")
    target_sample_rate: int = 16000
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    language: str = "und"
    low_overlap_asr_model: str = "mock"

    def meeting_dir(self, meeting_id: str) -> Path:
        """Return the per-meeting output directory."""
        return self.outputs_root / meeting_id
