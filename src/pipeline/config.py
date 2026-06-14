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
    # Keep library/test runs deterministic. CLI and UI explicitly select the
    # primary real baseline, faster-whisper.
    low_overlap_asr_model: str = "mock"
    faster_whisper_model_size: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    enable_denoise: bool = False
    denoise_strength: float = 0.5
    speech_separation_backend: str = "none"
    sepformer_model_source: str = "speechbrain/sepformer-whamr16k"
    speech_separation_device: str = "cpu"
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
