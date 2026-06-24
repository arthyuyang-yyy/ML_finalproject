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
    suspected_overlap_threshold: float = 0.3
    language: str = "und"
    # Keep library/test runs deterministic. CLI and UI can select "auto", which
    # currently prefers FunASR for Chinese meetings when available.
    low_overlap_asr_model: str = "mock"
    vad_max_segment_s: float = 30.0
    vad_speech_pad_ms: int = 400
    vad_min_silence_ms: int = 500
    asr_context_padding_s: float = 0.2
    high_overlap_min_segment_s: float = 2.0
    high_overlap_decode_context_s: float = 2.0
    suspected_overlap_min_confidence_gain: float = 0.15
    suspected_overlap_max_text_cer: float = 0.35
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
    gemma_base_url: str | None = None
    memory_root: Path | None = None

    def meeting_dir(self, meeting_id: str) -> Path:
        """Return the per-meeting output directory."""
        return self.outputs_root / meeting_id

    def episodic_memory_path(self) -> Path:
        """Return the long-term memory path associated with this run."""
        root = self.memory_root or (self.outputs_root.parent / "memory")
        return root / "episodic_memory.json"
