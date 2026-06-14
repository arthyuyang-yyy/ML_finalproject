"""Command-line entry point for the end-to-end meeting pipeline."""

import argparse

from src.pipeline import run_meeting_pipeline
from src.pipeline.config import PipelineConfig


def main() -> None:
    """Run the pipeline for one audio file."""
    parser = argparse.ArgumentParser(description="Run the meeting-memory audio pipeline.")
    parser.add_argument("input_audio_path", help="Path to the source meeting audio file.")
    parser.add_argument("--meeting-id", default="meeting_001", help="Stable meeting ID for output grouping.")
    parser.add_argument("--asr", default="faster-whisper", choices=["auto", "whisperx", "faster-whisper", "whisper", "funasr", "mock"])
    parser.add_argument("--faster-whisper-model", default="small", help="Model size/name for the faster-whisper baseline.")
    parser.add_argument("--asr-device", default="cpu", help="Device for faster-whisper, for example cpu or cuda.")
    parser.add_argument("--asr-compute-type", default="int8", help="faster-whisper compute type, for example int8 or float16.")
    parser.add_argument("--denoise", action="store_true", help="Enable optional stationary-noise reduction before ASR.")
    parser.add_argument("--denoise-strength", type=float, default=0.5, help="Denoise strength in [0, 1].")
    parser.add_argument("--speech-separation", default="none", choices=["none", "sepformer"], help="Optional high-overlap speech-separation backend.")
    parser.add_argument("--sepformer-model", default="speechbrain/sepformer-whamr16k", help="SpeechBrain SepFormer model source.")
    parser.add_argument("--speech-separation-device", default="cpu", help="Device used by the speech-separation model.")
    parser.add_argument("--language", default="und")
    parser.add_argument("--gemma-backend", default="none", choices=["none", "ollama"])
    parser.add_argument("--gemma-model", default="gemma3:4b")
    parser.add_argument("--gemma-base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    config = PipelineConfig(
        low_overlap_asr_model=args.asr,
        faster_whisper_model_size=args.faster_whisper_model,
        faster_whisper_device=args.asr_device,
        faster_whisper_compute_type=args.asr_compute_type,
        enable_denoise=args.denoise,
        denoise_strength=args.denoise_strength,
        speech_separation_backend=args.speech_separation,
        sepformer_model_source=args.sepformer_model,
        speech_separation_device=args.speech_separation_device,
        language=args.language,
        gemma_backend=args.gemma_backend,
        gemma_model=args.gemma_model,
        gemma_base_url=args.gemma_base_url,
    )
    result = run_meeting_pipeline(args.input_audio_path, args.meeting_id, config=config)
    print(f"Pipeline complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
