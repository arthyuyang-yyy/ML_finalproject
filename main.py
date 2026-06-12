"""Command-line entry point for the end-to-end meeting pipeline."""

import argparse

from src.pipeline import run_meeting_pipeline
from src.pipeline.config import PipelineConfig


def main() -> None:
    """Run the pipeline for one audio file."""
    parser = argparse.ArgumentParser(description="Run the meeting-memory audio pipeline.")
    parser.add_argument("input_audio_path", help="Path to the source meeting audio file.")
    parser.add_argument("--meeting-id", default="meeting_001", help="Stable meeting ID for output grouping.")
    parser.add_argument("--asr", default="auto", choices=["auto", "whisperx", "faster-whisper", "whisper", "funasr", "mock"])
    parser.add_argument("--language", default="und")
    parser.add_argument("--gemma-backend", default="none", choices=["none", "ollama"])
    parser.add_argument("--gemma-model", default="gemma3:4b")
    parser.add_argument("--gemma-base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    config = PipelineConfig(
        low_overlap_asr_model=args.asr,
        language=args.language,
        gemma_backend=args.gemma_backend,
        gemma_model=args.gemma_model,
        gemma_base_url=args.gemma_base_url,
    )
    result = run_meeting_pipeline(args.input_audio_path, args.meeting_id, config=config)
    print(f"Pipeline complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
