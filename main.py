"""Command-line entry point for the end-to-end meeting pipeline."""

import argparse

from src.pipeline import run_meeting_pipeline


def main() -> None:
    """Run the pipeline for one audio file."""
    parser = argparse.ArgumentParser(description="Run the meeting-memory audio pipeline.")
    parser.add_argument("input_audio_path", help="Path to the source meeting audio file.")
    parser.add_argument("--meeting-id", default="meeting_001", help="Stable meeting ID for output grouping.")
    args = parser.parse_args()
    result = run_meeting_pipeline(args.input_audio_path, args.meeting_id)
    print(f"Pipeline complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
