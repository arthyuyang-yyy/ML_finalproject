"""Command-line entry point for the end-to-end meeting pipeline."""

import argparse
import logging
import sys

from src.pipeline import run_meeting_pipeline
from src.pipeline.config import PipelineConfig
from src.utils import load_dotenv


def _configure_logging() -> None:
    """Stream pipeline stage progress to stderr at INFO level.

    Runs under ``nohup`` need the log line-buffered so the matrix runner can
    tail the log while the process is still running.
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger("meeting_memory")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False


def main() -> None:
    load_dotenv()
    _configure_logging()
    """Run the pipeline for one audio file."""
    parser = argparse.ArgumentParser(description="Run the meeting-memory audio pipeline.")
    parser.add_argument("input_audio_path", help="Path to the source meeting audio file.")
    parser.add_argument("--meeting-id", default="meeting_001", help="Stable meeting ID for output grouping.")
    parser.add_argument("--asr", default="auto", choices=["auto", "whisperx", "faster-whisper", "whisper", "funasr", "mock"])
    parser.add_argument("--vad-max-segment-s", type=float, default=30.0, help="Maximum VAD speech segment duration before splitting.")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=400, help="VAD speech padding in milliseconds.")
    parser.add_argument("--vad-min-silence-ms", type=int, default=500, help="VAD minimum silence duration in milliseconds.")
    parser.add_argument("--asr-context-padding-s", type=float, default=0.2, help="Extra context seconds read around each ASR segment.")
    parser.add_argument("--high-overlap-min-segment-s", type=float, default=2.0, help="Minimum authoritative overlap duration routed to the high-overlap ASR path.")
    parser.add_argument("--high-overlap-decode-context-s", type=float, default=2.0, help="Extra local context seconds decoded around short authoritative overlap regions.")
    parser.add_argument("--suspected-overlap-threshold", type=float, default=0.3, help="Lower threshold for recall-oriented suspected high-overlap routing.")
    parser.add_argument("--suspected-overlap-min-confidence-gain", type=float, default=0.15, help="Minimum candidate confidence gain needed to replace baseline text for suspected overlap.")
    parser.add_argument("--suspected-overlap-max-text-cer", type=float, default=0.35, help="Maximum candidate-vs-baseline character error ratio allowed before preserving baseline text.")
    parser.add_argument("--faster-whisper-model", default="small", help="Model size/name for the faster-whisper baseline.")
    parser.add_argument("--asr-device", default="cpu", help="Device for faster-whisper, for example cpu or cuda.")
    parser.add_argument("--asr-compute-type", default="int8", help="faster-whisper compute type, for example int8 or float16.")
    parser.add_argument("--denoise", action="store_true", help="Enable optional stationary-noise reduction before ASR.")
    parser.add_argument("--denoise-strength", type=float, default=0.5, help="Denoise strength in [0, 1].")
    parser.add_argument("--speech-separation", default="none", choices=["none", "nmf", "sepformer"], help="Optional high-overlap speech-separation backend (nmf is dependency-free; sepformer needs SpeechBrain).")
    parser.add_argument("--sepformer-model", default="speechbrain/sepformer-whamr16k", help="SpeechBrain SepFormer model source.")
    parser.add_argument("--speech-separation-device", default="cpu", help="Device used by the speech-separation model.")
    parser.add_argument("--language", default="und")
    parser.add_argument("--gemma-backend", default="none", choices=["none", "ollama", "openai", "deepseek", "transformers"])
    parser.add_argument("--gemma-model", default="gemma3:4b")
    parser.add_argument("--gemma-base-url", default=None)
    args = parser.parse_args()
    config = PipelineConfig(
        low_overlap_asr_model=args.asr,
        vad_max_segment_s=args.vad_max_segment_s,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        asr_context_padding_s=args.asr_context_padding_s,
        high_overlap_min_segment_s=args.high_overlap_min_segment_s,
        high_overlap_decode_context_s=args.high_overlap_decode_context_s,
        suspected_overlap_threshold=args.suspected_overlap_threshold,
        suspected_overlap_min_confidence_gain=args.suspected_overlap_min_confidence_gain,
        suspected_overlap_max_text_cer=args.suspected_overlap_max_text_cer,
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
