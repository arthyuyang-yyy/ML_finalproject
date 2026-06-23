# Strict 4-Minute Full-Flow Tuning

## Setup

- Source audio: `/Users/lymn/MLData/meeting-memory/datasets/alimeeting/eval/Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav`
- Window: `1110.0s` to `1350.0s`
- Clip: `outputs/experiments/fullflow_tuning/clips/R8001_M8004_1110_1350_4min.wav`
- Reference labels: AliMeeting near-field TextGrid directory
- Strict rule: reject any run that uses non-pyannote overlap detection, mock ASR, fallback high-overlap candidates, fallback resolver output, or deterministic fallback event extraction.

## Runs

| Run | Status | Key Params | CER | Overlap Recall | Known Speaker Coverage | High Path Ratio | Events | Score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_001_baseline` | failed | `min_high=1.0`, `suspected=0.2`, `overlap=0.4`, `padding=0.2` | - | - | - | - | - | - |
| `run_011_strict_high_min2` | completed | `min_high=2.0`, `suspected=0.3`, `overlap=0.4`, `padding=0.2` | 0.5080 | 0.3524 | 0.9553 | 0.3636 | 15 | 0.5021 |
| `run_013_strict_pad_0p1` | completed | `min_high=2.0`, `suspected=0.3`, `overlap=0.4`, `padding=0.1` | 0.5121 | 0.3524 | 0.9553 | 0.3636 | 15 | 0.5003 |
| `run_014_strict_pad_0p5` | completed | `min_high=2.0`, `suspected=0.3`, `overlap=0.4`, `padding=0.5` | 0.5126 | 0.3524 | 0.9553 | 0.3636 | 13 | 0.5000 |
| `run_015_strict_route_precision` | completed | `min_high=2.0`, `suspected=0.3`, `overlap=0.5`, `padding=0.2` | 0.5075 | 0.3524 | 0.9553 | 0.3636 | 18 | 0.5023 |

## Selected Config

Use `run_015_strict_route_precision` for this 4-minute tuning window:

```text
vad_max_segment_s = 30.0
vad_speech_pad_ms = 400
vad_min_silence_ms = 500
asr_context_padding_s = 0.2
overlap_threshold = 0.5
suspected_overlap_threshold = 0.3
high_overlap_min_segment_s = 2.0
suspected_overlap_min_confidence_gain = 0.15
suspected_overlap_max_text_cer = 0.35
enable_denoise = false
speech_separation_backend = none
low_overlap_asr_model = funasr
high_overlap_asr_model = faster-whisper small/cpu/int8
gemma_backend = deepseek
```

## Follow-up Optimization Status

- The long-context ASR rewrite is intentionally not implemented.
- Short pyannote/provided overlap regions now remain in the high-overlap candidate path and decode only with local context.
- Added `high_overlap_decode_context_s = 2.0` as the local context window for short authoritative overlaps.
- High-overlap candidate decoding now uses a stricter Chinese-focused beam set, disables previous-text conditioning, and records the decode window in candidate metadata.
- The experiment runner now reuses completed `run_result.json` files by default, adds `run_018_short_overlap_context`, and supports a second validation window with `--window-preset second`.
- Retest result for `run_018_short_overlap_context`: completed strict mode, CER `0.5266`, overlap recall `0.4224`, known speaker coverage `0.9480`, high path ratio `0.4894`, events `9`, score `0.4914`.

## Notes

- The original baseline failed strict mode because three short high-overlap regions produced no usable faster-whisper candidates and would have fallen back to placeholder candidates.
- Raising `high_overlap_min_segment_s` to `2.0` and `suspected_overlap_threshold` to `0.3` removed those invalid high-overlap candidate cases.
- Padding `0.2s` was better than `0.1s` and `0.5s` on CER for this window.
- `overlap_threshold=0.5` was marginally best among completed strict runs, but the difference from `0.4` is small; verify on a second 4-minute window before treating it as a global setting.
