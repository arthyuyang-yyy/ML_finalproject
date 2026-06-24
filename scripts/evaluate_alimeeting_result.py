"""Compare one pipeline output directory against AliMeeting TextGrid labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prefill_annotation_csv import read_meeting_utterances  # noqa: E402
from scripts.prepare_alimeeting import overlap_regions_from_turns  # noqa: E402

PUNCTUATION_RE = re.compile(r'[\s，。！？、；：,.!?;:"“”‘’（）()《》\[\]【】\-—…]')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="pipeline output directory")
    parser.add_argument("--textgrid-dir", required=True, type=Path, help="AliMeeting near-field textgrid_dir")
    parser.add_argument("--meeting-id", required=True, help="AliMeeting meeting id, e.g. R8001_M8004")
    parser.add_argument("--source-speaker", help="reference speaker token, e.g. N_SPK8013")
    parser.add_argument("--window-start", required=True, type=float, help="start time in the original meeting")
    parser.add_argument("--window-end", required=True, type=float, help="end time in the original meeting")
    parser.add_argument(
        "--hyp-time-offset",
        type=float,
        default=None,
        help="seconds to add to pipeline segment timestamps; defaults to --window-start",
    )
    parser.add_argument("--overlap-threshold", type=float, default=0.4)
    args = parser.parse_args(argv)

    hyp_offset = args.window_start if args.hyp_time_offset is None else args.hyp_time_offset
    evidence = _read_json(args.output_dir / "evidence_segments.json")
    events = _read_json(args.output_dir / "meeting_events.json")
    by_meeting = read_meeting_utterances(args.textgrid_dir)
    if args.meeting_id not in by_meeting:
        raise SystemExit(f"meeting_id not found in TextGrid directory: {args.meeting_id}")

    all_reference = _clip_utterances(by_meeting[args.meeting_id], args.window_start, args.window_end)
    if args.source_speaker:
        asr_reference = [utt for utt in all_reference if utt["speaker"] == args.source_speaker]
    else:
        asr_reference = all_reference
    if not asr_reference:
        raise SystemExit("no reference utterances found for the requested window/source speaker")

    ref_text = "".join(str(utt["text"]) for utt in asr_reference)
    hyp_text = "".join(str(segment.get("text", "")) for segment in evidence)
    ref_norm = _normalize_text(ref_text)
    hyp_norm = _normalize_text(hyp_text)
    distance, substitutions, insertions, deletions = _edit_distance(ref_norm, hyp_norm)
    cer = distance / len(ref_norm) if ref_norm else 0.0

    gt_overlap_regions = _clip_regions(
        overlap_regions_from_turns(all_reference),
        args.window_start,
        args.window_end,
    )
    hyp_overlap_regions = [
        (
            float(segment["start_time"]) + hyp_offset,
            float(segment["end_time"]) + hyp_offset,
        )
        for segment in evidence
        if float(segment.get("overlap_score", 0.0)) >= args.overlap_threshold
    ]
    gt_overlap_seconds = _total_region_seconds(gt_overlap_regions)
    overlap_recall = (
        _covered_seconds_by_regions(gt_overlap_regions, hyp_overlap_regions) / gt_overlap_seconds
        if gt_overlap_seconds
        else None
    )

    speaker_report = _speaker_report(asr_reference, evidence, hyp_offset)
    payload = {
        "output_dir": str(args.output_dir),
        "meeting_id": args.meeting_id,
        "window": [args.window_start, args.window_end],
        "source_speaker": args.source_speaker,
        "asr": {
            "reference_text": ref_text,
            "hypothesis_text": hyp_text,
            "normalized_reference_length": len(ref_norm),
            "cer": round(cer, 4),
            "edit_distance": distance,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
        },
        "speaker": speaker_report,
        "overlap": {
            "ground_truth_regions": gt_overlap_regions,
            "hypothesis_regions": hyp_overlap_regions,
            "ground_truth_overlap_seconds": round(gt_overlap_seconds, 3),
            "recall": None if overlap_recall is None else round(overlap_recall, 4),
        },
        "reference_utterances": asr_reference,
        "all_reference_utterances": all_reference,
        "hypothesis_segments": [
            {
                **segment,
                "absolute_start_time": round(float(segment["start_time"]) + hyp_offset, 3),
                "absolute_end_time": round(float(segment["end_time"]) + hyp_offset, 3),
            }
            for segment in evidence
        ],
        "meeting_summary": events.get("meeting_summary", "") if isinstance(events, dict) else "",
        "events": events.get("events", []) if isinstance(events, dict) else [],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_utterances(utterances: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for utt in utterances:
        utt_start = float(utt["start_time"])
        utt_end = float(utt["end_time"])
        if utt_end <= start or utt_start >= end:
            continue
        clipped.append(dict(utt))
    return sorted(clipped, key=lambda item: (item["start_time"], item["end_time"], item["speaker"]))


def _normalize_text(text: str) -> str:
    return PUNCTUATION_RE.sub("", text.lower())


def _edit_distance(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                choices = [(dp[i - 1][j - 1], "M")]
            else:
                choices = [(dp[i - 1][j - 1] + 1, "S")]
            choices.extend([(dp[i - 1][j] + 1, "D"), (dp[i][j - 1] + 1, "I")])
            dp[i][j], op[i][j] = min(choices, key=lambda item: item[0])

    i, j = n, m
    substitutions = insertions = deletions = 0
    while i > 0 or j > 0:
        action = op[i][j]
        if action == "M":
            i -= 1
            j -= 1
        elif action == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif action == "D":
            deletions += 1
            i -= 1
        elif action == "I":
            insertions += 1
            j -= 1
        else:
            break
    return dp[n][m], substitutions, insertions, deletions


def _speaker_report(
    reference: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    hyp_offset: float,
) -> dict[str, Any]:
    if not reference:
        return {"dominant_hypothesis_speaker": "", "known_speaker_coverage": 0.0}
    regions_by_speaker: dict[str, list[tuple[float, float]]] = {}
    for segment in evidence:
        speaker = str(segment.get("speaker", ""))
        start = float(segment["start_time"]) + hyp_offset
        end = float(segment["end_time"]) + hyp_offset
        regions_by_speaker.setdefault(speaker, []).append((start, end))

    ref_regions = [(float(utt["start_time"]), float(utt["end_time"])) for utt in reference]
    overlap_by_speaker = {
        speaker: _covered_seconds_by_regions(ref_regions, regions)
        for speaker, regions in regions_by_speaker.items()
    }
    dominant = max(overlap_by_speaker.items(), key=lambda item: item[1])[0] if overlap_by_speaker else ""
    total_ref_seconds = _total_region_seconds(ref_regions)
    known_seconds = sum(seconds for speaker, seconds in overlap_by_speaker.items() if speaker != "UNKNOWN")
    unknown_seconds = overlap_by_speaker.get("UNKNOWN", 0.0)
    return {
        "dominant_hypothesis_speaker": dominant,
        "known_speaker_coverage": round(known_seconds / total_ref_seconds, 4) if total_ref_seconds else 0.0,
        "unknown_speaker_coverage": round(unknown_seconds / total_ref_seconds, 4) if total_ref_seconds else 0.0,
        "overlap_seconds_by_hypothesis_speaker": {
            speaker: round(seconds, 3) for speaker, seconds in sorted(overlap_by_speaker.items())
        },
    }


def _clip_regions(regions: list[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    clipped = []
    for region_start, region_end in regions:
        lo = max(start, region_start)
        hi = min(end, region_end)
        if hi > lo:
            clipped.append((round(lo, 3), round(hi, 3)))
    return clipped


def _total_region_seconds(regions: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in regions)


def _covered_seconds_by_regions(
    targets: list[tuple[float, float]],
    covers: list[tuple[float, float]],
) -> float:
    total = 0.0
    for target_start, target_end in targets:
        for cover_start, cover_end in covers:
            total += max(0.0, min(target_end, cover_end) - max(target_start, cover_start))
    return total


if __name__ == "__main__":
    raise SystemExit(main())
