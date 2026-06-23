"""Overlap scoring and overlapped-speech detection adapters.

The preferred path uses pyannote's overlapped speech detection model and scores
each VAD segment by the fraction of its duration covered by detected overlap.
When pyannote or a Hugging Face token is unavailable, a conservative energy
fallback keeps the demo runnable without pretending to be a calibrated overlap
model.
"""

import os
from typing import Any

import numpy as np

from src.audio.preprocess import TARGET_SAMPLE_RATE, load_audio, silero_vad
from src.diarization.core import load_pyannote_pipeline
from src.errors import BackendExecutionError, BackendUnavailableError
from src.fallbacks.overlap import estimate_with_energy_fallback

DEFAULT_OVERLAP_THRESHOLD = 0.5
MIN_AUTHORITATIVE_OVERLAP_SECONDS = 0.2
PYANNOTE_OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation"
PYANNOTE_SEGMENTATION_REVISION = "Interspeech2021"

OverlapRegion = tuple[float, float]


def _move_model_to_accelerator(model: Any) -> None:
    """Move a pyannote model to CUDA/MPS when available, else keep CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            model.to(torch.device("mps"))
    except Exception:
        pass


def _resolve_segmentation_checkpoint() -> str | None:
    """Return the local HF cache snapshot for ``pyannote/segmentation`` when
    all required assets are present (config + weights + hparams).

    Returns ``None`` when the cache is incomplete or absent, in which case the
    caller should fall back to ``Model.from_pretrained(<model_id>)``.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None

    required = ("pytorch_model.bin", "hparams.yaml")
    snapshot_root: str | None = None
    for filename in required:
        cached = try_to_load_from_cache(
            PYANNOTE_SEGMENTATION_MODEL,
            filename,
            revision=PYANNOTE_SEGMENTATION_REVISION,
        )
        if cached is None or not isinstance(cached, str):
            return None
        if not os.path.isfile(cached):
            return None
        if snapshot_root is None:
            # trim "/pytorch_model.bin" (or "/hparams.yaml") from the end
            snapshot_root = os.path.dirname(cached)
    return snapshot_root


def estimate_segment_overlap_scores(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    sample_rate: int = TARGET_SAMPLE_RATE,
    audio_path: str | None = None,
    overlap_regions: list[OverlapRegion] | None = None,
    diarization_turns: list[dict[str, Any]] | None = None,
    asr_instability: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Attach ``overlap_score`` to each VAD segment.

    If explicit ``overlap_regions`` are supplied, or pyannote can detect them
    from ``audio_path``, the score is overlap-covered duration divided by the
    segment duration. Otherwise a conservative energy fallback produces a weak
    uncertainty signal while marking the detector source accordingly.
    """
    if not segments:
        return []

    regions = overlap_regions
    detector = "provided_regions" if regions is not None else "energy_fallback"
    if regions is None and audio_path:
        regions = detect_pyannote_overlap_regions(audio_path)
        if regions is not None:
            detector = "pyannote"

    if regions is not None:
        base = [
            {
                **segment,
                "overlap_score": _round_score(_overlap_fraction(segment, regions)),
                "overlap_seconds": round(_overlap_seconds(segment, regions), 3),
                "overlap_regions": _overlap_intersections(segment, regions),
                "overlap_detector": detector,
            }
            for segment in segments
        ]
        return _fuse_overlap_signals(base, diarization_turns or [], asr_instability or {})

    base = estimate_with_energy_fallback(samples, segments, sample_rate)
    return _fuse_overlap_signals(base, diarization_turns or [], asr_instability or {})


def detect_pyannote_overlap_regions(
    audio_path: str,
    model_name: str = PYANNOTE_OVERLAP_MODEL,
    auth_token: str | None = None,
) -> list[OverlapRegion] | None:
    """Return pyannote overlapped-speech regions when configured.

    A missing token disables this optional backend. Once configured, the
    pipeline is run on an in-memory waveform (loaded via the project's own
    :func:`load_audio`, bypassing pyannote.audio 4.x's torchcodec/FFmpeg file
    decoder). The ``pyannote/overlapped-speech-detection`` pipeline is not
    shipped with pyannote.audio 4.x (its ``OverlappedSpeechDetection`` pipeline
    class was removed); in that case the OSD sub-model is still cached and a
    direct :class:`Inference` over ``pyannote/segmentation`` is used with the
    pipeline's published onset/offset thresholds to recover overlap regions.

    Hard failures (missing dependency, model load error) are surfaced; a
    pipeline class that no longer exists falls back to the inference path
    rather than crashing the whole overlap detector.
    """
    token = auth_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        return None

    pipeline: Any = None
    try:
        pipeline = load_pyannote_pipeline(model_name, token)
    except ImportError as exc:
        raise BackendUnavailableError(
            "pyannote OSD is configured but pyannote.audio is unavailable"
        ) from exc
    except AttributeError:
        # pyannote.audio 4.x removed the OverlappedSpeechDetection pipeline
        # class, so the model can no longer be loaded as a pipeline. Leave
        # ``pipeline`` unset and recover overlap regions via a direct
        # Inference over the segmentation sub-model below.
        pass
    except (BackendUnavailableError, BackendExecutionError):
        raise
    except Exception as exc:
        raise BackendExecutionError(
            f"pyannote OSD failed for model '{model_name}'"
        ) from exc

    try:
        import torch

        from src.audio.preprocess import TARGET_SAMPLE_RATE, load_audio

        samples, sample_rate = load_audio(audio_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            from src.audio.preprocess import resample

            samples = resample(samples, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        file = {"waveform": waveform, "sample_rate": int(sample_rate)}

        if pipeline is None:
            output = _osd_inference_fallback(file, token)
        else:
            try:
                output = pipeline(file)
            except AttributeError:
                output = _osd_inference_fallback(file, token)
    except ImportError as exc:
        raise BackendUnavailableError(
            "pyannote OSD is configured but pyannote.audio is unavailable"
        ) from exc
    except (BackendUnavailableError, BackendExecutionError):
        raise
    except Exception:
        # Local model files are incomplete (hf-mirror does not serve
        # pyannote/segmentation weights) or the accelerator is busy.
        # Returning None lets the caller fall back to the energy-based
        # detector instead of aborting the whole pipeline.
        return None

    return _coerce_pyannote_regions(output)


def _osd_inference_fallback(file: dict[str, Any], token: str) -> Any:
    """Run overlap detection directly on ``pyannote/segmentation``.

    The ``pyannote/overlapped-speech-detection`` config pins the
    ``pyannote/segmentation`` model at the ``Interspeech2021`` revision and a
    pair of onset/offset thresholds. pyannote.audio 4.x dropped the
    ``OverlappedSpeechDetection`` pipeline class, so this helper loads the
    segmentation model with :class:`Inference`, takes the overlap probability
    track (channel index 1 of the segmentation output), and binarises it with
    the same thresholds to produce an :class:`Annotation` of overlap regions.
    """
    from pyannote.audio import Inference
    from pyannote.audio.core.model import Model
    from pyannote.core import Annotation, Segment

    # Prefer the local HF cache snapshot so we bypass ``hf_hub_download``'s
    # HEAD request — this stack runs with ``HF_HUB_OFFLINE=1`` and the
    # segmentation weights are only available on the upstream HF hub (not on
    # hf-mirror), so the HEAD would otherwise raise ``OfflineModeIsEnabled``.
    checkpoint = _resolve_segmentation_checkpoint()
    if checkpoint is None:
        checkpoint = PYANNOTE_SEGMENTATION_MODEL

    model = Model.from_pretrained(checkpoint, token=token)
    _move_model_to_accelerator(model)
    inference = Inference(model, batch_size=32, pre_aggregation_hook=lambda scores: scores)
    scores = inference(file)  # SlidingWindowFeature: (num_frames, num_classes)

    # Channel layout for pyannote/segmentation: [no-speech, overlap, speech].
    # The overlap track is index 1; robustly pick the overlap column when the
    # model exposes the standard 3-class layout, otherwise pick the argmax
    # against the speech column.
    data = scores.data
    if data.ndim == 3:
        data = data.reshape(data.shape[0], data.shape[-1])
    num_classes = data.shape[-1] if data.ndim == 2 else 1
    if num_classes >= 3:
        overlap_prob = data[:, 1]
    elif num_classes == 2:
        overlap_prob = data[:, 0]
    else:
        overlap_prob = data[:, 0]

    onset, offset = 0.81, 0.48  # from pyannote/overlapped-speech-detection config
    timeline = scores.sliding_window
    annotation = Annotation()
    active = False
    start = 0.0
    for i, prob in enumerate(overlap_prob):
        t = timeline[i].start
        if not active and prob >= onset:
            active = True
            start = t
        elif active and prob < offset:
            active = False
            if t > start:
                annotation[Segment(start, t)] = "OVERLAP"
    if active:
        end = float(timeline[len(overlap_prob) - 1].end)
        if end > start:
            annotation[Segment(start, end)] = "OVERLAP"
    return annotation


def _fuse_overlap_signals(
    segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
    asr_instability: dict[str, float],
) -> list[dict[str, Any]]:
    """Fuse OSD/energy, diarization overlap, speaker changes, and ASR instability."""
    fused: list[dict[str, Any]] = []
    for segment in segments:
        base = float(segment["overlap_score"])
        diarization = _diarization_overlap_fraction(segment, diarization_turns)
        speaker_change = _speaker_change_signal(segment, diarization_turns)
        instability = max(0.0, min(1.0, float(asr_instability.get(str(segment.get("segment_id", "")), 0.0))))
        available = bool(diarization_turns) or bool(asr_instability)
        score = max(base, 0.55 * base + 0.25 * diarization + 0.10 * speaker_change + 0.10 * instability) if available else base
        fused.append({
            **segment,
            "overlap_score": _round_score(score),
            "overlap_components": {
                "osd_or_energy": _round_score(base),
                "diarization_overlap": _round_score(diarization),
                "speaker_change": _round_score(speaker_change),
                "asr_instability": _round_score(instability),
            },
        })
    return fused


def _diarization_overlap_fraction(
    segment: dict[str, Any], diarization_turns: list[dict[str, Any]]
) -> float:
    """Approximate simultaneous multi-speaker coverage from diarization turns."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    duration = max(0.0, end - start)
    if duration == 0.0:
        return 0.0
    events: list[tuple[float, int]] = []
    for turn in diarization_turns:
        left = max(start, float(turn["start_time"]))
        right = min(end, float(turn["end_time"]))
        if right > left:
            events.extend([(left, 1), (right, -1)])
    active = 0
    previous = start
    overlap = 0.0
    for timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        if active >= 2:
            overlap += timestamp - previous
        active += delta
        previous = timestamp
    return min(1.0, overlap / duration)


def _speaker_change_signal(segment: dict[str, Any], diarization_turns: list[dict[str, Any]]) -> float:
    """Return a bounded signal for speaker transitions inside a segment."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    speakers = [
        str(turn["speaker"])
        for turn in sorted(diarization_turns, key=lambda item: float(item["start_time"]))
        if min(end, float(turn["end_time"])) > max(start, float(turn["start_time"]))
    ]
    changes = sum(left != right for left, right in zip(speakers, speakers[1:]))
    return min(1.0, changes / 2.0)


def estimate_overlap_score(audio_path: str) -> float:
    """Estimate the maximum per-segment overlap score in an audio file."""
    samples, sample_rate = load_audio(audio_path)
    regions = silero_vad(samples, sample_rate)
    segments = [
        {"segment_id": f"overlap_seg_{index + 1:03d}", "start_time": start, "end_time": end}
        for index, (start, end) in enumerate(regions)
    ]
    scored = estimate_segment_overlap_scores(samples, segments, sample_rate, audio_path=audio_path)
    if not scored:
        return 0.0
    return round(max(float(segment["overlap_score"]) for segment in scored), 3)


def detect_overlap_segments(audio_path: str, threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> list[dict]:
    """Return timestamped VAD segments routed as high-overlap candidates."""
    samples, sample_rate = load_audio(audio_path)
    regions = silero_vad(samples, sample_rate)
    segments = [
        {"segment_id": f"overlap_seg_{index + 1:03d}", "start_time": start, "end_time": end}
        for index, (start, end) in enumerate(regions)
    ]
    return [
        segment
        for segment in estimate_segment_overlap_scores(samples, segments, sample_rate, audio_path=audio_path)
        if float(segment["overlap_score"]) >= threshold
    ]


def _overlap_fraction(segment: dict[str, Any], overlap_regions: list[OverlapRegion]) -> float:
    """Compute how much of one VAD segment is covered by overlap regions."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    duration = max(0.0, end - start)
    if duration == 0.0:
        return 0.0

    return min(1.0, _overlap_seconds(segment, overlap_regions) / duration)


def _overlap_seconds(segment: dict[str, Any], overlap_regions: list[OverlapRegion]) -> float:
    """Compute seconds of one VAD segment covered by overlap regions."""
    return sum(end - start for start, end in _overlap_intersections(segment, overlap_regions))


def _overlap_intersections(segment: dict[str, Any], overlap_regions: list[OverlapRegion]) -> list[list[float]]:
    """Return overlap subregions clipped to one VAD segment."""
    start = float(segment["start_time"])
    end = float(segment["end_time"])

    intersections: list[list[float]] = []
    for region_start, region_end in _merge_regions(overlap_regions):
        intersection_start = max(start, region_start)
        intersection_end = min(end, region_end)
        if intersection_end > intersection_start:
            intersections.append([round(intersection_start, 3), round(intersection_end, 3)])
    return intersections


def _merge_regions(regions: list[OverlapRegion]) -> list[OverlapRegion]:
    """Merge overlapping or touching overlap regions."""
    if not regions:
        return []
    ordered = sorted((float(start), float(end)) for start, end in regions if end > start)
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _coerce_pyannote_regions(output: Any) -> list[OverlapRegion]:
    """Convert common pyannote outputs into ``[(start, end), ...]``."""
    regions: list[OverlapRegion] = []

    if hasattr(output, "itertracks"):
        for item in output.itertracks(yield_label=True):
            segment = item[0]
            regions.append((float(segment.start), float(segment.end)))
    elif hasattr(output, "get_timeline"):
        for segment in output.get_timeline():
            regions.append((float(segment.start), float(segment.end)))
    elif isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                regions.append((float(item["start"]), float(item["end"])))
            else:
                regions.append((float(item[0]), float(item[1])))

    return _merge_regions(regions)


def _round_score(value: float) -> float:
    """Clamp and round an overlap score."""
    return round(max(0.0, min(1.0, float(value))), 3)
