"""Automatic speech recognition adapters.

This module turns audio into transcript text with timestamps and a calibrated
``asr_confidence`` in ``[0, 1]``. The recogniser is intentionally pluggable: the
pipeline should never hard-code one engine, because Chinese meetings are usually
better served by Paraformer/FunASR while Whisper is stronger for multilingual or
English audio, and disagreement between two engines is itself a useful
uncertainty signal (see :mod:`src.candidates.generator`).

Heavy backends (``whisper``, ``whisperx``, ``funasr``) are imported lazily inside
each adapter so this module stays importable, and the unit tests run, without
any model download or GPU. :class:`MockASRAdapter` provides a dependency-free
recogniser for wiring and testing the rest of the pipeline before real models
are present.

The transcript-level result has the shape::

    {
        "text": str,             # full transcript
        "language": str,
        "model": str,            # adapter name, e.g. "whisper"
        "asr_confidence": float, # aggregate confidence in [0, 1]
        "segments": [
            {"start_time": float, "end_time": float,
             "text": str, "asr_confidence": float},
            ...
        ],
    }
"""

from typing import Any

import numpy as np

from src.audio.preprocess import TARGET_SAMPLE_RATE
from src.utils import validate_score


def logprob_to_confidence(avg_logprob: float, no_speech_prob: float = 0.0) -> float:
    """Map a Whisper-style average token log-probability to a ``[0, 1]`` score.

    Whisper reports ``avg_logprob`` (mean natural-log token probability, usually
    in ``[-1.5, 0]``) and ``no_speech_prob`` per segment. ``exp(avg_logprob)``
    recovers the geometric-mean token probability; scaling by ``1 - no_speech_prob``
    discounts segments the model thinks are silence/hallucination. The result is
    clamped to ``[0, 1]``.

    Note this is the model's *self-reported* confidence, which is known to be
    over-confident in overlapping speech; downstream modules combine it with
    cross-engine disagreement rather than trusting it alone.
    """
    speech_prob = 1.0 - _clamp01(no_speech_prob)
    confidence = float(np.exp(avg_logprob)) * speech_prob
    return _clamp01(confidence)


class ASRAdapter:
    """Base class for pluggable speech recognisers.

    Subclasses implement :meth:`transcribe_array`. File-level and per-segment
    helpers are derived from it so every adapter shares the same wiring.
    """

    name = "base"

    def transcribe_array(self, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
        """Transcribe mono float32 samples and return a transcript dict."""
        raise NotImplementedError

    def transcribe_file(self, audio_path: str) -> dict[str, Any]:
        """Load an audio file (via :mod:`src.audio.preprocess`) and transcribe it."""
        from ..audio.preprocess import load_audio

        samples, sample_rate = load_audio(audio_path)
        return self.transcribe_array(samples, sample_rate)


class MockASRAdapter(ASRAdapter):
    """Deterministic, dependency-free recogniser for tests and pipeline wiring.

    It produces placeholder text and a fixed confidence so the rest of the
    pipeline can be exercised without downloading a model. The output keeps the
    same shape as the real adapters.
    """

    name = "mock"

    def __init__(self, confidence: float = 0.9, language: str = "und") -> None:
        self.confidence = validate_score(confidence, "confidence")
        self.language = language

    def transcribe_array(self, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
        samples = np.asarray(samples, dtype=np.float32)
        duration = float(samples.size) / sample_rate if sample_rate else 0.0
        text = f"[mock transcript {duration:.2f}s]"
        segment = {
            "start_time": 0.0,
            "end_time": round(duration, 3),
            "text": text,
            "asr_confidence": self.confidence,
        }
        return {
            "text": text,
            "language": self.language,
            "model": self.name,
            "asr_confidence": self.confidence,
            "segments": [segment],
        }


class WhisperAdapter(ASRAdapter):
    """OpenAI Whisper recogniser (lazy ``whisper`` import).

    Confidence is derived per segment from ``avg_logprob`` and ``no_speech_prob``
    via :func:`logprob_to_confidence`.
    """

    name = "whisper"

    def __init__(self, model_size: str = "large-v3", device: str | None = None, language: str | None = None) -> None:
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            import whisper  # lazy: heavy, optional

            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe_array(self, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
        samples = _ensure_sample_rate(samples, sample_rate)
        model = self._ensure_model()
        result = model.transcribe(samples, language=self.language, verbose=False)
        return _from_whisper_result(result, self.name)


class WhisperXAdapter(ASRAdapter):
    """WhisperX recogniser (lazy ``whisperx`` import), recommended for low-overlap paths."""

    name = "whisperx"

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        default_confidence: float = 0.75,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.default_confidence = validate_score(default_confidence, "default_confidence")
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            import whisperx  # lazy: heavy, optional

            self._model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=self.language,
            )
        return self._model

    def transcribe_array(self, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
        samples = _ensure_sample_rate(samples, sample_rate)
        model = self._ensure_model()
        result = model.transcribe(samples, batch_size=16, language=self.language)
        return _from_whisperx_result(result, self.name, self.default_confidence, len(samples) / TARGET_SAMPLE_RATE)


class FasterWhisperAdapter(ASRAdapter):
    """faster-whisper recognizer sharing the cached candidate-generation model."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

    def transcribe_array(self, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
        from src.candidates.generator import _load_faster_whisper_model

        samples = _ensure_sample_rate(samples, sample_rate)
        model = _load_faster_whisper_model(self.model_size, self.device, self.compute_type)
        raw_segments, info = model.transcribe(samples, language=self.language)
        decoded = list(raw_segments)
        segments = [
            {
                "start_time": round(float(segment.start), 3),
                "end_time": round(float(segment.end), 3),
                "text": str(segment.text).strip(),
                "asr_confidence": logprob_to_confidence(
                    float(getattr(segment, "avg_logprob", 0.0)),
                    float(getattr(segment, "no_speech_prob", 0.0)),
                ),
            }
            for segment in decoded
        ]
        return {
            "text": " ".join(segment["text"] for segment in segments).strip(),
            "language": str(getattr(info, "language", self.language or "und")),
            "model": self.name,
            "asr_confidence": _aggregate_confidence(segments),
            "segments": segments,
        }


class FunASRAdapter(ASRAdapter):
    """FunASR Paraformer recogniser (lazy ``funasr`` import), strong for Chinese.

    FunASR does not expose token log-probabilities, so a neutral
    ``default_confidence`` is attached; the cross-engine disagreement signal is
    what flags Paraformer's uncertain regions downstream.
    """

    name = "funasr"

    def __init__(
        self,
        model: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        device: str | None = None,
        default_confidence: float = 0.6,
    ) -> None:
        self.model = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.device = device
        self.default_confidence = validate_score(default_confidence, "default_confidence")
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from funasr import AutoModel  # lazy: heavy, optional

            self._model = AutoModel(
                model=self.model,
                vad_model=self.vad_model,
                punc_model=self.punc_model,
                device=self.device,
                disable_update=True,
            )
        return self._model

    def transcribe_array(self, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
        samples = _ensure_sample_rate(samples, sample_rate)
        model = self._ensure_model()
        result = model.generate(input=samples, batch_size=1)[0]
        return _from_funasr_result(result, self.name, self.default_confidence, len(samples) / TARGET_SAMPLE_RATE)


_ADAPTERS: dict[str, type[ASRAdapter]] = {
    "mock": MockASRAdapter,
    "whisper": WhisperAdapter,
    "whisperx": WhisperXAdapter,
    "faster-whisper": FasterWhisperAdapter,
    "funasr": FunASRAdapter,
    "paraformer": FunASRAdapter,
}

def get_adapter(name: str = "mock", **kwargs: Any) -> ASRAdapter:
    """Build an ASR adapter by name (``mock``, ``whisperx``, ``whisper``, ``funasr``)."""
    from src.fallbacks import resolve_asr_backend

    key = name.lower()
    if key == "auto":
        key = resolve_asr_backend()
    if key not in _ADAPTERS:
        raise ValueError(f"unknown ASR adapter '{name}'; choose from {sorted(_ADAPTERS)}")
    return _ADAPTERS[key](**kwargs)


def transcribe_audio(audio_path: str, adapter: ASRAdapter | None = None, model: str = "mock") -> dict[str, Any]:
    """Return transcript text, timestamps, and confidence for an audio file.

    Pass an ``adapter`` instance to reuse a loaded model, or a ``model`` name to
    build one on the fly. Defaults to the dependency-free mock recogniser so the
    call never silently downloads a multi-gigabyte model.
    """
    adapter = adapter or get_adapter(model)
    return adapter.transcribe_file(audio_path)


def transcribe_segments(
    samples: np.ndarray,
    segments: list[dict[str, Any]],
    adapter: ASRAdapter,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Attach ``text`` and ``asr_confidence`` to VAD segments from preprocessing.

    Each input segment must carry ``start_time``/``end_time`` (as produced by
    :func:`src.audio.preprocess.segment_waveform`). The matching audio slice is
    transcribed and the result merged in, leaving the original keys intact so the
    enriched segments can flow into :func:`src.evidence.builder.build_metadata_segment`.
    """
    samples = np.asarray(samples, dtype=np.float32)
    enriched: list[dict[str, Any]] = []
    for segment in segments:
        start = int(round(float(segment["start_time"]) * sample_rate))
        end = int(round(float(segment["end_time"]) * sample_rate))
        clip = samples[max(0, start):max(0, end)]
        if clip.size == 0:
            text, confidence = "", 0.0
        else:
            result = adapter.transcribe_array(clip, sample_rate)
            text, confidence = result["text"], result["asr_confidence"]
        enriched.append({**segment, "text": text, "asr_confidence": confidence})
    return enriched


def _from_whisper_result(result: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Normalize a raw Whisper result into the shared transcript shape."""
    segments: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        confidence = logprob_to_confidence(
            float(seg.get("avg_logprob", 0.0)),
            float(seg.get("no_speech_prob", 0.0)),
        )
        segments.append({
            "start_time": round(float(seg.get("start", 0.0)), 3),
            "end_time": round(float(seg.get("end", 0.0)), 3),
            "text": str(seg.get("text", "")).strip(),
            "asr_confidence": confidence,
        })
    return {
        "text": str(result.get("text", "")).strip(),
        "language": str(result.get("language", "und")),
        "model": model_name,
        "asr_confidence": _aggregate_confidence(segments),
        "segments": segments,
    }


def _from_funasr_result(
    result: dict[str, Any],
    model_name: str,
    default_confidence: float,
    duration: float,
) -> dict[str, Any]:
    """Normalize a raw FunASR result into the shared transcript shape."""
    text = str(result.get("text", "")).strip()
    sentences = result.get("sentences")
    segments: list[dict[str, Any]] = []
    if isinstance(sentences, list) and sentences:
        for sentence in sentences:
            # FunASR timestamps are milliseconds.
            segments.append({
                "start_time": round(float(sentence.get("start", 0.0)) / 1000.0, 3),
                "end_time": round(float(sentence.get("end", 0.0)) / 1000.0, 3),
                "text": str(sentence.get("text", "")).strip(),
                "asr_confidence": default_confidence,
            })
    else:
        segments.append({
            "start_time": 0.0,
            "end_time": round(duration, 3),
            "text": text,
            "asr_confidence": default_confidence,
        })
    return {
        "text": text,
        "language": "zh",
        "model": model_name,
        "asr_confidence": _aggregate_confidence(segments),
        "segments": segments,
    }


def _from_whisperx_result(
    result: dict[str, Any],
    model_name: str,
    default_confidence: float,
    duration: float,
) -> dict[str, Any]:
    """Normalize a raw WhisperX result into the shared transcript shape."""
    segments: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        confidence = _segment_confidence(seg, default_confidence)
        segments.append({
            "start_time": round(float(seg.get("start", 0.0)), 3),
            "end_time": round(float(seg.get("end", 0.0)), 3),
            "text": str(seg.get("text", "")).strip(),
            "asr_confidence": confidence,
        })

    text = str(result.get("text", "")).strip()
    if not text:
        text = " ".join(segment["text"] for segment in segments).strip()
    if not segments:
        segments.append({
            "start_time": 0.0,
            "end_time": round(duration, 3),
            "text": text,
            "asr_confidence": default_confidence,
        })
    return {
        "text": text,
        "language": str(result.get("language", "und")),
        "model": model_name,
        "asr_confidence": _aggregate_confidence(segments),
        "segments": segments,
    }


def _segment_confidence(segment: dict[str, Any], default_confidence: float) -> float:
    """Best-effort confidence extraction for WhisperX-style segments."""
    if "asr_confidence" in segment:
        return _clamp01(float(segment["asr_confidence"]))
    if "confidence" in segment:
        return _clamp01(float(segment["confidence"]))
    if "avg_logprob" in segment:
        return logprob_to_confidence(
            float(segment.get("avg_logprob", 0.0)),
            float(segment.get("no_speech_prob", 0.0)),
        )
    if "words" in segment and isinstance(segment["words"], list):
        scores = [
            float(word["score"])
            for word in segment["words"]
            if isinstance(word, dict) and "score" in word
        ]
        if scores:
            return _clamp01(sum(scores) / len(scores))
    return default_confidence


def _aggregate_confidence(segments: list[dict[str, Any]]) -> float:
    """Duration-weighted mean of per-segment confidences (``0.0`` if empty)."""
    if not segments:
        return 0.0
    total_weight = 0.0
    weighted = 0.0
    for seg in segments:
        weight = max(0.0, float(seg["end_time"]) - float(seg["start_time"])) or 1.0
        weighted += weight * float(seg["asr_confidence"])
        total_weight += weight
    return _clamp01(weighted / total_weight) if total_weight else 0.0


def _ensure_sample_rate(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample to the 16 kHz rate expected by the speech models if needed."""
    samples = np.asarray(samples, dtype=np.float32)
    if sample_rate == TARGET_SAMPLE_RATE:
        return samples
    from ..audio.preprocess import resample

    return resample(samples, sample_rate, TARGET_SAMPLE_RATE)


def _clamp01(value: float) -> float:
    """Clamp a value to the ``[0.0, 1.0]`` range."""
    return max(0.0, min(1.0, float(value)))
