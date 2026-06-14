"""Optional speech-separation adapters for high-overlap audio."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.audio.preprocess import TARGET_SAMPLE_RATE, load_audio, resample
from src.errors import BackendExecutionError, BackendOutputError, BackendUnavailableError
from src.nmf_separation import DEFAULT_NUM_SOURCES, NmfSeparationBackend

DEFAULT_SEPFORMER_MODEL = "speechbrain/sepformer-whamr16k"
logger = logging.getLogger(__name__)


class SpeechSeparationAdapter(Protocol):
    """Interface implemented by replaceable speech-separation backends."""

    name: str

    def separate_array(self, samples: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """Return one mono float32 waveform per estimated speaker source."""
        ...


class DisabledSpeechSeparationAdapter:
    """No-op adapter used by default so the lightweight pipeline stays runnable."""

    name = "none"

    def separate_array(self, samples: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        return []


class MockSpeechSeparationAdapter:
    """Deterministic dependency-free adapter for tests and integration checks."""

    name = "mock"

    def __init__(self, sources: list[np.ndarray] | None = None) -> None:
        self.sources = sources

    def separate_array(self, samples: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        if self.sources is not None:
            return [np.asarray(source, dtype=np.float32) for source in self.sources]
        waveform = np.asarray(samples, dtype=np.float32)
        return [waveform.copy(), waveform.copy()]


class NmfSeparationAdapter:
    """Dependency-free NMF separation baseline wrapped as a pluggable adapter.

    Wraps the from-scratch numpy ``NmfSeparationBackend`` (see
    :mod:`src.nmf_separation`) in the :class:`SpeechSeparationAdapter` interface
    so it can be selected from the CLI/pipeline like any other backend. Unlike
    SepFormer it needs no heavy dependencies, so it is the default *real*
    separator that CI can exercise end to end. Source order carries no speaker
    identity; downstream keeps the neutral ``SEPARATED_SOURCE_0x`` labels.
    """

    name = "nmf"

    def __init__(
        self,
        num_sources: int = DEFAULT_NUM_SOURCES,
        n_components: int | None = None,
        n_iter: int = 200,
        seed: int = 0,
    ) -> None:
        if num_sources <= 0:
            raise ValueError("num_sources must be positive")
        self.num_sources = num_sources
        self._backend = NmfSeparationBackend(
            n_components=n_components, n_iter=n_iter, seed=seed
        )

    def separate_array(self, samples: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return []
        try:
            sources = self._backend.separate(waveform, sample_rate, self.num_sources)
        except Exception as exc:
            raise BackendExecutionError("NMF separation failed") from exc
        return [np.asarray(source, dtype=np.float32) for source in sources]


class SepFormerAdapter:
    """SpeechBrain SepFormer baseline using the 16 kHz WHAMR model."""

    name = "sepformer"

    def __init__(
        self,
        model_source: str = DEFAULT_SEPFORMER_MODEL,
        device: str = "cpu",
        model_sample_rate: int = TARGET_SAMPLE_RATE,
        savedir: str | Path | None = None,
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.model_sample_rate = model_sample_rate
        self.savedir = str(savedir or _default_model_cache(model_source))

    def separate_array(self, samples: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return []
        model_input = (
            resample(waveform, sample_rate, self.model_sample_rate)
            if sample_rate != self.model_sample_rate
            else waveform
        )
        try:
            separator = _load_sepformer(self.model_source, self.savedir, self.device)
            estimates = separator.separate_batch(_to_model_batch(model_input))
            sources = _estimate_to_sources(estimates)
        except (BackendUnavailableError, BackendOutputError):
            raise
        except Exception as exc:
            raise BackendExecutionError(
                f"SepFormer separation failed for model '{self.model_source}'"
            ) from exc

        if sample_rate != self.model_sample_rate:
            sources = [resample(source, self.model_sample_rate, sample_rate) for source in sources]
        return [_match_length(source, waveform.size) for source in sources]


def get_separation_adapter(name: str = "none", **kwargs: Any) -> SpeechSeparationAdapter:
    """Build a speech-separation adapter without loading its model."""
    normalized = (name or "none").strip().lower()
    if normalized in {"none", "disabled", "off"}:
        return DisabledSpeechSeparationAdapter()
    if normalized == "mock":
        return MockSpeechSeparationAdapter(**kwargs)
    if normalized == "nmf":
        return NmfSeparationAdapter(**kwargs)
    if normalized in {"sepformer", "speechbrain-sepformer"}:
        return SepFormerAdapter(**kwargs)
    raise ValueError(f"unknown speech-separation backend: {name}")


def separate_waveform(
    samples: np.ndarray,
    sample_rate: int,
    adapter: SpeechSeparationAdapter | None = None,
    *,
    fallback_on_error: bool = True,
) -> list[np.ndarray]:
    """Separate a waveform, optionally returning no sources when a backend fails."""
    backend = adapter or DisabledSpeechSeparationAdapter()
    try:
        sources = backend.separate_array(np.asarray(samples, dtype=np.float32), sample_rate)
        return _validate_sources(sources)
    except (BackendUnavailableError, BackendExecutionError, BackendOutputError) as exc:
        if not fallback_on_error:
            raise
        logger.warning("Speech separation failed; using multi-decode fallback: %s", exc)
        return []


def separate_speakers(
    audio_path: str,
    output_dir: str | Path | None = None,
    adapter: SpeechSeparationAdapter | None = None,
) -> list[str]:
    """Separate an audio file and write one float WAV per estimated source."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - only exercised without backend
        raise BackendUnavailableError(
            "writing separated sources requires soundfile; install requirements.txt"
        ) from exc

    samples, sample_rate = load_audio(audio_path)
    sources = separate_waveform(samples, sample_rate, adapter, fallback_on_error=False)
    destination = Path(output_dir or f"{Path(audio_path).with_suffix('')}_separated")
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, source in enumerate(sources, start=1):
        path = destination / f"source_{index:02d}.wav"
        sf.write(path, source, sample_rate, subtype="FLOAT")
        paths.append(str(path))
    return paths


def _validate_sources(sources: list[np.ndarray]) -> list[np.ndarray]:
    validated: list[np.ndarray] = []
    for source in sources:
        waveform = np.asarray(source, dtype=np.float32).reshape(-1)
        if waveform.size and np.isfinite(waveform).all():
            validated.append(waveform)
    return validated


def _default_model_cache(model_source: str) -> Path:
    slug = model_source.replace("/", "--")
    return Path.home() / ".cache" / "ml_finalproject" / slug


def _to_model_batch(samples: np.ndarray) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "SepFormer requires SpeechBrain and PyTorch; install requirements-separation.txt"
        ) from exc
    return torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)


@lru_cache(maxsize=4)
def _load_sepformer(model_source: str, savedir: str, device: str) -> Any:
    try:
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation
    except ImportError as exc:
        raise BackendUnavailableError(
            "SepFormer requires SpeechBrain and PyTorch; install requirements-separation.txt"
        ) from exc
    try:
        return SepformerSeparation.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": device},
        )
    except Exception as exc:
        raise BackendUnavailableError(
            f"unable to load SepFormer model '{model_source}'"
        ) from exc


def _estimate_to_sources(estimates: Any) -> list[np.ndarray]:
    try:
        array = estimates.detach().cpu().numpy()
    except AttributeError:
        array = np.asarray(estimates)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        return [array]
    if array.ndim != 2:
        raise BackendOutputError(
            f"SepFormer returned unsupported output shape {array.shape}"
        )
    if array.shape[1] <= 8:
        return [array[:, index] for index in range(array.shape[1])]
    if array.shape[0] <= 8:
        return [array[index] for index in range(array.shape[0])]
    raise BackendOutputError(
        f"SepFormer output shape {array.shape} does not expose a source axis"
    )


def _match_length(samples: np.ndarray, target_length: int) -> np.ndarray:
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform.size >= target_length:
        return waveform[:target_length]
    return np.pad(waveform, (0, target_length - waveform.size)).astype(np.float32)


__all__ = [
    "DEFAULT_SEPFORMER_MODEL",
    "DisabledSpeechSeparationAdapter",
    "MockSpeechSeparationAdapter",
    "NmfSeparationAdapter",
    "SepFormerAdapter",
    "SpeechSeparationAdapter",
    "get_separation_adapter",
    "separate_speakers",
    "separate_waveform",
]
