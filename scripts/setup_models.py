#!/usr/bin/env python3
"""Pre-download every production model used by the meeting-memory pipeline.

Usage (from project root):
    python scripts/setup_models.py

The script is idempotent: already-cached repos are skipped by
``huggingface_hub.snapshot_download`` and ``whisper.load_model``.

Models cached by this script (and where the pipeline loads them from):

    Systran/faster-whisper-small          ~/.cache/huggingface/hub   (default faster-whisper ASR + candidates generator)
    Systran/faster-whisper-large-v3       ~/.cache/huggingface/hub   (WhisperX large-v3 ASR)
    pyannote/speaker-diarization-3.1      ~/.cache/huggingface/hub   (diarization pipeline config)
    pyannote/segmentation-3.0             ~/.cache/huggingface/hub   (diarization segmentation sub-model)
    pyannote/wespeaker-voxceleb-resnet34-LM  ~/.cache/huggingface/hub (diarization embedding sub-model)
    pyannote/speaker-diarization-community-1  ~/.cache/huggingface/hub (PLDA xvec transform, pyannote 4.x)
    pyannote/overlapped-speech-detection  ~/.cache/huggingface/hub   (OSD pipeline config)
    pyannote/segmentation @Interspeech2021  ~/.cache/huggingface/hub (OSD segmentation sub-model, pinned revision)
    speechbrain/sepformer-whamr16k        ~/.cache/huggingface/hub   (optional SepFormer separation)
    openai-whisper large-v3.pt            ~/.cache/whisper/          (openai-whisper --asr whisper path)
    Resemblyzer VoiceEncoder pretrained.pt  resemblyzer package cache (learned d-vector speaker embedding)
    FunASR paraformer-zh / fsmn-vad / ct-punc  ~/.cache/modelscope/hub (Chinese ASR path)

Gated pyannote repos require a HF token AND that the user has accepted the
model terms on Hugging Face:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/overlapped-speech-detection
    https://huggingface.co/pyannote/segmentation-3.0
    https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
    https://huggingface.co/pyannote/segmentation
    https://huggingface.co/pyannote/speaker-diarization-community-1
Set ``HF_TOKEN`` in ``.env`` (see ``.env.example``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 1. Bootstrap: set env vars BEFORE any library that reads them
# ═══════════════════════════════════════════════════════════════════════════

_PROJECT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT / ".env"

if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _stripped = _line.strip()
        if not _stripped or _stripped.startswith("#"):
            continue
        if "=" in _stripped:
            _k, _, _v = _stripped.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip("\"'")
            if _k and _k not in os.environ:
                os.environ[_k] = _v

# Token from env or cached file
if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_TOKEN"):
    _TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
    if _TOKEN_FILE.is_file():
        os.environ["HF_TOKEN"] = _TOKEN_FILE.read_text(encoding="utf-8").strip()

sys.path.insert(0, str(_PROJECT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup_models")

# Suppress the noisy torchcodec/torchaudio import warnings that flood the log
# on first import of pyannote.audio / speechbrain.
logging.getLogger("speechbrain.utils.torch_audio_backend").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Download helpers
# ═══════════════════════════════════════════════════════════════════════════


def _hf_snapshot(repo_id: str, label: str, revision: str | None = None) -> bool:
    """Download a HF repo. Uses HF_ENDPOINT mirror if set.

    On ``GatedRepoError`` the script opens the repo's HF page in the default
    browser so the token's account owner can click "Agree and access
    repository" (Hugging Face deliberately exposes no API for this step — it
    must be done in a browser). The download then polls until access is
    granted or the timeout expires.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    from huggingface_hub.utils import LocalEntryNotFoundError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    target = f"{repo_id}@{revision}" if revision else repo_id
    logger.info("  [%s] hf: %s", label, target)

    def _try() -> tuple[bool, Exception | None]:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=token or None,
                resume_download=True,
            )
            return True, None
        except LocalEntryNotFoundError as exc:
            return False, exc
        except GatedRepoError as exc:
            return False, exc
        except Exception as exc:
            return False, exc

    ok, err = _try()
    if ok:
        logger.info("  [%s] ✓ cached", label)
        return True

    # If the repo is gated, walk the user through the one browser click.
    if isinstance(err, GatedRepoError):
        import time
        import webbrowser

        page_url = f"https://huggingface.co/{repo_id}"
        if revision:
            page_url += f"/tree/{revision}"
        logger.warning(
            "  [%s] gated repo — your HF account has not accepted the terms yet.\n"
            "    Opening browser to: %s\n"
            "    ➜ Log in as '%s', fill Company/University + Website, click\n"
            "      \"Agree and access repository\". The script will auto-retry.",
            label,
            page_url,
            _hf_username(token) or "(your HF account)",
        )
        try:
            webbrowser.open(page_url)
        except Exception:
            pass

        deadline = time.time() + 300  # 5 minutes to click agree
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            time.sleep(5)
            ok, err = _try()
            if ok:
                logger.info("  [%s] ✓ access granted, cached (after %d retries)", label, attempt)
                return True
            if not isinstance(err, GatedRepoError):
                break  # a different error surfaced — stop polling
        logger.error("  [%s] ✗ still gated after 5 min (last: %s)", label, err)
        return False

    logger.error("  [%s] ✗ %s", label, err)
    return False


def _hf_username(token: str | None) -> str | None:
    """Best-effort lookup of the HF account name behind *token*."""
    if not token:
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi(token=token).whoami().get("name")
    except Exception:
        return None


def _download_openai_whisper(model_size: str = "large-v3") -> bool:
    """Trigger openai-whisper's own download to ~/.cache/whisper/<size>.pt."""
    try:
        import whisper
    except ImportError:
        logger.warning("  [openai-whisper] package not installed; skipping %s", model_size)
        return False

    cache_dir = Path.home() / ".cache" / "whisper"
    expected = cache_dir / f"{model_size}.pt"
    if expected.is_file() and expected.stat().st_size > 0:
        logger.info("  [openai-whisper %s] ✓ already cached (%s)", model_size, expected)
        return True

    logger.info("  [openai-whisper %s] downloading .pt (this can take a while)...", model_size)
    try:
        whisper.load_model(model_size, device="cpu")
        logger.info("  [openai-whisper %s] ✓ cached", model_size)
        return True
    except Exception as exc:
        logger.error("  [openai-whisper %s] ✗ %s", model_size, exc)
        return False


def _download_resemblyzer() -> bool:
    """Trigger Resemblyzer's VoiceEncoder pretrained.pt download by instantiating it."""
    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        logger.warning("  [resemblyzer] package not installed; skipping")
        return False

    logger.info("  [resemblyzer] loading VoiceEncoder (triggers pretrained.pt download)...")
    try:
        encoder = VoiceEncoder()
        # Force the pretrained weights to materialize by running one embedding.
        import numpy as np

        wav = np.zeros(16000, dtype=np.float32)
        encoder.embed_utterance(wav)
        logger.info("  [resemblyzer] ✓ VoiceEncoder ready")
        return True
    except Exception as exc:
        logger.error("  [resemblyzer] ✗ %s", exc)
        return False


def _download_funasr() -> bool:
    """Download FunASR models. They auto-fetch from ModelScope on first use."""
    import numpy as np

    try:
        from funasr import AutoModel
    except ImportError:
        logger.warning("  [funasr] package not installed; skipping")
        return False

    logger.info("  [funasr] loading (triggers auto-download from modelscope if missing)...")
    try:
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )
        dummy = np.zeros(16000, dtype=np.float32)
        model.generate(input=dummy, batch_size=1)
        logger.info("  [funasr] ✓ models cached")
        return True
    except Exception as exc:
        logger.error("  [funasr] ✗ %s", exc)
        return False


def _download_all() -> dict[str, bool]:
    results: dict[str, bool] = {}

    # ── 1. faster-whisper models (used by WhisperX + faster-whisper adapter) ──
    results["faster-whisper-small"] = _hf_snapshot(
        "Systran/faster-whisper-small",
        "faster-whisper-small (~0.5GB, default ASR + candidates)",
    )
    results["faster-whisper-large-v3"] = _hf_snapshot(
        "Systran/faster-whisper-large-v3",
        "faster-whisper-large-v3 (~3GB, WhisperX large-v3)",
    )

    # ── 2. pyannote diarization 3.1 + its sub-models ──
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.warning("  Skipping pyannote models: no HF_TOKEN set")
        results["pyannote-diarization"] = False
        results["pyannote-diar-segmentation"] = False
        results["pyannote-diar-embedding"] = False
        results["pyannote-osd"] = False
        results["pyannote-osd-segmentation"] = False
    else:
        results["pyannote-diarization"] = _hf_snapshot(
            "pyannote/speaker-diarization-3.1",
            "pyannote diarization pipeline config",
        )
        results["pyannote-diar-segmentation"] = _hf_snapshot(
            "pyannote/segmentation-3.0",
            "pyannote segmentation-3.0 (~80MB, diarization sub-model)",
        )
        results["pyannote-diar-embedding"] = _hf_snapshot(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            "pyannote wespeaker-voxceleb-resnet34-LM (~66MB, diarization embedding)",
        )
        # The wespeaker embedding backend in pyannote.audio 4.x also pulls a
        # PLDA transform from the legacy community-1 repo, which is separately
        # gated. Pre-cache it so diarization does not fail at inference time.
        results["pyannote-community-1"] = _hf_snapshot(
            "pyannote/speaker-diarization-community-1",
            "pyannote speaker-diarization-community-1 (~32MB, PLDA xvec transform)",
        )
        # ── 3. pyannote overlapped-speech-detection + its sub-model ──
        #    OSD config references pyannote/segmentation at the Interspeech2021
        #    revision, so pin that revision explicitly.
        results["pyannote-osd"] = _hf_snapshot(
            "pyannote/overlapped-speech-detection",
            "pyannote OSD pipeline config",
        )
        results["pyannote-osd-segmentation"] = _hf_snapshot(
            "pyannote/segmentation",
            "pyannote segmentation@Interspeech2021 (~30MB, OSD sub-model)",
            revision="Interspeech2021",
        )

    # ── 4. FunASR Chinese models (via ModelScope, auto-fetched on first use) ──
    results["funasr"] = _download_funasr()

    # ── 5. SpeechBrain SepFormer (optional high-overlap separation) ──
    results["sepformer"] = _hf_snapshot(
        "speechbrain/sepformer-whamr16k",
        "SepFormer (~0.5GB, optional speech separation)",
    )

    # ── 6. OpenAI Whisper large-v3 (.pt, independent of faster-whisper) ──
    results["openai-whisper-large-v3"] = _download_openai_whisper("large-v3")

    # ── 7. Resemblyzer VoiceEncoder (optional learned d-vector backend) ──
    results["resemblyzer"] = _download_resemblyzer()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. Smoke-test (verify models actually load through their project adapters)
# ═══════════════════════════════════════════════════════════════════════════


def _smoke_faster_whisper_small() -> bool:
    """faster-whisper small (default ASR + candidates generator)."""
    import numpy as np

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("  [faster-whisper] not installed")
        return False
    logger.info("  [faster-whisper small] loading...")
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        dummy = np.zeros(16000, dtype=np.float32)
        list(model.transcribe(dummy, beam_size=1)[0])
        logger.info("  [faster-whisper small] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [faster-whisper small] ✗ %s", exc)
        return False


def _smoke_whisperx() -> bool:
    """WhisperX large-v3 (the project's WhisperXAdapter default)."""
    import numpy as np

    try:
        import whisperx
    except ImportError:
        logger.warning("  [whisperx] not installed")
        return False

    logger.info("  [whisperx large-v3] loading (may take a while on first run)...")
    try:
        model = whisperx.load_model("large-v3", "cpu", compute_type="int8", language=None)
        dummy = np.zeros(16000, dtype=np.float32)
        model.transcribe(dummy, batch_size=1, language=None)
        logger.info("  [whisperx large-v3] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [whisperx large-v3] ✗ %s", exc)
        return False


def _smoke_openai_whisper() -> bool:
    """openai-whisper large-v3 (the project's WhisperAdapter default)."""
    import numpy as np

    try:
        import whisper
    except ImportError:
        logger.warning("  [openai-whisper] not installed")
        return False

    logger.info("  [openai-whisper large-v3] loading...")
    try:
        model = whisper.load_model("large-v3", device="cpu")
        dummy = np.zeros(16000, dtype=np.float32)
        model.transcribe(dummy, verbose=False)
        logger.info("  [openai-whisper large-v3] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [openai-whisper large-v3] ✗ %s", exc)
        return False


def _smoke_pyannote_diarization() -> bool:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return False
    try:
        from pyannote.audio import Pipeline
        import numpy as np
        import torch

        logger.info("  [pyannote diarization] loading pipeline (downloads sub-models if missing)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        dummy = np.zeros((1, 16000 * 2), dtype=np.float32)
        pipeline({"waveform": torch.from_numpy(dummy), "sample_rate": 16000})
        logger.info("  [pyannote diarization] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [pyannote diarization] ✗ %s", exc)
        return False


def _smoke_pyannote_osd() -> bool:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return False
    try:
        from pyannote.audio import Inference
        import numpy as np
        import torch

        logger.info("  [pyannote OSD] loading (downloads sub-model if missing)...")
        model = Inference(
            "pyannote/overlapped-speech-detection",
            token=token,
        )
        dummy = np.zeros((1, 16000 * 2), dtype=np.float32)
        model({"waveform": torch.from_numpy(dummy), "sample_rate": 16000})
        logger.info("  [pyannote OSD] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [pyannote OSD] ✗ %s", exc)
        return False


def _smoke_funasr() -> bool:
    import numpy as np

    try:
        from funasr import AutoModel
    except ImportError:
        return False
    logger.info("  [funasr] loading paraformer-zh + fsmn-vad + ct-punc...")
    try:
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )
        dummy = np.zeros(16000, dtype=np.float32)
        model.generate(input=dummy, batch_size=1)
        logger.info("  [funasr] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [funasr] ✗ %s", exc)
        return False


def _smoke_sepformer() -> bool:
    import numpy as np

    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError:
        logger.warning("  [sepformer] speechbrain not installed")
        return False
    logger.info("  [sepformer] loading speechbrain/sepformer-whamr16k...")
    try:
        sep = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir=str(Path.home() / ".cache" / "ml_finalproject" / "speechbrain--sepformer-whamr16k"),
            run_opts={"device": "cpu"},
        )
        dummy = np.zeros(16000, dtype=np.float32)
        sep.separate_batch(_to_torch_batch(dummy))
        logger.info("  [sepformer] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [sepformer] ✗ %s", exc)
        return False


def _to_torch_batch(samples):
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)


def _smoke_resemblyzer() -> bool:
    import numpy as np

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return False
    logger.info("  [resemblyzer] loading VoiceEncoder...")
    try:
        encoder = VoiceEncoder()
        wav = preprocess_wav(np.zeros(16000 * 2, dtype=np.float32), source_sr=16000)
        encoder.embed_utterance(wav)
        logger.info("  [resemblyzer] ✓ loads & runs")
        return True
    except Exception as exc:
        logger.error("  [resemblyzer] ✗ %s", exc)
        return False


def _smoke_ollama_gemma() -> bool:
    """Verify the local Ollama server and the gemma3:4b model are available."""
    import json
    import urllib.error
    import urllib.request

    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL") or os.environ.get("GEMMA_MODEL") or "gemma3:4b"
    logger.info("  [ollama %s] probing %s ...", model, base_url)
    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps({"model": model, "prompt": "ping", "stream": False, "options": {"num_predict": 1}}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            json.loads(resp.read().decode("utf-8"))
        logger.info("  [ollama %s] ✓ responds", model)
        return True
    except Exception as exc:
        logger.error("  [ollama %s] ✗ %s", model, exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    print()
    print("=" * 64)
    print("  Model Setup — Overlap-aware Dual-path ASR with Episodic Memory")
    print("=" * 64)
    print()

    logger.info("HF_ENDPOINT  = %s", os.environ.get("HF_ENDPOINT", "(default huggingface.co)"))
    logger.info("https_proxy  = %s", os.environ.get("https_proxy", "(not set)"))
    logger.info("HF_TOKEN     = %s", "present" if os.environ.get("HF_TOKEN") else "MISSING")
    logger.info("OLLAMA_URL   = %s", os.environ.get("OLLAMA_URL", "(default http://localhost:11434)"))
    print()

    if not os.environ.get("HF_TOKEN"):
        logger.warning(
            "No HF_TOKEN found. pyannote diarization/OSD will be skipped.\n"
            "  Set HF_TOKEN in .env or export it, then re-run this script.\n"
        )

    logger.info(
        "Gated pyannote repos require accepting terms FIRST:\n"
        "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "  https://huggingface.co/pyannote/overlapped-speech-detection\n"
        "  https://huggingface.co/pyannote/segmentation-3.0\n"
        "  https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM\n"
        "  https://huggingface.co/pyannote/segmentation\n"
        "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        "If a repo is still gated when the script runs, it will open the\n"
        "repo page in your browser so you can click 'Agree and access\n"
        "repository', then auto-retry the download.\n"
    )

    # ─── Phase 1: Download ───
    print()
    print("─" * 64)
    print("  Phase 1/2: Downloading models...")
    print("─" * 64)
    print()

    results = _download_all()

    # ─── Phase 2: Smoke tests ───
    print()
    print("─" * 64)
    print("  Phase 2/2: Smoke-testing models (loading each through its adapter)...")
    print("─" * 64)
    print()

    results["smoke:faster-whisper-small"] = _smoke_faster_whisper_small()
    results["smoke:whisperx-large-v3"] = _smoke_whisperx()
    results["smoke:openai-whisper-large-v3"] = _smoke_openai_whisper()
    results["smoke:pyannote-diarization"] = _smoke_pyannote_diarization()
    results["smoke:pyannote-osd"] = _smoke_pyannote_osd()
    results["smoke:funasr"] = _smoke_funasr()
    results["smoke:sepformer"] = _smoke_sepformer()
    results["smoke:resemblyzer"] = _smoke_resemblyzer()
    results["smoke:ollama-gemma3:4b"] = _smoke_ollama_gemma()

    # ─── Summary ───
    print()
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status}  {name}")
    print()

    ok_count = sum(results.values())
    total = len(results)
    print(f"  {ok_count}/{total} steps ok")
    print()
    print("  Run pipeline with WhisperX + Ollama Gemma:")
    print("    python main.py <audio.wav> --asr whisperx --gemma-backend ollama --gemma-model gemma3:4b")
    print("  Run with faster-whisper (default) + Ollama Gemma:")
    print("    python main.py <audio.wav> --asr faster-whisper --gemma-backend ollama --gemma-model gemma3:4b")
    print()


if __name__ == "__main__":
    main()
