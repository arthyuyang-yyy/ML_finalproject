"""Probe the OSD path end-to-end via the project's helpers."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Unset the mirror endpoint so any accidental HEAD doesn't go to hf-mirror.
# (We won't reach the network because we'll resolve the local snapshot.)
os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_HUB_OFFLINE"] = "1"

print("HF_ENDPOINT =", repr(os.environ.get("HF_ENDPOINT")))
print("HF_HUB_OFFLINE =", repr(os.environ.get("HF_HUB_OFFLINE")))

from src.overlap.detector import _osd_inference_fallback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Build a tiny 5-second waveform (16 kHz mono) — enough for one PyanNet window.
duration_s = 5.0
sr = 16000
t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
# Two interleaved tones to mimic overlap (not that it matters — we only test
# that the Inference path runs without crashing).
wave = 0.2 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 660 * t)).astype("float32")
file = {
    "waveform": torch.from_numpy(wave).unsqueeze(0),  # (channel=1, time)
    "sample_rate": sr,
}

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    print("FAILED: set HF_TOKEN or HUGGINGFACE_TOKEN before running this probe.")
    sys.exit(2)
print("token starts with hf_:", token.startswith("hf_"))

try:
    print(">>> _osd_inference_fallback(file, token)")
    output = _osd_inference_fallback(file, token)
    print("OK:", type(output).__name__)
    print("repr:", repr(output)[:200])
    if hasattr(output, "itertracks"):
        regions = list(output.itertracks(yield_label=True))
        print("regions:", regions[:5], "... total:", len(regions))
except Exception:
    print("FAILED:")
    traceback.print_exc()
    sys.exit(1)
