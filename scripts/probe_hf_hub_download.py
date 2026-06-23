"""Probe what hf_hub_download does for pytorch_model.bin."""
from __future__ import annotations

import os
import sys
import traceback

print("HF_ENDPOINT =", repr(os.environ.get("HF_ENDPOINT")))
print("HF_HUB_OFFLINE =", repr(os.environ.get("HF_HUB_OFFLINE")))
print("HF_TOKEN set =", bool(os.environ.get("HF_TOKEN")))

try:
    from huggingface_hub import hf_hub_download
    print(">>> hf_hub_download('pyannote/segmentation', 'pytorch_model.bin', revision='Interspeech2021')")
    p = hf_hub_download(
        "pyannote/segmentation",
        "pytorch_model.bin",
        revision="Interspeech2021",
        token=os.environ.get("HF_TOKEN"),
    )
    print("OK:", p)
except Exception:
    traceback.print_exc()

try:
    print(">>> hf_hub_download('pyannote/segmentation', 'hparams.yaml', revision='Interspeech2021')")
    p = hf_hub_download(
        "pyannote/segmentation",
        "hparams.yaml",
        revision="Interspeech2021",
        token=os.environ.get("HF_TOKEN"),
    )
    print("OK:", p)
except Exception:
    traceback.print_exc()