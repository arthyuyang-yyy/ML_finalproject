"""Probe loading pyannote/segmentation from local directory."""
from __future__ import annotations

import os
import sys
import traceback

print("HF_ENDPOINT =", repr(os.environ.get("HF_ENDPOINT")))
print("HF_HUB_OFFLINE =", repr(os.environ.get("HF_HUB_OFFLINE")))

local_dir = os.path.expanduser(
    "~/.cache/huggingface/hub/models--pyannote--segmentation/snapshots/Interspeech2021"
)
print("local_dir =", local_dir)
print("contents:", os.listdir(local_dir))

try:
    from pyannote.audio.core.model import Model
    print(">>> Model.from_pretrained(local_dir)")
    model = Model.from_pretrained(local_dir)
    print("OK:", type(model).__name__)
    print("model.specifications =", getattr(model, "specifications", None))
except Exception:
    traceback.print_exc()