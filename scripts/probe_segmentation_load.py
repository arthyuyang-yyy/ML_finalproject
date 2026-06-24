"""Minimal repro of pyannote/segmentation Model.from_pretrained."""
from __future__ import annotations

import os
import sys
import traceback

# Print the env we're testing under
print("HF_ENDPOINT =", repr(os.environ.get("HF_ENDPOINT")))
print("HF_HUB_OFFLINE =", repr(os.environ.get("HF_HUB_OFFLINE")))
print("HF_TOKEN set =", bool(os.environ.get("HF_TOKEN")))

try:
    from pyannote.audio.core.model import Model
    print(">>> Model.from_pretrained('pyannote/segmentation', revision='Interspeech2021')")
    model = Model.from_pretrained(
        "pyannote/segmentation",
        revision="Interspeech2021",
        token=os.environ.get("HF_TOKEN"),
    )
    print("OK:", type(model).__name__)
    print("model.specifications =", getattr(model, "specifications", None))
except Exception:
    print("FAILED:")
    traceback.print_exc()
    sys.exit(1)