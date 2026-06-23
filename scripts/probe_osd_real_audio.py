"""Probe the full detect_pyannote_overlap_regions path on real audio."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import traceback

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Use offline mode (cache hit) but keep HF_ENDPOINT unset.
os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_HUB_OFFLINE"] = "1"

from src.overlap.detector import detect_pyannote_overlap_regions  # noqa: E402

audio = sys.argv[1] if len(sys.argv) > 1 else str(repo_root / "data/demo/R8001_M8004_clip5min.wav")
print("audio =", audio, "exists =", os.path.isfile(audio))
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    print("FAILED: set HF_TOKEN or HUGGINGFACE_TOKEN before running this probe.")
    sys.exit(2)

t0 = time.time()
try:
    regions = detect_pyannote_overlap_regions(audio, auth_token=token)
    print(f"elapsed {time.time()-t0:.1f}s")
    print("regions:", len(regions) if regions else "None")
    if regions:
        for s, e in regions[:8]:
            print(f"  {s:.2f} -> {e:.2f}  ({e-s:.2f}s)")
        total = sum(e - s for s, e in regions)
        print(f"total overlap seconds = {total:.2f}")
except Exception:
    traceback.print_exc()
