"""Stage pyannote/segmentation from modelscope into the HF cache layout.

Pyannote.audio 4.x pulls model files via huggingface_hub, but hf-mirror.com
does not proxy gated-repo weights. We sidestep that by downloading the same
files from ModelScope (a public mirror) and copying them into the standard
HuggingFace cache layout so the next ``Pipeline.from_pretrained`` call finds
them locally (and HF_HUB_OFFLINE=1 stops any further attempts).
"""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

CACHE_ROOT = Path(os.path.expanduser("~/.cache/huggingface/hub"))
DEST = CACHE_ROOT / "models--pyannote--segmentation"
SRC = Path(os.path.expanduser("~/.cache/modelscope/pyannote/segmentation"))
REVISION = "Interspeech2021"  # the revision pinned by OSD's config.yaml


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    # Copy files into blobs/<sha256> and create snapshot symlinks.
    (DEST / "blobs").mkdir(parents=True, exist_ok=True)
    (DEST / "refs").mkdir(exist_ok=True)
    snap_dir = DEST / "snapshots" / REVISION
    snap_dir.mkdir(parents=True, exist_ok=True)

    files = ["config.yaml", "configuration.json", "pytorch_model.bin", "README.md"]
    for name in files:
        src_file = SRC / name
        if not src_file.exists():
            print(f"  SKIP {name} (not in modelscope cache)")
            continue
        blob_hash = _sha256(src_file)
        blob_path = DEST / "blobs" / blob_hash
        if not blob_path.exists():
            shutil.copy(src_file, blob_path)
            print(f"  copied {name} -> blobs/{blob_hash[:8]} ({src_file.stat().st_size} B)")
        snap_link = snap_dir / name
        if snap_link.is_symlink() or snap_link.exists():
            snap_link.unlink()
        os.symlink(f"../../blobs/{blob_hash}", snap_link)
        print(f"  linked snapshots/{REVISION}/{name}")

    (DEST / "refs" / "main").write_text(REVISION + "\n")
    print(f"  wrote refs/main = {REVISION}")
    print("\nDone. Use HF_HUB_OFFLINE=1 to skip network checks.")


if __name__ == "__main__":
    main()