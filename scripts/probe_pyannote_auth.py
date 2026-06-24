"""Probe pyannote HF repos to see which still need authorization.

Run with the project's HF_TOKEN set. Prints which repos load OK and which
raise GatedRepoError (i.e. the user hasn't accepted terms on HF).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

REPOS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
    "pyannote/segmentation",
    "pyannote/overlapped-speech-detection",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/embedding",
]

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
api = HfApi(token=token)
print(f"Probing with token={'set' if token else 'NONE'}\n")

ok, blocked, missing = [], [], []
for r in REPOS:
    try:
        # Try to list files (HEAD-style probe). If gated, raises GatedRepoError.
        info = api.model_info(r)
        ok.append(r)
        print(f"  OK     {r}")
    except GatedRepoError:
        blocked.append(r)
        print(f"  GATED  {r}  <-- needs authorization at https://huggingface.co/{r}")
    except RepositoryNotFoundError:
        missing.append(r)
        print(f"  404    {r}")
    except Exception as e:
        print(f"  ERR    {r}  {type(e).__name__}: {str(e)[:120]}")

print(f"\n{len(ok)} OK, {len(blocked)} GATED, {len(missing)} NOT FOUND")
if blocked:
    print("\nAuthorize these on huggingface.co then re-run:")
    for r in blocked:
        print(f"  https://huggingface.co/{r}")
    sys.exit(1)