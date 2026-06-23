#!/usr/bin/env bash
# Create the project's .venv from scratch.
#
# Reads pyproject.toml for the Python version (>=3.11,<3.13) and uv.toml for
# the PyPI mirrors. Idempotent — running twice is a no-op once .venv exists.
#
# Usage:
#   bash scripts/setup_venv.sh        # default: pip
#   bash scripts/setup_venv.sh uv     # use uv (much faster, reads uv.lock)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
USE_UV=0
[[ "${1:-}" == "uv" ]] && USE_UV=1

# Sanity checks — better to fail loudly here than 5 minutes into pip install.
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' not found. Install Python 3.11 first." >&2
  echo "  - Debian/Ubuntu: sudo apt install python3.11 python3.11-venv" >&2
  echo "  - macOS:         brew install python@3.11" >&2
  exit 1
fi

# PyTorch CUDA wheels come from a separate index. Without this, pip grabs the
# CPU-only torch and the GPU pipeline silently runs on CPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "→ GPU detected via nvidia-smi; will install torch cu128 wheels."
  export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128${PIP_EXTRA_INDEX_URL:+,$PIP_EXTRA_INDEX_URL}"
fi

# Create the venv if it doesn't exist
if [[ ! -d ".venv" ]]; then
  echo "→ Creating .venv with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ "$USE_UV" == "1" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not installed. Install from https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  echo "→ uv sync (reads uv.toml for PyPI mirrors)"
  uv sync --extra dev
else
  echo "→ pip install -e .[dev]"
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev]"
fi

echo
echo "✓ .venv ready at $(pwd)/.venv"
echo "  Python:  $(python --version)"
echo "  Pinned:  $(python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())' 2>/dev/null || echo 'torch not installed (CPU-only path)')"
echo
echo "Next:"
echo "  cp .env.example .env       # then fill in HF_TOKEN, OPENAI_API_KEY, ..."
echo "  python scripts/setup_models.py   # pre-download weights"
echo "  pytest -q                  # smoke test"
