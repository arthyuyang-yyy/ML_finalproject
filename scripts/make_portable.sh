#!/usr/bin/env bash
# Copy the project source tree into a sibling ``*-portable`` directory,
# stripped of every per-machine, per-run, or per-user artefact so the
# result is ready to:
#   - ``git init && git add . && git push`` to GitHub
#   - ``tar czf`` for download / archival
#   - clone onto a fresh machine and ``bash scripts/setup_venv.sh``
#
# Defaults: destination = ``../meeting-memory-deploy-portable``
#
# Usage:
#   bash scripts/make_portable.sh                # default destination
#   bash scripts/make_portable.sh /tmp/foo       # custom destination
#   bash scripts/make_portable.sh --tar          # also produce .tar.gz
#   bash scripts/make_portable.sh --tar --keep   # keep the staging copy
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEST="${1:-${REPO_ROOT%/}-portable}"
MAKE_TAR=0
KEEP_STAGING=0
for arg in "$@"; do
  case "$arg" in
    --tar) MAKE_TAR=1 ;;
    --keep) KEEP_STAGING=1 ;;
  esac
done

# Strip a trailing slash / --tar / --keep accidentally passed as positional
DEST="${DEST%/}"
[[ "$DEST" == "--tar" || "$DEST" == "--keep" ]] && DEST="${REPO_ROOT%/}-portable"

if [[ "$DEST" == "$REPO_ROOT" ]]; then
  echo "ERROR: destination cannot be the source directory" >&2
  exit 1
fi

echo "→ source: $REPO_ROOT"
echo "→ dest:   $DEST"

rm -rf "$DEST"
mkdir -p "$DEST"

# ---------------------------------------------------------------- rsync ----
# Every path listed under ``--exclude`` is something we explicitly DON'T want
# in the portable copy. rsync's ``/`` suffix means "and everything under it".
rsync -a --info=stats2 --info=progress2 \
  --exclude='.venv/' \
  --exclude='.git/' \
  --exclude='.coverage' \
  --exclude='.coverage.*' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.egg-info/' \
  --exclude='*.egg' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  \
  --exclude='experiments/runs/' \
  --exclude='outputs/' \
  --exclude='memory/' \
  --exclude='vendor/' \
  --exclude='models/' \
  \
  --exclude='data/alimeeting/' \
  --exclude='data/processed/' \
  --exclude='data/raw/' \
  --exclude='data/.cache/' \
  \
  --exclude='*.log' \
  --exclude='*.tmp' \
  --exclude='*.swp' \
  --exclude='*.swo' \
  --exclude='.env' \
  --exclude='.env.local' \
  --exclude='.env.*.local' \
  \
  --exclude='htmlcov/' \
  --exclude='.tox/' \
  --exclude='.nox/' \
  \
  --exclude='experiments/_matrix.log' \
  --exclude='experiments/_reeval.log' \
  --exclude='experiments/_matrix_status.json' \
  \
  "$REPO_ROOT/" "$DEST/"

# ---------------------------------------------- strip macOS fork garbage ---
# macOS writes ``._*`` resource forks; rsync --exclude catches most, but
# belt-and-braces: walk once more and remove any stragglers.
find "$DEST" -name '._*' -type f -delete 2>/dev/null || true

# ---------------------------------------------- report ---------------------
echo
echo "=== portable copy breakdown ==="
du -sh "$DEST" 2>/dev/null
du -sh "$DEST"/* 2>/dev/null | sort -h
echo

# ---------------------------------------------- tarball (optional) -------
if [[ "$MAKE_TAR" == "1" ]]; then
  TARBALL="${DEST}.tar.gz"
  echo "→ creating tarball $TARBALL"
  tar -C "$(dirname "$DEST")" -czf "$TARBALL" "$(basename "$DEST")"
  echo "✓ $(du -h "$TARBALL" | cut -f1)  $TARBALL"
  if [[ "$KEEP_STAGING" != "1" ]]; then
    rm -rf "$DEST"
    echo "(staging copy removed; tarball kept)"
  fi
fi

echo
echo "Done."
