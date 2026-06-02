#!/usr/bin/env bash
# TokenWise — local setup for macOS.
# Run from inside the TokenWise/ directory after copying it to
# "/Users/tpatil/AI Projects/TokenWise".
set -euo pipefail

echo "==> Python version"
python3 --version

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing TokenWise (editable) + dev deps"
pip install --upgrade pip >/dev/null
pip install -e ".[dev]"

echo "==> Running the Phase 0 test suite"
pytest -q

echo
echo "Done. The repo already has git history (see: git log --oneline)."
echo "Activate the env later with:  source .venv/bin/activate"
