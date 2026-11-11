#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3.12}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32'
if [ -e .venv ]; then echo 'Existing .venv; use it or a fresh package directory.' >&2; exit 1; fi
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name ins-revision --display-name 'INS Revision (Python 3.12)'
