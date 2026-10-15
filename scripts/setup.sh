#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Educational environment adaptation for Google Research Ravens.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for this tested environment"'
if [ ! -e .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
if [ ! -x .venv/bin/python ]; then
  echo 'The existing .venv is not a Linux Python virtual environment.' >&2
  exit 1
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing .venv must use Python 3.10"'
mkdir -p logs results
# No sudo, global pip config, CUDA installation, or proxy changes.
# PIP_INDEX_URL may be supplied by the caller if a mirror is required.
.venv/bin/python -m pip install 'pip==24.3.1' 'setuptools==69.5.1' 'wheel==0.48.0'
.venv/bin/python -m pip install -r requirements.lock.txt 2>&1 | tee logs/install.log
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > results/installed-requirements.txt
echo 'Environment ready. Next: bash scripts/fetch_source.sh'
