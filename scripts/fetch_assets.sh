#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the teaching subset of official Ravens data and pretrained checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/fetch_subset.py --fetch --project-root . \
  --config config/minimal-assets.json 2>&1 | tee logs/fetch_assets.log
"$PYTHON_BIN" scripts/verify_assets.py
