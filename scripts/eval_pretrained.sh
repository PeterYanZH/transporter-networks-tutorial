#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the official Ravens/Transporter policy on the three-scene tutorial subset.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
test -x "$PYTHON"
mkdir -p logs results
"$PYTHON" scripts/verify_assets.py
"$PYTHON" upstream/ravens/test.py \
  --root_dir="$PROJECT_ROOT" --data_dir="$PROJECT_ROOT/data" \
  --assets_root="$PROJECT_ROOT/upstream/ravens/environments/assets" \
  --task=block-insertion --agent=transporter \
  --n_demos=10 --n_steps=10000 --n_runs=1 \
  2>&1 | tee logs/eval_pretrained.log
"$PYTHON" scripts/summarize_eval.py
