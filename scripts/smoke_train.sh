#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Optional one-update check of the original Ravens/Transporter training pipeline.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
test -x "$PYTHON"
mkdir -p logs results
"$PYTHON" scripts/verify_assets.py
# Each network receives exactly one optimizer update. Do not omit interval=1:
# upstream otherwise performs its default 1000-update block.
# These generated weights are isolated from official pretrained checkpoints.
"$PYTHON" upstream/ravens/train.py \
  --train_dir="$PROJECT_ROOT/smoke_run" --data_dir="$PROJECT_ROOT/data" \
  --task=block-insertion --agent=transporter \
  --n_demos=10 --n_steps=1 --interval=1 --n_runs=1 \
  2>&1 | tee logs/train_smoke.log
"$PYTHON" scripts/verify_smoke.py
