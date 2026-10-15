#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Educational tooling for Google Research Ravens / Transporter Networks.
# Source this file; all paths and dependencies remain inside this checkout.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/upstream${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1
export TF_ADDONS_PY_OPS=1
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TF_CPP_MIN_LOG_LEVEL=2
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT_ROOT"
