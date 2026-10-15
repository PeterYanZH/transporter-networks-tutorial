#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Educational tooling for Google Research Ravens / Transporter Networks.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
mkdir -p logs results
test -x "$PROJECT_ROOT/.venv/bin/python"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/check_environment.py" 2>&1 | tee logs/check_environment.log
