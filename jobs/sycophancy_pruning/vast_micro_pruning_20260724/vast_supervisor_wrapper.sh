#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/logging.sh ""
source /opt/supervisor-scripts/utils/environment.sh

export PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
export REPO_DIR="${REPO_DIR:-/workspace/LLMsKnow}"
exec /bin/bash \
  "$REPO_DIR/jobs/sycophancy_pruning/vast_micro_pruning_20260724/run_vast_micro_pruning.sh"
