#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p jobs/sycophancy_bias_probe/logs

SBATCH_CMD=(
  sbatch
  jobs/sycophancy_bias_probe/probe_displacement_mini_qwen25_7b_20260602_seas.sbatch
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[submit] dry run: '
  printf '%q ' "${SBATCH_CMD[@]}" "$@"
  printf '\n'
  exit 0
fi

"${SBATCH_CMD[@]}" "$@"
