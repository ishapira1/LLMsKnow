#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p jobs/sycophancy_bias_probe/logs

sbatch jobs/sycophancy_bias_probe/full_arc_challenge_qwen25_7b_20260322_seas.sbatch
