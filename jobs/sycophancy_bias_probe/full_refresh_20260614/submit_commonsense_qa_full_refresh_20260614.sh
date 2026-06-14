#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p jobs/sycophancy_bias_probe/logs/full_refresh_20260614/commonsense_qa

sbatch jobs/sycophancy_bias_probe/full_refresh_20260614/full_commonsense_qa_llama31_8b_20260614_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_refresh_20260614/full_commonsense_qa_qwen25_7b_20260614_seas.sbatch
