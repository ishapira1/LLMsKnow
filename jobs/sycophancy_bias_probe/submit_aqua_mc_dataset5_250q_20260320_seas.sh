#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

sbatch jobs/sycophancy_bias_probe/full_aqua_mc_dataset5_gpt54mini_250q_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_dataset5_mistral7b_250q_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_dataset5_llama31_8b_250q_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_dataset5_qwen3_4b_250q_20260320_seas.sbatch
