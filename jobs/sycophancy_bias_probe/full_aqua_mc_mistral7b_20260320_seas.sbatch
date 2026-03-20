#!/bin/bash
#SBATCH --job-name=full_aqua_mc_mistral7b_20260320_seas
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/%x.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
RUN_NAME="${RUN_NAME:-full_aqua_mc_mistral7b_20260320_auto_allq_l32_seas}"

exec "$REPO_DIR/jobs/sycophancy_bias_probe/aqua_mc_seas_common.sh" \
  --model "mistralai/Mistral-7B-Instruct-v0.2" \
  --probe_layer_min 1 \
  --probe_layer_max 32 \
  --run_name "$RUN_NAME" \
  "$@"
