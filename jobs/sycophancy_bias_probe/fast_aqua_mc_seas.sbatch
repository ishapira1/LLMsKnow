#!/bin/bash
#SBATCH --job-name=fast_aqua_mc_mistral7b_seas
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/%x.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
RUN_NAME="${RUN_NAME:-fast_aqua_mc_mistral7b_auto_q24_l8_seas}"

exec "$REPO_DIR/jobs/sycophancy_bias_probe/aqua_mc_seas_common.sh" \
  --smoke_test \
  --smoke_questions 24 \
  --probe_layer_min 1 \
  --probe_layer_max 8 \
  --run_name "$RUN_NAME" \
  "$@"
