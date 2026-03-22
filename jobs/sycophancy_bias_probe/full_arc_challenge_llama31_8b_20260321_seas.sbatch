#!/bin/bash
#SBATCH --job-name=full_arc_challenge_llama31_8b_20260321_seas
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/%x.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
RUN_NAME="${RUN_NAME:-full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas}"

exec "$REPO_DIR/jobs/sycophancy_bias_probe/aqua_mc_seas_common.sh" \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --dataset_name arc_challenge \
  --ays_mc_datasets arc_challenge \
  --split_seed 5 \
  --seed 5 \
  --probe_seed 5 \
  --probe_layer_min 1 \
  --probe_layer_max 999 \
  --run_name "$RUN_NAME" \
  "$@"
