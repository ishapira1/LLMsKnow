#!/bin/bash
#SBATCH --job-name=full_commonsense_qa_dataset5_gpt54mini_250q_20260320_seas
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/commonsense_qa_dataset5_250q_20260320_seas/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/commonsense_qa_dataset5_250q_20260320_seas/%x.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
export MODEL_ID="${MODEL_ID:-gpt-5.4-mini}"
export DATASET_NAME="${DATASET_NAME:-commonsense_qa}"
export AYS_MC_DATASETS="${AYS_MC_DATASETS:-commonsense_qa}"
export MAX_QUESTIONS="${MAX_QUESTIONS:-250}"
export RUN_NAME="${RUN_NAME:-full_commonsense_qa_dataset5_gpt54mini_250q_20260320_samples_seas}"

exec "$REPO_DIR/jobs/sycophancy_bias_probe/full_aqua_mc_gpt54nano_samples.sh" \
  --split_seed 5 \
  --seed 5 \
  "$@"
