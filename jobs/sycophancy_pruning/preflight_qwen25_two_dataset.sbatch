#!/usr/bin/env bash
#SBATCH --job-name=syco_prune_preflight_qwen25
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/sycophancy_pruning/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_pruning/logs/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
cd "$REPO_DIR"

export RUN_MODE=preflight
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export DATASETS_CSV="${DATASETS_CSV:-arc_challenge,commonsense_qa}"
export RUN_NAME="${RUN_NAME:-sycophancy_pruning_preflight_qwen25_two_dataset}"
export DEVICE="${DEVICE:-cuda}"
export TORCH_DTYPE="${TORCH_DTYPE:-auto}"
export PREFLIGHT_SAMPLE_PER_DATASET="${PREFLIGHT_SAMPLE_PER_DATASET:-4}"
export PREFLIGHT_KNOWN_SOURCE_EXAMPLE_IDS="${PREFLIGHT_KNOWN_SOURCE_EXAMPLE_IDS:-Mercury_7081270}"

exec bash jobs/sycophancy_pruning/run_common.sh "$@"
