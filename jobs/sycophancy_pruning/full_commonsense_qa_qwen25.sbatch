#!/usr/bin/env bash
#SBATCH --job-name=syco_prune_csqa_qwen25
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=jobs/sycophancy_pruning/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_pruning/logs/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
cd "$REPO_DIR"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export DATASETS_CSV="${DATASETS_CSV:-commonsense_qa}"
export RUN_NAME="${RUN_NAME:-sycophancy_pruning_full_commonsense_qa_qwen25}"
export SPARSITIES_CSV="${SPARSITIES_CSV:-0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3}"

exec bash jobs/sycophancy_pruning/run_common.sh "$@"
