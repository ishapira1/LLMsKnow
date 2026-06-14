#!/usr/bin/env bash
#SBATCH --job-name=syco_prune_smoke_qwen25
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/sycophancy_pruning/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_pruning/logs/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
cd "$REPO_DIR"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export DATASETS_CSV="${DATASETS_CSV:-arc_challenge,commonsense_qa}"
export RUN_NAME="${RUN_NAME:-sycophancy_pruning_smoke_qwen25_two_dataset}"
export SPARSITIES_CSV="${SPARSITIES_CSV:-0,1e-5}"
export MAX_QUESTIONS_PER_DATASET="${MAX_QUESTIONS_PER_DATASET:-8}"
export MAX_CALIBRATION_RECORDS="${MAX_CALIBRATION_RECORDS:-16}"
export MAX_PRESERVATION_RECORDS="${MAX_PRESERVATION_RECORDS:-32}"
export MAX_EVAL_RECORDS="${MAX_EVAL_RECORDS:-48}"
export WRONG_CONTROL_MIN_EXAMPLES="${WRONG_CONTROL_MIN_EXAMPLES:-50}"

exec bash jobs/sycophancy_pruning/run_common.sh "$@"
