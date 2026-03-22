#!/bin/bash
#SBATCH --job-name=full_commonsense_qa_dataset5_mistral7b_250q_20260320_seas
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/commonsense_qa_dataset5_250q_20260320_seas/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/commonsense_qa_dataset5_250q_20260320_seas/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
RUN_NAME="${RUN_NAME:-full_commonsense_qa_dataset5_mistral7b_250q_20260320_l32_seas}"

exec "$REPO_DIR/jobs/sycophancy_bias_probe/aqua_mc_seas_common.sh" \
  --model "mistralai/Mistral-7B-Instruct-v0.2" \
  --dataset_name commonsense_qa \
  --ays_mc_datasets commonsense_qa \
  --max_questions 250 \
  --split_seed 5 \
  --seed 5 \
  --probe_seed 5 \
  --probe_layer_min 1 \
  --probe_layer_max 32 \
  --run_name "$RUN_NAME" \
  "$@"
