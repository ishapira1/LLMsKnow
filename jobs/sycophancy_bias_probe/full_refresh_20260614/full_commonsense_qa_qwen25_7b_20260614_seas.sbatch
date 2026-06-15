#!/bin/bash
#SBATCH --job-name=full_csqa_qwen25_7b_20260614
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/full_refresh_20260614/commonsense_qa/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/full_refresh_20260614/commonsense_qa/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
RUN_NAME="${RUN_NAME:-full_commonsense_qa_qwen25_7b_20260614_allq_fulldepth_seas}"

exec bash "$REPO_DIR/jobs/sycophancy_bias_probe/full_refresh_20260614/run_full_refresh_common.sh" \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_name commonsense_qa \
  --ays_mc_datasets commonsense_qa \
  --run_name "$RUN_NAME" \
  "$@"
