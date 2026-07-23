#!/bin/bash
#SBATCH --job-name=weight_pruning_prepare
#SBATCH --partition=shared,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/prepare/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/prepare/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
require_slurm_job
setup_environment

TASK_LABEL="weight_pruning_prepare_alpaca"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/prepare/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_0.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_0.err" >&2)

cmd=(
  "$ENV_PYTHON" scripts/prepare_weight_pruning_alpaca.py
  --output "$ALPACA_DATA"
  --hf-datasets-cache "$HF_DATASETS_CACHE"
)
printf -v command_text '%q ' "${cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
