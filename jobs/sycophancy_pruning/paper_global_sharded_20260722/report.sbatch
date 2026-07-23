#!/bin/bash
#SBATCH --job-name=weight_pruning_report
#SBATCH --partition=shared,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/report/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/report/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
require_slurm_job
setup_environment

REPORT_BOOTSTRAP_SEED="${REPORT_BOOTSTRAP_SEED:-5}"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-$EXPERIMENT_ROOT/reports/minimum_result_package}"
REPORT_OVERWRITE="${REPORT_OVERWRITE:-0}"
if [[ "$REPORT_OVERWRITE" != "0" && "$REPORT_OVERWRITE" != "1" ]]; then
  printf 'REPORT_OVERWRITE must be 0 or 1, got %s\n' "$REPORT_OVERWRITE" >&2
  exit 2
fi
REPORT_SCRIPT="$REPO_DIR/scripts/build_pruning_result_package.py"
require_file "$REPORT_SCRIPT"
require_dir "$EXPERIMENT_ROOT"

TASK_LABEL="paper_global_minimum_result_package"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/report/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR" "$(dirname "$REPORT_OUTPUT_DIR")"
exec > >(tee -a "$STRUCTURED_DIR/task_0.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_0.err" >&2)

cmd=(
  "$ENV_PYTHON" "$REPORT_SCRIPT"
  --experiment-root "$EXPERIMENT_ROOT"
  --output-dir "$REPORT_OUTPUT_DIR"
  --bootstrap-seed "$REPORT_BOOTSTRAP_SEED"
)
if [[ "$REPORT_OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
printf -v command_text '%q ' "${cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
printf '[report] experiment_root=%s output_dir=%s bootstrap_seed=%s overwrite=%s\n' \
  "$EXPERIMENT_ROOT" "$REPORT_OUTPUT_DIR" "$REPORT_BOOTSTRAP_SEED" "$REPORT_OVERWRITE"

START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
