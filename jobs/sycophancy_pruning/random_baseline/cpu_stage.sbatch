#!/usr/bin/env bash
#SBATCH --job-name=rb_cpu_20260803
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -Eeuo pipefail
if [[ -n "${RANDOM_BASELINE_BUNDLE_DIR:-}" ]]; then
  source "$RANDOM_BASELINE_BUNDLE_DIR/common.sh"
elif [[ -n "${REPO_DIR:-}" ]]; then
  source "$REPO_DIR/jobs/sycophancy_pruning/random_baseline/common.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

STAGE="${STAGE:?Set STAGE}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODEL_OVERRIDE:-campaign}"
if [[ -z "${MODEL_OVERRIDE:-}" && ( "$STAGE" == audit_masks || "$STAGE" == aggregate_core ) ]]; then
  MODEL="llama"
  if [[ "$TASK_ID" -eq 1 ]]; then MODEL="qwen"; fi
fi
start_task_log "$STAGE" "$MODEL" "$STAGE"

on_exit() {
  local status="$?"
  finish_task_log "$status"
  if [[ "$status" -ne 0 ]]; then failure_email "$status" "$STAGE" "$MODEL"; fi
  exit "$status"
}
trap on_exit EXIT

case "$STAGE" in
  preflight)
    command=(bash "$BUNDLE_DIR/preflight_internal.sh")
    ;;
  audit_masks)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/random_baseline.py" audit-masks
      --model "$MODEL" --result-root "$RESULT_ROOT")
    ;;
  audit_complete)
    command=(send_progress_email mask_audit_complete
      "[random_baseline] mask audit complete"
      "All 80 controls passed count, uniqueness, disjointness, module, bin, and hash audits.")
    ;;
  aggregate_core)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/random_baseline.py" aggregate-core
      --model "$MODEL" --result-root "$RESULT_ROOT")
    ;;
  llama_core_email|qwen_core_email)
    MODEL="${STAGE%%_*}"
    command=(send_progress_email "${MODEL}_core_complete"
      "[random_baseline] $MODEL core complete"
      "$MODEL core aggregation and confirmatory inference completed.")
    ;;
  broad_complete)
    command=(send_progress_email broad_suite_complete
      "[random_baseline] broad suite complete"
      "Both models completed all predeclared broad-suite states and benchmarks.")
    ;;
  package)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/package_feedback.py" package
      --result-root "$RESULT_ROOT")
    ;;
  final)
    labels="$RESULT_ROOT/judging/feedback_labels.jsonl"
    command=(bash "$BUNDLE_DIR/finalize_internal.sh" "$labels")
    ;;
  *) printf 'Unknown CPU stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
print_command "${command[@]}"
"${command[@]}"
