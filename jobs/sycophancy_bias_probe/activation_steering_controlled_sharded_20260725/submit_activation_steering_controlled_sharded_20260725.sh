#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"
BUNDLE="jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725"
CONFIG="${ACTIVATION_STEERING_CONFIG:-configs/experiments/activation_steering_controlled_20260725.json}"
MANIFEST="${QUESTION_MANIFEST:-configs/experiments/activation_steering_audited_1000_20260725.jsonl}"
ALPACA_MANIFEST="${ALPACA_UTILITY_MANIFEST:-jobs/sycophancy_pruning/paper_global_sharded_20260722/evaluation/alpaca_utility.jsonl}"
INSPECTION_REPORT="${ACTIVATION_STEERING_INSPECTION_REPORT:-}"
TINY_COMPUTE_REPORT="${ACTIVATION_STEERING_TINY_COMPUTE_REPORT:-}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_LOG_ROOT="${ACTIVATION_STEERING_SUBMIT_LOG_ROOT:-$REPO_DIR/jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725}"
SUBMIT_LOG_DIR="$SUBMIT_LOG_ROOT/submit"

test -s "$CONFIG"
test -s run_activation_steering.py
scripts=(
  "$BUNDLE/fit_directions_array.sbatch"
  "$BUNDLE/fit_dataset_directions_array.sbatch"
  "$BUNDLE/screen_layers_array.sbatch"
  "$BUNDLE/aggregate_dev.sbatch"
  "$BUNDLE/selected_dose_array.sbatch"
  "$BUNDLE/selected_test_array.sbatch"
  "$BUNDLE/fixed_probe_array.sbatch"
  "$BUNDLE/cross_dataset_transfer_array.sbatch"
  "$BUNDLE/geometry_array.sbatch"
  "$BUNDLE/alpaca_guardrail_array.sbatch"
  "$BUNDLE/aggregate_test.sbatch"
)
for script in "$BUNDLE/common.sh" "$BUNDLE/submit_activation_steering_controlled_sharded_20260725.sh" "${scripts[@]}"; do
  bash -n "$script"
done

commit="$(git rev-parse HEAD)"
export SUBMISSION_GIT_COMMIT="$commit"
mkdir -p "$SUBMIT_LOG_DIR"
submit_log="$SUBMIT_LOG_DIR/submit_${EXPERIMENT_RUN_ID:-controlled}_$(date '+%Y%m%dT%H%M%S')_$$.log"
exec 3>>"$submit_log"
log_printf() {
  printf "$@"
  printf "$@" >&3
}
log_error() {
  printf "$@" >&2
  printf "$@" >&3
}
log_printf '[submit] dry_run=%s commit=%s config=%s manifest=%s task_filter=%s\n' \
  "$DRY_RUN" "$commit" "$CONFIG" "$MANIFEST" "${TASK_FILTER:-}"

if [[ "$DRY_RUN" == "1" ]]; then
  log_printf '[dry-run] sbatch %s\n' "${scripts[0]}"
  log_printf '[dry-run] sbatch %s\n' "${scripts[1]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<fit_pooled> %s\n' "${scripts[2]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<screen> %s\n' "${scripts[3]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[4]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[5]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[6]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<fit_dataset>:<select> %s\n' "${scripts[7]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[8]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[9]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<dose>:<test>:<probe>:<transfer>:<geometry>:<alpaca> %s\n' "${scripts[10]}"
  log_printf '%s\n' "[dry-run] no jobs submitted"
  exit 0
fi

if [[ "${ALLOW_FULL_SUBMISSION:-0}" != "1" ]]; then
  log_error '%s\n' \
    "Full controlled steering is blocked. Set ALLOW_FULL_SUBMISSION=1 only after the approved manifest, inspection, tiny run, numeric gates, and diff review pass."
  exit 2
fi
if [[ ! -s "$MANIFEST" ]]; then
  log_error '%s\n' "Missing approved audited manifest: $MANIFEST"
  exit 2
fi
python3 scripts/validate_activation_steering_manifest.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --require-full-cohort
if [[ ! -s "$ALPACA_MANIFEST" ]]; then
  log_error '%s\n' "Missing fixed disjoint Alpaca utility manifest: $ALPACA_MANIFEST"
  exit 2
fi
if [[ -z "$INSPECTION_REPORT" || ! -s "$INSPECTION_REPORT" ]]; then
  log_error '%s\n' \
    "Missing approved eight-example GPU inspection report; set ACTIVATION_STEERING_INSPECTION_REPORT."
  exit 2
fi
if [[ -z "$TINY_COMPUTE_REPORT" || ! -s "$TINY_COMPUTE_REPORT" ]]; then
  log_error '%s\n' \
    "Missing reviewed tiny-run compute projection; set ACTIVATION_STEERING_TINY_COMPUTE_REPORT."
  exit 2
fi

mkdir -p \
  "$SUBMIT_LOG_ROOT/slurm/fit" \
  "$SUBMIT_LOG_ROOT/slurm/screen" \
  "$SUBMIT_LOG_ROOT/slurm/selection" \
  "$SUBMIT_LOG_ROOT/slurm/dose" \
  "$SUBMIT_LOG_ROOT/slurm/test" \
  "$SUBMIT_LOG_ROOT/slurm/probe" \
  "$SUBMIT_LOG_ROOT/slurm/transfer" \
  "$SUBMIT_LOG_ROOT/slurm/geometry" \
  "$SUBMIT_LOG_ROOT/slurm/alpaca" \
  "$SUBMIT_LOG_ROOT/slurm/aggregate" \
  "$SUBMIT_LOG_ROOT/by_task"

export ACTIVATION_STEERING_CONFIG="$CONFIG"
export QUESTION_MANIFEST="$MANIFEST"
export ALPACA_UTILITY_MANIFEST="$ALPACA_MANIFEST"
fit_job="$(sbatch --parsable "${scripts[0]}")"
fit_dataset_job="$(sbatch --parsable "${scripts[1]}")"
screen_job="$(sbatch --parsable --dependency="afterok:$fit_job" "${scripts[2]}")"
selection_job="$(sbatch --parsable --dependency="afterok:$screen_job" "${scripts[3]}")"
dose_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[4]}")"
test_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[5]}")"
probe_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[6]}")"
transfer_job="$(sbatch --parsable --dependency="afterok:$fit_dataset_job:$selection_job" "${scripts[7]}")"
geometry_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[8]}")"
alpaca_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[9]}")"
aggregate_job="$(sbatch --parsable --dependency="afterok:$dose_job:$test_job:$probe_job:$transfer_job:$geometry_job:$alpaca_job" "${scripts[10]}")"
log_printf '[submit] fit=%s fit_dataset=%s screen=%s selection=%s dose=%s test=%s probe=%s transfer=%s geometry=%s alpaca=%s aggregate=%s\n' \
  "$fit_job" "$fit_dataset_job" "$screen_job" "$selection_job" "$dose_job" "$test_job" "$probe_job" "$transfer_job" "$geometry_job" "$alpaca_job" "$aggregate_job"
