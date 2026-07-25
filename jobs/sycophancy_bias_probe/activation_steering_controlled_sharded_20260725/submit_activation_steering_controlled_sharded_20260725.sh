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
FULL_GATE_APPROVAL="${ACTIVATION_STEERING_FULL_GATE_APPROVAL:-}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_LOG_ROOT="${ACTIVATION_STEERING_SUBMIT_LOG_ROOT:-$REPO_DIR/jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725}"
SUBMIT_LOG_DIR="$SUBMIT_LOG_ROOT/submit"

test -s "$CONFIG"
test -s run_activation_steering.py
scripts=(
  "$BUNDLE/validate_sources_array.sbatch"
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
  log_printf '[dry-run] sbatch --dependency=afterok:<validate> %s\n' "${scripts[1]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<validate> %s\n' "${scripts[2]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<fit_pooled> %s\n' "${scripts[3]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<screen> %s\n' "${scripts[4]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[5]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[6]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[7]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<fit_dataset>:<select> %s\n' "${scripts[8]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[9]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[10]}"
  log_printf '[dry-run] sbatch --dependency=afterok:<dose>:<test>:<probe>:<transfer>:<geometry>:<alpaca> %s\n' "${scripts[11]}"
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
if [[ -z "$FULL_GATE_APPROVAL" || ! -s "$FULL_GATE_APPROVAL" ]]; then
  log_error '%s\n' \
    "Missing hash-bound researcher approval; set ACTIVATION_STEERING_FULL_GATE_APPROVAL."
  exit 2
fi
python3 scripts/validate_activation_steering_full_gate.py \
  --repo-dir "$REPO_DIR" \
  --config "$CONFIG" \
  --question-manifest "$MANIFEST" \
  --alpaca-manifest "$ALPACA_MANIFEST" \
  --inspection-report "$INSPECTION_REPORT" \
  --tiny-compute-report "$TINY_COMPUTE_REPORT" \
  --approval "$FULL_GATE_APPROVAL" \
  --expected-git-commit "$commit"

run_id="${EXPERIMENT_RUN_ID:-}"
if [[ -z "$run_id" || ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  log_error '%s\n' \
    "EXPERIMENT_RUN_ID must be non-empty and contain only letters, digits, dot, underscore, or hyphen."
  exit 2
fi
if [[ -n "${SYCOPHANCY_STORAGE_ROOT_OVERRIDE:-}" ]]; then
  export SYCOPHANCY_STORAGE_ROOT="$SYCOPHANCY_STORAGE_ROOT_OVERRIDE"
fi
source jobs/sycophancy_bias_probe/storage_common.sh
configure_sycophancy_bias_storage "activation_steering_controlled_sharded_20260725"
file_sha256() {
  python3 -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "$1"
}
full_identity_sha256="$(
  python3 -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(b"\0".join(pathlib.Path(p).read_bytes() for p in sys.argv[1:])).hexdigest())' \
    "$CONFIG" "$MANIFEST" "$ALPACA_MANIFEST"
)"
full_identity_hash="${full_identity_sha256:0:16}"
INTERVENTION_BASE_ROOT="${INTERVENTION_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_controlled_20260725}"
INTERVENTION_ROOT="${INTERVENTION_ROOT:-$INTERVENTION_BASE_ROOT/${run_id}_$full_identity_hash}"
sycophancy_bias_reject_home_path "INTERVENTION_ROOT" "$INTERVENTION_ROOT"
mkdir -p "$INTERVENTION_BASE_ROOT"
if ! mkdir "$INTERVENTION_ROOT"; then
  log_error '%s\n' \
    "Refusing overwrite: controlled run root already exists: $INTERVENTION_ROOT"
  exit 2
fi
reservation="$INTERVENTION_ROOT/submission_reservation.env"
printf 'protocol_version=%s\nrun_id=%s\nfull_identity_sha256=%s\ngit_commit=%s\nconfig_sha256=%s\nquestion_manifest_sha256=%s\nalpaca_manifest_sha256=%s\napproval_sha256=%s\ncreated_at=%s\n' \
  "controlled_prompt_only_v1_20260725" \
  "$run_id" \
  "$full_identity_sha256" \
  "$commit" \
  "$(file_sha256 "$CONFIG")" \
  "$(file_sha256 "$MANIFEST")" \
  "$(file_sha256 "$ALPACA_MANIFEST")" \
  "$(file_sha256 "$FULL_GATE_APPROVAL")" \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  >"$reservation"
chmod 0444 "$reservation"
export INTERVENTION_ROOT
log_printf '[submit] reserved_run_root=%s identity_hash=%s\n' \
  "$INTERVENTION_ROOT" "$full_identity_hash"

mkdir -p \
  "$SUBMIT_LOG_ROOT/slurm/validation" \
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
validation_job="$(sbatch --parsable "${scripts[0]}")"
fit_job="$(sbatch --parsable --dependency="afterok:$validation_job" "${scripts[1]}")"
fit_dataset_job="$(sbatch --parsable --dependency="afterok:$validation_job" "${scripts[2]}")"
screen_job="$(sbatch --parsable --dependency="afterok:$fit_job" "${scripts[3]}")"
selection_job="$(sbatch --parsable --dependency="afterok:$screen_job" "${scripts[4]}")"
dose_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[5]}")"
test_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[6]}")"
probe_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[7]}")"
transfer_job="$(sbatch --parsable --dependency="afterok:$fit_dataset_job:$selection_job" "${scripts[8]}")"
geometry_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[9]}")"
alpaca_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[10]}")"
aggregate_job="$(sbatch --parsable --dependency="afterok:$dose_job:$test_job:$probe_job:$transfer_job:$geometry_job:$alpaca_job" "${scripts[11]}")"
log_printf '[submit] validation=%s fit=%s fit_dataset=%s screen=%s selection=%s dose=%s test=%s probe=%s transfer=%s geometry=%s alpaca=%s aggregate=%s\n' \
  "$validation_job" "$fit_job" "$fit_dataset_job" "$screen_job" "$selection_job" "$dose_job" "$test_job" "$probe_job" "$transfer_job" "$geometry_job" "$alpaca_job" "$aggregate_job"
