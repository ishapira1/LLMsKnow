#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_DIR"
BUNDLE="jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726"
LOG_ROOT="$REPO_DIR/jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726"
mkdir -p \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/cohort" \
  "$LOG_ROOT/slurm/bf16" \
  "$LOG_ROOT/slurm/projection" \
  "$LOG_ROOT/slurm/validation" \
  "$LOG_ROOT/slurm/selection" \
  "$LOG_ROOT/slurm/test" \
  "$LOG_ROOT/slurm/control" \
  "$LOG_ROOT/slurm/sensitivity" \
  "$LOG_ROOT/slurm/aggregate"
STAGE="${STAGE:-}"
DEPENDENCY="${DEPENDENCY:-}"
case "$STAGE" in
  cohort) script="$BUNDLE/build_cohorts.sbatch" ;;
  bf16) script="$BUNDLE/bf16_gate_array.sbatch" ;;
  projection) script="$BUNDLE/project_compute.sbatch" ;;
  validation) script="$BUNDLE/validation_array.sbatch" ;;
  selection) script="$BUNDLE/select_validation.sbatch" ;;
  test) script="$BUNDLE/test_learned_array.sbatch" ;;
  controls) script="$BUNDLE/test_controls_array.sbatch" ;;
  sensitivity) script="$BUNDLE/suffix_sensitivity_array.sbatch" ;;
  aggregate) script="$BUNDLE/aggregate_test.sbatch" ;;
  *)
    printf 'Set STAGE to one of: cohort bf16 projection validation selection test controls sensitivity aggregate\n' >&2
    exit 2
    ;;
esac

for check_script in "$BUNDLE"/*.sh "$BUNDLE"/*.sbatch; do
  bash -n "$check_script"
done
SUBMISSION_GIT_COMMIT="$(git rev-parse HEAD)"
export SUBMISSION_GIT_COMMIT REPO_DIR
dependency_args=()
if [[ -n "$DEPENDENCY" ]]; then
  dependency_args=(--dependency="afterok:$DEPENDENCY")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN=1: validated Stage-B %s at commit %s\n' \
    "$STAGE" "$SUBMISSION_GIT_COMMIT"
  printf 'Would submit: sbatch %s --export=ALL,SUBMISSION_GIT_COMMIT=%q %q\n' \
    "${dependency_args[*]:-}" "$SUBMISSION_GIT_COMMIT" "$script"
  exit 0
fi
job_id="$(
  sbatch \
    --parsable \
    "${dependency_args[@]}" \
    --export="ALL,SUBMISSION_GIT_COMMIT=$SUBMISSION_GIT_COMMIT" \
    "$script"
)"
printf 'Submitted conditioned Stage-B %s job %s at commit %s\n' \
  "$STAGE" "$job_id" "$SUBMISSION_GIT_COMMIT" | tee \
  "$LOG_ROOT/submit/${STAGE}_submission_${job_id}.log"
