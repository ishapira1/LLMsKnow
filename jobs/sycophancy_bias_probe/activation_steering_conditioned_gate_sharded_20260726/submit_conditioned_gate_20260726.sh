#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_DIR"
BUNDLE="jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726"
LOG_ROOT="$REPO_DIR/jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726"
mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/audit"

SUBMISSION_GIT_COMMIT="$(git rev-parse HEAD)"
export SUBMISSION_GIT_COMMIT
export REPO_DIR

scripts=("$BUNDLE/stage_a_audit.sbatch")
for script in "$BUNDLE/common.sh" "$BUNDLE/submit_conditioned_gate_20260726.sh" "${scripts[@]}"; do
  bash -n "$script"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN=1: validated conditioned-gate submitter at commit %s\n' \
    "$SUBMISSION_GIT_COMMIT"
  printf 'Would submit: sbatch --export=ALL,SUBMISSION_GIT_COMMIT=%q %q\n' \
    "$SUBMISSION_GIT_COMMIT" "${scripts[0]}"
  exit 0
fi

job_id="$(
  sbatch \
    --parsable \
    --export="ALL,SUBMISSION_GIT_COMMIT=$SUBMISSION_GIT_COMMIT" \
    "${scripts[0]}"
)"
printf 'Submitted Stage-A mean-cancellation audit job %s\n' "$job_id" | tee \
  "$LOG_ROOT/submit/submission_${job_id}.log"
