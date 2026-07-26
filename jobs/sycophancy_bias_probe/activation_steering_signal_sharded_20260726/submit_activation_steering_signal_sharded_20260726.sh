#!/usr/bin/env bash

set -euo pipefail

BUNDLE="jobs/sycophancy_bias_probe/activation_steering_signal_sharded_20260726"
CONFIG="${ACTIVATION_STEERING_CONFIG:-configs/experiments/activation_steering_signal_20260726.json}"
MANIFEST="${QUESTION_MANIFEST:-configs/experiments/activation_steering_signal_300_20260726.jsonl}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-activation_steering_signal_20260726_v1}"
export ACTIVATION_STEERING_CONFIG="$CONFIG"
export QUESTION_MANIFEST="$MANIFEST"
export EXPERIMENT_RUN_ID
export SUBMISSION_GIT_COMMIT="${SUBMISSION_GIT_COMMIT:-$(git rev-parse HEAD)}"

scripts=(
  "$BUNDLE/validate_sources_array.sbatch"
  "$BUNDLE/fit_directions_array.sbatch"
  "$BUNDLE/screen_layers_array.sbatch"
  "$BUNDLE/aggregate_dev.sbatch"
  "$BUNDLE/test_selected_array.sbatch"
  "$BUNDLE/fixed_probe_array.sbatch"
  "$BUNDLE/aggregate_signal.sbatch"
)
for script in "${scripts[@]}"; do
  test -s "$script"
  bash -n "$script"
done
test -s "$CONFIG"
test -s "$MANIFEST"
[[ -z "$(git status --porcelain)" ]] || {
  printf '%s\n' "Signal submission requires a clean Git worktree." >&2
  exit 1
}
python3 -c \
  'import json,sys; c=json.load(open(sys.argv[1])); assert c["study_scope"]=="exploratory_benchmark_label_signal_v1_20260726"; rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]; assert len(rows)==300; assert all(r["correct_choice"]!=r["endorsed_choice"] for r in rows); assert all(r["semantic_b_review_status"]=="not_requested_exploratory" for r in rows)' \
  "$CONFIG" "$MANIFEST"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] commit=%s run_id=%s config=%s manifest=%s\n' \
    "$SUBMISSION_GIT_COMMIT" "$EXPERIMENT_RUN_ID" "$CONFIG" "$MANIFEST"
  printf '[dry-run] sbatch %s\n' "${scripts[0]}"
  printf '[dry-run] sbatch --dependency=afterok:<validate> %s\n' "${scripts[1]}"
  printf '[dry-run] sbatch --dependency=afterok:<fit> %s\n' "${scripts[2]}"
  printf '[dry-run] sbatch --dependency=afterok:<screen> %s\n' "${scripts[3]}"
  printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[4]}"
  printf '[dry-run] sbatch --dependency=afterok:<select> %s\n' "${scripts[5]}"
  printf '[dry-run] sbatch --dependency=afterok:<test>:<probe> %s\n' "${scripts[6]}"
  printf '%s\n' "[dry-run] no jobs submitted"
  exit 0
fi

validation_job="$(sbatch --parsable "${scripts[0]}")"
fit_job="$(sbatch --parsable --dependency="afterok:$validation_job" "${scripts[1]}")"
screen_job="$(sbatch --parsable --dependency="afterok:$fit_job" "${scripts[2]}")"
selection_job="$(sbatch --parsable --dependency="afterok:$screen_job" "${scripts[3]}")"
test_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[4]}")"
probe_job="$(sbatch --parsable --dependency="afterok:$selection_job" "${scripts[5]}")"
aggregate_job="$(sbatch --parsable --dependency="afterok:$test_job:$probe_job" "${scripts[6]}")"
printf '[submit] validation=%s fit=%s screen=%s selection=%s test=%s probe=%s aggregate=%s\n' \
  "$validation_job" "$fit_job" "$screen_job" "$selection_job" "$test_job" "$probe_job" "$aggregate_job"
