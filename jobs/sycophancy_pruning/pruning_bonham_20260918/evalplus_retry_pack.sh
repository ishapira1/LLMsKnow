#!/usr/bin/env bash
set -Eeuo pipefail

: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
: "${TASK_IDS:?TASK_IDS is required}"

LANE_COUNT="${LANE_COUNT:-4}"
[[ "$LANE_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  printf 'LANE_COUNT must be a positive integer: %s\n' "$LANE_COUNT" >&2
  exit 2
}

IFS=: read -r -a task_ids <<< "$TASK_IDS"
(( ${#task_ids[@]} > 0 )) || {
  printf 'TASK_IDS must contain at least one task id\n' >&2
  exit 2
}
for task_id in "${task_ids[@]}"; do
  [[ "$task_id" =~ ^[0-9]+$ ]] && (( task_id < 192 )) || {
    printf 'Invalid EvalPlus task id: %s\n' "$task_id" >&2
    exit 2
  }
done

run_lane() {
  local lane="$1" position task_id
  for ((position = lane; position < ${#task_ids[@]}; position += LANE_COUNT)); do
    task_id="${task_ids[$position]}"
    printf 'evalplus_retry_task_start lane=%s task_id=%s time=%s\n' \
      "$lane" "$task_id" "$(date -Is)"
    env \
      STAGE=evalplus_run \
      MODEL_KEY=shared \
      BONHAM_BUNDLE_DIR="$BONHAM_BUNDLE_DIR" \
      SLURM_ARRAY_TASK_ID="$task_id" \
      EVALPLUS_PARALLEL_WORKERS="${EVALPLUS_PARALLEL_WORKERS:-4}" \
      OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
      MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
      bash "$BONHAM_BUNDLE_DIR/cpu_stage.sbatch"
    printf 'evalplus_retry_task_complete lane=%s task_id=%s time=%s\n' \
      "$lane" "$task_id" "$(date -Is)"
  done
}

children=()
for ((lane = 0; lane < LANE_COUNT && lane < ${#task_ids[@]}; lane++)); do
  run_lane "$lane" &
  children+=("$!")
done

status=0
for child in "${children[@]}"; do
  if ! wait "$child"; then
    status=1
  fi
done
exit "$status"
