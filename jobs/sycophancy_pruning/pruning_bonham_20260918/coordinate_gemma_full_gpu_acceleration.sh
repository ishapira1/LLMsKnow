#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"

CURRENT_JOB_ID="${BONHAM_CURRENT_GEMMA_JOB_ID:?BONHAM_CURRENT_GEMMA_JOB_ID is required}"
ACCELERATOR_SPECS="${BONHAM_GEMMA_ACCELERATOR_SPECS:?BONHAM_GEMMA_ACCELERATOR_SPECS is required}"
POLL_SECONDS="${POLL_SECONDS:-10}"
CUTOFF_SHARD="${BONHAM_ACCELERATOR_CUTOFF_SHARD:-39}"

[[ "$CURRENT_JOB_ID" =~ ^[0-9]+$ ]] || {
  printf 'BONHAM_CURRENT_GEMMA_JOB_ID must be numeric\n' >&2
  exit 2
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be positive\n' >&2
  exit 2
}
[[ "$CUTOFF_SHARD" =~ ^[0-9]+$ && "$CUTOFF_SHARD" -le 40 ]] || {
  printf 'BONHAM_ACCELERATOR_CUTOFF_SHARD must be between 0 and 40\n' >&2
  exit 2
}

control_root="$RESULT_ROOT/control/gemma_full_gpu_acceleration"
mkdir -p "$control_root"

log() { printf '%s time=%s\n' "$*" "$(date -Is)"; }

job_state() {
  local job_id="$1" state
  state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -1)"
  if [[ -z "$state" ]]; then
    state="$(sacct -X -n -j "$job_id" --parsable2 --format=State 2>/dev/null \
      | cut -d'|' -f1 | head -1)"
  fi
  printf '%s\n' "$state"
}

cutoff_reached() {
  local state shard_path
  shard_path="$(printf 'shard_%04d' "$CUTOFF_SHARD")"
  for state in unpruned n1_mechanism random_n1 random_n2; do
    if [[ -f "$RESULT_ROOT/evaluations/results/gemma4_12b/$state/capabilities/$shard_path/COMPLETE" ]]; then
      return 0
    fi
  done
  return 1
}

transfer_lane_ownership() {
  local state_id="$1" lane_pid="$2"
  srun --overlap --jobid="$CURRENT_JOB_ID" --nodes=1 --ntasks=1 \
    env TARGET_LANE_PID="$lane_pid" TARGET_STATE_ID="$state_id" bash -c '
      set -Eeuo pipefail
      args="$(ps -p "$TARGET_LANE_PID" -o args= 2>/dev/null || true)"
      [[ -n "$args" ]] || {
        printf "missing queued lane pid=%s state=%s\n" "$TARGET_LANE_PID" "$TARGET_STATE_ID" >&2
        exit 3
      }
      [[ "$args" == *"capabilities_${TARGET_STATE_ID}.out"* ]] || {
        printf "lane output mismatch pid=%s state=%s args=%s\n" \
          "$TARGET_LANE_PID" "$TARGET_STATE_ID" "$args" >&2
        exit 4
      }
      [[ "$args" == *"evaluation ${TARGET_STATE_ID} capabilities"* ]] || {
        printf "lane command mismatch pid=%s state=%s args=%s\n" \
          "$TARGET_LANE_PID" "$TARGET_STATE_ID" "$args" >&2
        exit 5
      }
      if pgrep -P "$TARGET_LANE_PID" >/dev/null 2>&1; then
        printf "refusing to stop active lane pid=%s state=%s\n" \
          "$TARGET_LANE_PID" "$TARGET_STATE_ID" >&2
        exit 6
      fi
      kill -TERM "$TARGET_LANE_PID"
    '
}

for spec in $ACCELERATOR_SPECS; do
  IFS=: read -r job_id state_id lane_pid <<< "$spec"
  [[ "$job_id" =~ ^[0-9]+$ && "$lane_pid" =~ ^[0-9]+$ && -n "$state_id" ]] || {
    printf 'Invalid accelerator spec (expected job_id:state_id:lane_pid): %s\n' "$spec" >&2
    exit 2
  }
done

log "coordinator_start current_job_id=$CURRENT_JOB_ID specs=$ACCELERATOR_SPECS cutoff_shard=$CUTOFF_SHARD"
while true; do
  remaining=0
  for spec in $ACCELERATOR_SPECS; do
    IFS=: read -r job_id state_id lane_pid <<< "$spec"
    done_path="$control_root/$job_id.done"
    [[ -f "$done_path" ]] && continue
    remaining=$((remaining + 1))
    state="$(job_state "$job_id")"
    case "$state" in
      RUNNING|CONFIGURING)
        log "claim_start job_id=$job_id state_id=$state_id scheduler_state=$state"
        if transfer_lane_ownership "$state_id" "$lane_pid"; then
          printf 'accelerator_job_id=%s\nstate_id=%s\ncurrent_job_id=%s\ntime=%s\n' \
            "$job_id" "$state_id" "$CURRENT_JOB_ID" "$(date -Is)" > "$done_path"
          log "claim_complete job_id=$job_id state_id=$state_id"
        else
          log "claim_failed_cancel job_id=$job_id state_id=$state_id"
          scancel "$job_id" 2>/dev/null || true
          printf 'claim_failed=1\naccelerator_job_id=%s\nstate_id=%s\ntime=%s\n' \
            "$job_id" "$state_id" "$(date -Is)" > "$done_path"
        fi
        ;;
      PENDING|REQUEUED|RESIZING|SUSPENDED|'')
        if cutoff_reached; then
          log "cutoff_cancel job_id=$job_id state_id=$state_id scheduler_state=${state:-unknown}"
          scancel "$job_id" 2>/dev/null || true
          printf 'cutoff_cancelled=1\naccelerator_job_id=%s\nstate_id=%s\ntime=%s\n' \
            "$job_id" "$state_id" "$(date -Is)" > "$done_path"
        fi
        ;;
      COMPLETED)
        printf 'completed_before_claim=1\naccelerator_job_id=%s\nstate_id=%s\ntime=%s\n' \
          "$job_id" "$state_id" "$(date -Is)" > "$done_path"
        log "completed_before_claim job_id=$job_id state_id=$state_id"
        ;;
      *)
        printf 'terminal_before_claim=%s\naccelerator_job_id=%s\nstate_id=%s\ntime=%s\n' \
          "$state" "$job_id" "$state_id" "$(date -Is)" > "$done_path"
        log "terminal_before_claim job_id=$job_id state_id=$state_id scheduler_state=$state"
        ;;
    esac
  done
  (( remaining == 0 )) && break
  sleep "$POLL_SECONDS"
done
log 'coordinator_complete=1'
