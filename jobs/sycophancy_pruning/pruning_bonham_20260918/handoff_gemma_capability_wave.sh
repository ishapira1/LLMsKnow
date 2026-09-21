#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"

CURRENT_JOB_ID="${BONHAM_CURRENT_GEMMA_JOB_ID:?BONHAM_CURRENT_GEMMA_JOB_ID is required}"
NEXT_JOB_ID="${BONHAM_NEXT_GEMMA_JOB_ID:?BONHAM_NEXT_GEMMA_JOB_ID is required}"
QUEUED_LANE_SPECS="${BONHAM_QUEUED_LANE_SPECS:?BONHAM_QUEUED_LANE_SPECS is required}"
HANDOFF_SHARD="${BONHAM_HANDOFF_SHARD:-38}"
POLL_SECONDS="${POLL_SECONDS:-10}"

[[ "$CURRENT_JOB_ID" =~ ^[0-9]+$ && "$NEXT_JOB_ID" =~ ^[0-9]+$ ]] || {
  printf 'Current and next Gemma job IDs must be numeric\n' >&2
  exit 2
}
[[ "$HANDOFF_SHARD" =~ ^[0-9]+$ && "$HANDOFF_SHARD" -le 40 ]] || {
  printf 'BONHAM_HANDOFF_SHARD must be between 0 and 40\n' >&2
  exit 2
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be positive\n' >&2
  exit 2
}

log() { printf '%s time=%s\n' "$*" "$(date -Is)"; }

scheduler_state() {
  local job_id="$1" state
  state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -1)"
  if [[ -z "$state" ]]; then
    state="$(sacct -X -n -j "$job_id" --parsable2 --format=State 2>/dev/null \
      | cut -d'|' -f1 | head -1)"
  fi
  printf '%s\n' "$state"
}

all_primary_states_have_shard() {
  local shard="$1" state shard_path
  shard_path="$(printf 'shard_%04d' "$shard")"
  for state in unpruned n1_mechanism random_n1 random_n2; do
    [[ -f "$RESULT_ROOT/evaluations/results/gemma4_12b/$state/capabilities/$shard_path/COMPLETE" ]] \
      || return 1
  done
  return 0
}

validate_specs() {
  local spec state_id lane_pid
  for spec in $QUEUED_LANE_SPECS; do
    IFS=: read -r state_id lane_pid <<< "$spec"
    [[ -n "$state_id" && "$lane_pid" =~ ^[0-9]+$ ]] || {
      printf 'Invalid queued-lane spec (expected state_id:pid): %s\n' "$spec" >&2
      return 2
    }
  done
}

stop_queued_lanes() {
  srun --overlap --jobid="$CURRENT_JOB_ID" --nodes=1 --ntasks=1 \
    env QUEUED_LANE_SPECS="$QUEUED_LANE_SPECS" bash -c '
      set -Eeuo pipefail
      for spec in $QUEUED_LANE_SPECS; do
        IFS=: read -r state_id lane_pid <<< "$spec"
        args="$(ps -p "$lane_pid" -o args= 2>/dev/null || true)"
        [[ -n "$args" ]] || {
          printf "missing queued lane pid=%s state=%s\n" "$lane_pid" "$state_id" >&2
          exit 3
        }
        [[ "$args" == *"capabilities_${state_id}.out"* ]] || {
          printf "lane output mismatch pid=%s state=%s args=%s\n" \
            "$lane_pid" "$state_id" "$args" >&2
          exit 4
        }
        [[ "$args" == *"evaluation ${state_id} capabilities"* ]] || {
          printf "lane command mismatch pid=%s state=%s args=%s\n" \
            "$lane_pid" "$state_id" "$args" >&2
          exit 5
        }
        if pgrep -P "$lane_pid" >/dev/null 2>&1; then
          printf "refusing to stop active lane pid=%s state=%s\n" \
            "$lane_pid" "$state_id" >&2
          exit 6
        fi
      done
      for spec in $QUEUED_LANE_SPECS; do
        IFS=: read -r state_id lane_pid <<< "$spec"
        kill -TERM "$lane_pid"
        printf "stopped queued lane pid=%s state=%s\n" "$lane_pid" "$state_id"
      done
    '
}

validate_specs
log "handoff_supervisor_start current_job_id=$CURRENT_JOB_ID next_job_id=$NEXT_JOB_ID handoff_shard=$HANDOFF_SHARD specs=$QUEUED_LANE_SPECS"

while ! all_primary_states_have_shard "$HANDOFF_SHARD"; do
  current_state="$(scheduler_state "$CURRENT_JOB_ID")"
  next_state="$(scheduler_state "$NEXT_JOB_ID")"
  case "$current_state" in
    RUNNING|CONFIGURING|COMPLETING|REQUEUED) ;;
    *)
      log "current_job_not_active job_id=$CURRENT_JOB_ID scheduler_state=$current_state"
      exit 1
      ;;
  esac
  case "$next_state" in
    PENDING|CONFIGURING|REQUEUED) ;;
    *)
      log "next_job_unexpected_before_handoff job_id=$NEXT_JOB_ID scheduler_state=$next_state"
      exit 1
      ;;
  esac
  sleep "$POLL_SECONDS"
done

log "handoff_threshold_complete shard=$HANDOFF_SHARD"
stop_queued_lanes
log 'queued_second_wave_lanes_stopped=1'

while ! all_primary_states_have_shard 40; do
  current_state="$(scheduler_state "$CURRENT_JOB_ID")"
  case "$current_state" in
    RUNNING|CONFIGURING|COMPLETING|REQUEUED) ;;
    *)
      log "current_job_ended_before_terminal_receipts scheduler_state=$current_state"
      exit 1
      ;;
  esac
  sleep "$POLL_SECONDS"
done

log 'primary_capability_wave_complete=1'
current_state="$(scheduler_state "$CURRENT_JOB_ID")"
case "$current_state" in
  RUNNING|CONFIGURING|COMPLETING|REQUEUED)
    scancel "$CURRENT_JOB_ID" 2>/dev/null || true
    log "released_current_allocation job_id=$CURRENT_JOB_ID"
    ;;
esac

while true; do
  next_state="$(scheduler_state "$NEXT_JOB_ID")"
  case "$next_state" in
    RUNNING|CONFIGURING|COMPLETING|COMPLETED)
      log "next_wave_started job_id=$NEXT_JOB_ID scheduler_state=$next_state"
      break
      ;;
    PENDING|REQUEUED|'') sleep "$POLL_SECONDS" ;;
    *)
      log "next_wave_failed_to_start job_id=$NEXT_JOB_ID scheduler_state=$next_state"
      exit 1
      ;;
  esac
done

log 'handoff_supervisor_complete=1'
