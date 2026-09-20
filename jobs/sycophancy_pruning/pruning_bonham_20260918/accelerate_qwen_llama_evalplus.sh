#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

POLL_SECONDS="${POLL_SECONDS:-30}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
USER_NAME="${USER:-ishapira}"
ACCOUNTING_START="${BONHAM_ACCOUNTING_START:-2026-09-19}"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
}

mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/evalplus_prepare_qwen_llama" \
  "$LOG_ROOT/slurm/evalplus_run" "$LOG_ROOT/slurm/evalplus_aggregate_qwen_llama" \
  "$LOG_ROOT/slurm/qwen_llama_complete_report"
supervisor_log="$LOG_ROOT/submit/qwen_llama_evalplus_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$supervisor_log") 2>&1

log() { printf '%s time=%s\n' "$*" "$(date -Is)" >&2; }

job_state() {
  local job_id="$1"
  sacct -X -n -j "$job_id" --parsable2 --format=State 2>/dev/null \
    | cut -d'|' -f1 | head -1
}

latest_job_by_name() {
  local name="$1"
  {
    squeue -h -u "$USER_NAME" -n "$name" -o '%A' 2>/dev/null || true
    sacct -S "$ACCOUNTING_START" -X -n --name "$name" \
      --parsable2 --format=JobIDRaw 2>/dev/null | cut -d'|' -f1 || true
  } | grep -E '^[0-9]+$' | sort -n | tail -1 || true
}

wait_job() {
  local label="$1" job_id="$2" state
  while true; do
    state="$(job_state "$job_id")"
    case "$state" in
      COMPLETED)
        log "stage_complete label=$label job_id=$job_id"
        return 0
        ;;
      PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|'') ;;
      *)
        log "stage_failure label=$label job_id=$job_id state=$state"
        return 1
        ;;
    esac
    sleep "$POLL_SECONDS"
  done
}

capabilities_complete() {
  local model="$1" index_path shard_count last_shard state
  local -a states=(
    unpruned n1_mechanism n2_selective random_n1
    random_n2 weak_prompt strong_prompt prompt_only_meandiff
  )
  index_path="$RESULT_ROOT/evaluations/inputs/$model/capabilities/index.jsonl"
  [[ -s "$index_path" ]] || return 1
  shard_count="$(wc -l < "$index_path" | tr -d ' ')"
  [[ "$shard_count" =~ ^[1-9][0-9]*$ ]] || return 1
  last_shard="$((shard_count - 1))"
  for state in "${states[@]}"; do
    [[ -f "$RESULT_ROOT/evaluations/results/$model/$state/capabilities/$(printf 'shard_%04d' "$last_shard")/COMPLETE" ]] || return 1
  done
}

wait_for_capabilities() {
  while ! capabilities_complete qwen25_7b || ! capabilities_complete llama31_8b; do
    log 'waiting_for_qwen_llama_capabilities=1'
    sleep "$POLL_SECONDS"
  done
  log 'qwen_llama_capabilities_complete=1'
}

wait_for_weight_analysis() {
  while [[ ! -f "$RESULT_ROOT/weight_analysis/qwen25_7b/COMPLETE.json" ]] || \
        [[ ! -f "$RESULT_ROOT/weight_analysis/llama31_8b/COMPLETE.json" ]]; do
    log 'waiting_for_qwen_llama_weight_analysis=1'
    sleep "$POLL_SECONDS"
  done
  log 'qwen_llama_weight_analysis_complete=1'
}

evalplus_missing_tasks() {
  local task state_index within model state benchmark shard receipt
  local -a states=(
    unpruned n1_mechanism n2_selective random_n1
    random_n2 weak_prompt strong_prompt prompt_only_meandiff
  )
  local -a missing=()
  for task in $(seq 0 127); do
    state_index="$((task / 8))"
    within="$((task % 8))"
    if ((state_index < 8)); then
      model=llama31_8b
    else
      model=qwen25_7b
    fi
    state="${states[$((state_index % 8))]}"
    if ((within < 4)); then
      benchmark=humaneval
      shard="$within"
    else
      benchmark=mbpp
      shard="$((within - 4))"
    fi
    receipt="$RESULT_ROOT/evalplus/shards/$model/$state/$benchmark/$(printf 'shard_%04d' "$shard")/COMPLETE.json"
    [[ -f "$receipt" ]] || missing+=("$task")
  done
  local IFS=,
  printf '%s' "${missing[*]}"
}

submit_cpu_job() {
  local name="$1" stage="$2" time_limit="$3" memory="$4" array_spec="${5:-}"
  local existing state raw job_id
  local -a command
  existing="$(latest_job_by_name "$name")"
  if [[ -n "$existing" ]]; then
    state="$(job_state "$existing")"
    case "$state" in
      PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|COMPLETED)
        log "reuse_job name=$name job_id=$existing state=$state"
        printf '%s\n' "$existing"
        return 0
        ;;
    esac
    name="${name}_$(date +%H%M%S)"
  fi
  command=(
    sbatch --parsable --account="$ACCOUNT" --partition=serial_requeue
    --job-name="$name" --time="$time_limit" --mem="$memory"
    --export="ALL,STAGE=$stage,MODEL_KEY=shared,BONHAM_BUNDLE_DIR=$BUNDLE_DIR"
    --output="$LOG_ROOT/slurm/$stage/%x_%A_%a.out"
    --error="$LOG_ROOT/slurm/$stage/%x_%A_%a.err"
  )
  [[ -z "$array_spec" ]] || command+=(--array="$array_spec" --cpus-per-task=8)
  command+=("$BUNDLE_DIR/cpu_stage.sbatch")
  raw="$("${command[@]}")"
  job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'Unexpected sbatch response for %s: %s\n' "$name" "$raw" >&2
    return 2
  }
  log "submitted_job name=$name job_id=$job_id stage=$stage array=${array_spec:-none}"
  printf '%s\n' "$job_id"
}

log "qwen_llama_evalplus_supervisor_start commit=$(git -C "$REPO_DIR" rev-parse --short HEAD)"
wait_for_capabilities

prepare_job="$(submit_cpu_job bonh_ql_eprep evalplus_prepare_qwen_llama 01:00:00 96G)"
wait_job evalplus_prepare_qwen_llama "$prepare_job"

# cpu_stage maps array tasks 0--63 to Llama and 64--127 to Qwen.  Resume
# from authenticated shard receipts so a transient array failure never reruns
# successful code executions or leaves the tail permanently blocked.
attempt=0
while missing_tasks="$(evalplus_missing_tasks)" && [[ -n "$missing_tasks" ]]; do
  attempt="$((attempt + 1))"
  if ((attempt > 5)); then
    log "evalplus_recovery_exhausted missing_tasks=$missing_tasks"
    exit 1
  fi
  run_job="$(submit_cpu_job "bonh_ql_eprun_r${attempt}" evalplus_run 03:00:00 24G "${missing_tasks}%40")"
  if ! wait_job evalplus_run_qwen_llama "$run_job"; then
    log "evalplus_retry_required attempt=$attempt job_id=$run_job"
  fi
done
[[ -z "$(evalplus_missing_tasks)" ]] || {
  log 'evalplus_shard_inventory_incomplete=1'
  exit 1
}
log 'evalplus_qwen_llama_shards_complete=128'

aggregate_job="$(submit_cpu_job bonh_ql_epagg evalplus_aggregate_qwen_llama 01:00:00 96G)"
wait_job evalplus_aggregate_qwen_llama "$aggregate_job"

report_job="$(submit_cpu_job bonh_ql_report qwen_llama_complete_report 02:00:00 96G)"
wait_job qwen_llama_complete_report "$report_job"

wait_for_weight_analysis
weight_job="$(submit_cpu_job bonh_ql_weightagg weight_aggregate_qwen_llama 01:00:00 96G)"
wait_job weight_aggregate_qwen_llama "$weight_job"

log "qwen_llama_evalplus_supervisor_complete=1 report=$RESULT_ROOT/reports/qwen_llama_complete/COMPLETE.json weight=$RESULT_ROOT/weight_analysis/COMPLETE_qwen25_7b_llama31_8b.json"
