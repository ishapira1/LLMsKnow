#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

POLL_SECONDS="${POLL_SECONDS:-30}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
GPU_PARTITION="${BONHAM_GPU_TEST_PARTITION:-gpu_test}"
GPU_GRES="${BONHAM_GPU_TEST_GRES:-gpu:nvidia_a100_3g.20gb}"
ACCOUNTING_START="${BONHAM_ACCOUNTING_START:-2026-09-19}"
USER_NAME="${USER:-ishapira}"
DEFER_MARKER="$RESULT_ROOT/control/DEFER_QWEN_LLAMA_CAPABILITIES_UNTIL_SOURCE"
CAPABILITY_PRIORITY_MARKER="$RESULT_ROOT/control/PRIORITIZE_QWEN_LLAMA_CAPABILITIES"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
}

mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/gpu_eval_states"
touch "$CAPABILITY_PRIORITY_MARKER"
supervisor_log="$LOG_ROOT/submit/capabilities_after_source_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$supervisor_log") 2>&1

log() { printf '%s time=%s\n' "$*" "$(date -Is)" >&2; }

job_state() {
  local job_id="$1"
  sacct -X -n -j "$job_id" --parsable2 --format=State 2>/dev/null \
    | cut -d'|' -f1 | head -1
}

wait_jobs() {
  local label="$1"
  shift
  local job_id state pending
  while true; do
    pending=0
    for job_id in "$@"; do
      state="$(job_state "$job_id")"
      case "$state" in
        COMPLETED) ;;
        PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|'')
          pending=1
          ;;
        *)
          log "stage_failure label=$label job_id=$job_id state=$state"
          return 1
          ;;
      esac
    done
    if (( pending == 0 )); then
      log "stage_complete label=$label jobs=$*"
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

family_complete() {
  local model="$1" family="$2" index_path shard_count last_shard state
  local -a states=(
    unpruned n1_mechanism n2_selective random_n1
    random_n2 weak_prompt strong_prompt prompt_only_meandiff
  )
  index_path="$RESULT_ROOT/evaluations/inputs/$model/$family/index.jsonl"
  [[ -s "$index_path" ]] || return 1
  shard_count="$(wc -l < "$index_path" | tr -d ' ')"
  [[ "$shard_count" =~ ^[1-9][0-9]*$ ]] || return 1
  last_shard="$((shard_count - 1))"
  for state in "${states[@]}"; do
    [[ -f "$RESULT_ROOT/evaluations/results/$model/$state/$family/$(printf 'shard_%04d' "$last_shard")/COMPLETE" ]] || return 1
  done
}

wait_for_sources() {
  while ! family_complete qwen25_7b source_attribution || \
        ! family_complete llama31_8b source_attribution; do
    log 'waiting_for_qwen_llama_source_attribution=1'
    sleep "$POLL_SECONDS"
  done
  log 'qwen_llama_source_attribution_complete=1'
}

wait_for_gpu_test_clear() {
  local active
  while true; do
    active="$(squeue -h -u "$USER_NAME" -p "$GPU_PARTITION" | wc -l | tr -d ' ')"
    if (( active == 0 )); then return 0; fi
    log "waiting_for_gpu_test_clear active=$active"
    sleep "$POLL_SECONDS"
  done
}

latest_job_by_name() {
  local name="$1"
  {
    squeue -h -u "$USER_NAME" -n "$name" -o '%A' 2>/dev/null || true
    sacct -S "$ACCOUNTING_START" -X -n --name "$name" \
      --parsable2 --format=JobIDRaw 2>/dev/null | cut -d'|' -f1 || true
  } | grep -E '^[0-9]+$' | sort -n | tail -1 || true
}

submit_capability_job() {
  local name="$1" model="$2" offset="$3" existing existing_state raw job_id
  existing="$(latest_job_by_name "$name")"
  if [[ -n "$existing" ]]; then
    existing_state="$(job_state "$existing")"
    case "$existing_state" in
      PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|COMPLETED)
        log "reuse_job name=$name job_id=$existing state=$existing_state"
        printf '%s\n' "$existing"
        return 0
        ;;
    esac
    name="${name}_$(date +%H%M%S)"
  fi
  raw="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$GPU_PARTITION" --job-name="$name" \
    --nodes=1 --ntasks=4 --cpus-per-task=4 --mem=192G --time=12:00:00 \
    --gres="$GPU_GRES:4" \
    --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$model,STATE_OFFSET=$offset,STATE_COUNT=4,GPUS_PER_STATE=1,CPUS_PER_STATE=4,MEM_PER_STATE=48G,EVALUATION_FAMILY_SET=capabilities,EVALUATION_BATCH_SIZE=4" \
    --output="$LOG_ROOT/slurm/gpu_eval_states/%x_%j.out" \
    --error="$LOG_ROOT/slurm/gpu_eval_states/%x_%j.err" \
    "$BUNDLE_DIR/gpu_eval_states.sbatch")"
  job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'Unexpected sbatch response for %s: %s\n' "$name" "$raw" >&2
    return 2
  }
  log "submitted_job name=$name job_id=$job_id model=$model offset=$offset"
  printf '%s\n' "$job_id"
}

run_wave() {
  local offset="$1" qwen_job llama_job
  if family_complete qwen25_7b capabilities && family_complete llama31_8b capabilities; then
    log 'qwen_llama_capabilities_already_complete=1'
    return 0
  fi
  wait_for_gpu_test_clear
  qwen_job="$(submit_capability_job "bonh_qwen_cap${offset}s" qwen25_7b "$offset")"
  llama_job="$(submit_capability_job "bonh_llama_cap${offset}s" llama31_8b "$offset")"
  wait_jobs "qwen_llama_capabilities_offset_$offset" "$qwen_job" "$llama_job"
}

log "capabilities_after_source_supervisor_start commit=$(git -C "$REPO_DIR" rev-parse --short HEAD)"
wait_for_sources
rm -f "$DEFER_MARKER"
log "defer_marker_removed=$DEFER_MARKER"
run_wave 0
run_wave 4
if ! family_complete qwen25_7b capabilities || ! family_complete llama31_8b capabilities; then
  printf 'Capability waves ended without complete Qwen/Llama receipts\n' >&2
  exit 1
fi
rm -f "$CAPABILITY_PRIORITY_MARKER"
log "capability_priority_marker_removed=$CAPABILITY_PRIORITY_MARKER"
log 'capabilities_after_source_supervisor_complete=1'
