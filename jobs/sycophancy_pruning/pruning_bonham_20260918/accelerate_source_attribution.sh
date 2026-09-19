#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

POLL_SECONDS="${POLL_SECONDS:-60}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
GPU_PARTITION="${BONHAM_GPU_TEST_PARTITION:-gpu_test}"
GPU_GRES="${BONHAM_GPU_TEST_GRES:-gpu:nvidia_a100_3g.20gb}"
ACCOUNTING_START="${BONHAM_ACCOUNTING_START:-2026-09-19}"
USER_NAME="${USER:-ishapira}"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
}

mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/gpu_eval_states"
supervisor_log="$LOG_ROOT/submit/source_attribution_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$supervisor_log") 2>&1

log() { printf '%s time=%s\n' "$*" "$(date -Is)" >&2; }

job_id_by_name() {
  local name="$1"
  {
    squeue -h -u "$USER_NAME" -n "$name" -o '%A' 2>/dev/null || true
    sacct -S "$ACCOUNTING_START" -X -n --name "$name" \
      --parsable2 --format=JobIDRaw 2>/dev/null | cut -d'|' -f1 || true
  } | grep -E '^[0-9]+$' | sort -n | tail -1 || true
}

job_state() {
  sacct -X -n -j "$1" --parsable2 --format=State 2>/dev/null \
    | cut -d'|' -f1 | head -1
}

job_partition() {
  scontrol show job -o "$1" 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^Partition=//p' | head -1
}

source_wave_complete() {
  local model="$1" offset="$2" shard_count state_index state_id observed
  local -a states=(
    unpruned n1_mechanism n2_selective random_n1
    random_n2 weak_prompt strong_prompt prompt_only_meandiff
  )
  shard_count="$(wc -l < "$RESULT_ROOT/evaluations/inputs/$model/source_attribution/index.jsonl" | tr -d ' ')"
  for ((state_index = offset; state_index < offset + 4; state_index++)); do
    state_id="${states[$state_index]}"
    observed="$(find "$RESULT_ROOT/evaluations/results/$model/$state_id/source_attribution" \
      -type f -name COMPLETE 2>/dev/null | wc -l | tr -d ' ')"
    (( observed >= shard_count )) || return 1
  done
}

wait_jobs() {
  local label="$1"
  shift
  local pending job_id state
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

gpu_test_job_count() {
  squeue -h -u "$USER_NAME" -p "$GPU_PARTITION" | wc -l | tr -d ' '
}

wait_for_gpu_test_slot() {
  local active
  while true; do
    active="$(gpu_test_job_count)"
    if (( active < 2 )); then return 0; fi
    log "waiting_for_gpu_test_slot active=$active"
    sleep "$POLL_SECONDS"
  done
}

wait_for_gpu_test_clear() {
  local active
  while true; do
    active="$(gpu_test_job_count)"
    if (( active == 0 )); then return 0; fi
    log "waiting_for_gpu_test_clear active=$active"
    sleep "$POLL_SECONDS"
  done
}

wait_for_model_inputs() {
  local model="$1"
  while [[ ! -s "$RESULT_ROOT/evaluations/inputs/$model/useful_assertions/index.jsonl" ]]; do
    log "waiting_for_useful_inputs model=$model"
    sleep "$POLL_SECONDS"
  done
}

freeze_model_source_sweep() {
  local model="$1" receipt
  receipt="$RESULT_ROOT/evaluations/inputs/$model/SOURCE_ATTRIBUTION_COMPLETE.json"
  if [[ ! -f "$receipt" ]]; then
    "$CPU_PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" prepare-source-attribution \
      --result-root "$RESULT_ROOT" --model-key "$model"
  fi
  [[ -f "$receipt" ]] || {
    printf 'Source-attribution freeze did not produce %s\n' "$receipt" >&2
    return 1
  }
}

wait_for_model_states() {
  local model="$1"
  while [[ ! -f "$RESULT_ROOT/states/$model/MASK_STATES_COMPLETE.json" ]] || \
        [[ ! -f "$RESULT_ROOT/steering/$model/frozen/COMPLETE.json" ]]; do
    log "waiting_for_model_states model=$model"
    sleep "$POLL_SECONDS"
  done
}

submit_wave() {
  local name="$1" model="$2" offset="$3" gpus_per_state="$4"
  local mem_per_state="$5" total_gpus="$6" total_mem="$7"
  local existing existing_state existing_partition raw job_id
  existing="$(job_id_by_name "$name")"
  if [[ -n "$existing" ]]; then
    existing_state="$(job_state "$existing")"
    existing_partition="$(job_partition "$existing")"
    # Regular-partition submissions are opportunistic fallbacks. If one is
    # still pending when a scarce gpu_test slot opens, replace it with the
    # immediately runnable gpu_test job. Running/completed work is preserved.
    if [[ "$existing_state" == PENDING && "$existing_partition" != "$GPU_PARTITION" ]] && \
       (( $(gpu_test_job_count) < 2 )); then
      log "replace_pending_regular_job name=$name job_id=$existing partition=$existing_partition"
      scancel "$existing"
      existing=''
    fi
    case "$existing_state" in
      FAILED|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|TIMEOUT|PREEMPTED)
        log "replace_terminal_job name=$name job_id=$existing state=$existing_state"
        existing=''
        ;;
    esac
  fi
  if [[ -n "$existing" ]]; then
    log "reuse_job name=$name job_id=$existing"
    printf '%s\n' "$existing"
    return 0
  fi
  raw="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$GPU_PARTITION" --job-name="$name" \
    --nodes=1 --ntasks=4 --cpus-per-task=4 --mem="$total_mem" --time=12:00:00 \
    --gres="$GPU_GRES:$total_gpus" \
    --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$model,STATE_OFFSET=$offset,STATE_COUNT=4,GPUS_PER_STATE=$gpus_per_state,CPUS_PER_STATE=4,MEM_PER_STATE=$mem_per_state,EVALUATION_FAMILY_SET=source_attribution,EVALUATION_BATCH_SIZE=4" \
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

wait_for_pipeline() {
  local model="$1" pipeline_name pipeline_id state
  case "$model" in
    qwen25_7b) pipeline_name=bonh_qwen_pipeline ;;
    llama31_8b) pipeline_name=bonh_llama_pipeline ;;
    *) printf 'No packed-pipeline name for %s\n' "$model" >&2; return 2 ;;
  esac
  pipeline_id="$(job_id_by_name "$pipeline_name")"
  [[ -n "$pipeline_id" ]] || {
    printf 'Could not resolve %s\n' "$pipeline_name" >&2
    return 2
  }
  while true; do
    state="$(job_state "$pipeline_id")"
    case "$state" in
      COMPLETED)
        log "pipeline_complete model=$model job_id=$pipeline_id"
        return 0
        ;;
      PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|'')
        sleep "$POLL_SECONDS"
        ;;
      *)
        log "pipeline_failure model=$model job_id=$pipeline_id state=$state"
        return 1
        ;;
    esac
  done
}

run_model_source_waves() {
  local model="$1" short_name job_id
  case "$model" in
    qwen25_7b) short_name=qwen ;;
    llama31_8b) short_name=llama ;;
    *) printf 'Unknown one-GPU source model: %s\n' "$model" >&2; return 2 ;;
  esac
  wait_for_pipeline "$model"
  if source_wave_complete "$model" 0; then
    log "source_wave_already_complete model=$model offset=0"
  else
    wait_for_gpu_test_slot
    job_id="$(submit_wave "bonh_${short_name}_src0" "$model" 0 1 48G 4 192G)"
    wait_jobs "${short_name}_source_0" "$job_id"
  fi
  if source_wave_complete "$model" 4; then
    log "source_wave_already_complete model=$model offset=4"
  else
    wait_for_gpu_test_slot
    job_id="$(submit_wave "bonh_${short_name}_src1" "$model" 4 1 48G 4 192G)"
    wait_jobs "${short_name}_source_1" "$job_id"
  fi
}

run_gemma_wave() {
  local suffix="$1" offset="$2" job_id
  wait_for_gpu_test_clear
  job_id="$(submit_wave "bonh_gemma_src${suffix}" gemma4_12b "$offset" 2 96G 8 384G)"
  wait_jobs "gemma_source_$suffix" "$job_id"
}

log "source_attribution_supervisor_start commit=$(git -C "$REPO_DIR" rev-parse --short HEAD)"
for model in qwen25_7b llama31_8b; do
  wait_for_model_inputs "$model"
  freeze_model_source_sweep "$model"
  wait_for_model_states "$model"
done
run_model_source_waves qwen25_7b &
qwen_worker="$!"
run_model_source_waves llama31_8b &
llama_worker="$!"
worker_status=0
wait "$qwen_worker" || worker_status=1
wait "$llama_worker" || worker_status=1
(( worker_status == 0 )) || exit "$worker_status"

wait_for_model_inputs gemma4_12b
freeze_model_source_sweep gemma4_12b
wait_for_model_states gemma4_12b
run_gemma_wave 0 0
run_gemma_wave 1 4
log 'source_attribution_supervisor_complete=1'
