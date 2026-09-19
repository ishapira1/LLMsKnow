#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

POLL_SECONDS="${POLL_SECONDS:-30}"
EARLY_QWEN_LLAMA_ONLY="${BONHAM_EARLY_QWEN_LLAMA_ONLY:-0}"
ARTIFACT_ONLY_FINAL_TAIL="${BONHAM_ARTIFACT_ONLY_FINAL_TAIL:-0}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
GPU_PARTITION="${BONHAM_GPU_TEST_PARTITION:-gpu_test}"
GPU_GRES="${BONHAM_GPU_TEST_GRES:-gpu:nvidia_a100_3g.20gb}"
ACCOUNTING_START="${BONHAM_ACCOUNTING_START:-2026-09-19}"
USER_NAME="${USER:-ishapira}"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
}
[[ "$EARLY_QWEN_LLAMA_ONLY" == 0 || "$EARLY_QWEN_LLAMA_ONLY" == 1 ]] || {
  printf 'BONHAM_EARLY_QWEN_LLAMA_ONLY must be 0 or 1\n' >&2
  exit 2
}
[[ "$ARTIFACT_ONLY_FINAL_TAIL" == 0 || "$ARTIFACT_ONLY_FINAL_TAIL" == 1 ]] || {
  printf 'BONHAM_ARTIFACT_ONLY_FINAL_TAIL must be 0 or 1\n' >&2
  exit 2
}
if (( EARLY_QWEN_LLAMA_ONLY == 1 && ARTIFACT_ONLY_FINAL_TAIL == 1 )); then
  printf 'Early-evaluation and artifact-only modes are mutually exclusive\n' >&2
  exit 2
fi

mkdir -p \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/gpu_eval_states" \
  "$LOG_ROOT/slurm/eval_validate" \
  "$LOG_ROOT/slurm/evalplus_prepare" \
  "$LOG_ROOT/slurm/evalplus_run" \
  "$LOG_ROOT/slurm/evalplus_aggregate" \
  "$LOG_ROOT/slurm/weight_aggregate" \
  "$LOG_ROOT/slurm/report" \
  "$LOG_ROOT/slurm/final_audit" \
  "$LOG_ROOT/slurm/final_email"

supervisor_log="$LOG_ROOT/submit/accelerate_tail_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$supervisor_log") 2>&1

log() { printf '%s time=%s\n' "$*" "$(date -Is)" >&2; }

job_id_by_name() {
  local name="$1" id
  id="$({
    squeue -h -u "$USER_NAME" -n "$name" -o '%A' 2>/dev/null || true
    sacct -S "$ACCOUNTING_START" -X -n --name "$name" \
      --parsable2 --format=JobIDRaw 2>/dev/null | cut -d'|' -f1 || true
  } | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
  printf '%s\n' "$id"
}

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
    log "waiting_for_jobs label=$label jobs=$*"
    sleep "$POLL_SECONDS"
  done
}

wait_for_eval_prerequisites() {
  local ready model
  while true; do
    ready=1
    [[ -f "$RESULT_ROOT/evaluations/inputs/COMPLETE.json" ]] || ready=0
    for model in qwen25_7b llama31_8b gemma4_12b; do
      [[ -f "$RESULT_ROOT/states/$model/MASK_STATES_COMPLETE.json" ]] || ready=0
      [[ -f "$RESULT_ROOT/steering/$model/frozen/COMPLETE.json" ]] || ready=0
    done
    if (( ready == 1 )); then
      log 'evaluation_prerequisites_complete=1'
      return 0
    fi
    log 'waiting_for_evaluation_prerequisites=1'
    sleep "$POLL_SECONDS"
  done
}

wait_for_model_eval_prerequisites() {
  local ready model
  while true; do
    ready=1
    for model in "$@"; do
      [[ -f "$RESULT_ROOT/evaluations/inputs/$model/COMPLETE.json" ]] || ready=0
      [[ -f "$RESULT_ROOT/states/$model/MASK_STATES_COMPLETE.json" ]] || ready=0
    done
    if (( ready == 1 )); then
      log "model_evaluation_prerequisites_complete models=$*"
      return 0
    fi
    log "waiting_for_model_evaluation_prerequisites models=$*"
    sleep "$POLL_SECONDS"
  done
}

wait_for_model_steering() {
  local ready model
  while true; do
    ready=1
    for model in "$@"; do
      [[ -f "$RESULT_ROOT/steering/$model/frozen/COMPLETE.json" ]] || ready=0
    done
    if (( ready == 1 )); then
      log "model_steering_complete models=$*"
      return 0
    fi
    log "waiting_for_model_steering models=$*"
    sleep "$POLL_SECONDS"
  done
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

submit_eval_job() {
  local name="$1" model="$2" offset="$3" family_set="$4"
  local gpus_per_state="$5" mem_per_state="$6" total_gpus="$7" total_mem="$8"
  local existing raw job_id
  existing="$(job_id_by_name "$name")"
  if [[ -n "$existing" ]]; then
    log "reuse_job name=$name job_id=$existing"
    printf '%s\n' "$existing"
    return 0
  fi
  raw="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$GPU_PARTITION" --job-name="$name" \
    --nodes=1 --ntasks=4 --cpus-per-task=4 --mem="$total_mem" --time=12:00:00 \
    --gres="$GPU_GRES:$total_gpus" \
    --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$model,STATE_OFFSET=$offset,STATE_COUNT=4,GPUS_PER_STATE=$gpus_per_state,CPUS_PER_STATE=4,MEM_PER_STATE=$mem_per_state,EVALUATION_FAMILY_SET=$family_set,EVALUATION_BATCH_SIZE=4" \
    --output="$LOG_ROOT/slurm/gpu_eval_states/%x_%j.out" \
    --error="$LOG_ROOT/slurm/gpu_eval_states/%x_%j.err" \
    "$BUNDLE_DIR/gpu_eval_states.sbatch")"
  job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'Unexpected sbatch response for %s: %s\n' "$name" "$raw" >&2
    return 2
  }
  log "submitted_job name=$name job_id=$job_id model=$model offset=$offset family_set=$family_set"
  printf '%s\n' "$job_id"
}

run_qwen_llama_wave() {
  local suffix="$1" offset="$2" family_set="$3"
  local qwen_job llama_job
  wait_for_gpu_test_clear
  qwen_job="$(submit_eval_job "bonh_qwen_${suffix}" qwen25_7b "$offset" "$family_set" 1 48G 4 192G)"
  llama_job="$(submit_eval_job "bonh_llama_${suffix}" llama31_8b "$offset" "$family_set" 1 48G 4 192G)"
  wait_jobs "$suffix" "$qwen_job" "$llama_job"
}

run_gemma_wave() {
  local suffix="$1" offset="$2" family_set="$3" job_id
  wait_for_gpu_test_clear
  job_id="$(submit_eval_job "bonh_gemma_${suffix}" gemma4_12b "$offset" "$family_set" 2 96G 8 384G)"
  wait_jobs "gemma_$suffix" "$job_id"
}

submit_cpu_job() {
  local name="$1" stage="$2" partition="$3" time_limit="$4" memory="$5"
  local array_spec="${6:-}" existing raw job_id
  local -a command
  existing="$(job_id_by_name "$name")"
  if [[ -n "$existing" ]]; then
    log "reuse_job name=$name job_id=$existing"
    printf '%s\n' "$existing"
    return 0
  fi
  command=(
    sbatch --parsable --account="$ACCOUNT" --partition="$partition"
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
  log "submitted_job name=$name job_id=$job_id stage=$stage"
  printf '%s\n' "$job_id"
}

wait_for_weight_analysis() {
  local ready model
  while true; do
    ready=1
    for model in qwen25_7b llama31_8b gemma4_12b; do
      [[ -f "$RESULT_ROOT/weight_analysis/$model/COMPLETE.json" ]] || ready=0
    done
    if (( ready == 1 )); then return 0; fi
    log 'waiting_for_weight_analysis=1'
    sleep "$POLL_SECONDS"
  done
}

wait_for_evaluation_artifacts() {
  local ready model state family index_path shard_count last_shard complete_path
  local -a states=(
    unpruned n1_mechanism n2_selective random_n1
    random_n2 weak_prompt strong_prompt prompt_only_meandiff
  )
  while true; do
    ready=1
    for model in qwen25_7b llama31_8b gemma4_12b; do
      for family in generalization useful_assertions capabilities; do
        index_path="$RESULT_ROOT/evaluations/inputs/$model/$family/index.jsonl"
        if [[ ! -s "$index_path" ]]; then
          ready=0
          continue
        fi
        shard_count="$(wc -l < "$index_path" | tr -d ' ')"
        if ! [[ "$shard_count" =~ ^[1-9][0-9]*$ ]]; then
          ready=0
          continue
        fi
        last_shard="$((shard_count - 1))"
        for state in "${states[@]}"; do
          complete_path="$(printf \
            '%s/evaluations/results/%s/%s/%s/shard_%04d/COMPLETE.json' \
            "$RESULT_ROOT" "$model" "$state" "$family" "$last_shard")"
          [[ -f "$complete_path" ]] || ready=0
        done
      done
    done
    if (( ready == 1 )); then
      log 'all_evaluation_terminal_shards_complete=1'
      return 0
    fi
    log 'waiting_for_evaluation_artifacts=1'
    sleep "$POLL_SECONDS"
  done
}

log "supervisor_start result_root=$RESULT_ROOT commit=$(git -C "$REPO_DIR" rev-parse --short HEAD) early_qwen_llama_only=$EARLY_QWEN_LLAMA_ONLY artifact_only_final_tail=$ARTIFACT_ONLY_FINAL_TAIL"
if (( EARLY_QWEN_LLAMA_ONLY == 1 )); then
  # Start the four non-steering states as soon as each model's frozen inputs and
  # mask states exist.  Steering can finish concurrently before the second wave.
  wait_for_model_eval_prerequisites qwen25_7b llama31_8b
  run_qwen_llama_wave core0 0 paper_core
  wait_for_model_steering qwen25_7b llama31_8b
  run_qwen_llama_wave core1 4 paper_core
  run_qwen_llama_wave cap0 0 capabilities
  run_qwen_llama_wave cap1 4 capabilities
  log 'early_qwen_llama_supervisor_complete=1'
  exit 0
fi
if (( ARTIFACT_ONLY_FINAL_TAIL == 1 )); then
  # A separately managed packed pipeline owns every model evaluation.  Wait on
  # immutable terminal-shard receipts and never submit duplicate GPU waves.
  wait_for_evaluation_artifacts
else
  wait_for_eval_prerequisites

  # Paper results first, without changing any frozen task or final-audit requirement.
  run_qwen_llama_wave core0 0 paper_core
  run_qwen_llama_wave core1 4 paper_core
  run_gemma_wave core0 0 paper_core
  run_gemma_wave core1 4 paper_core

  # Complete the capability family for all eight states and all three models.
  run_qwen_llama_wave cap0 0 capabilities
  run_qwen_llama_wave cap1 4 capabilities
  run_gemma_wave cap0 0 capabilities
  run_gemma_wave cap1 4 capabilities
fi

eval_validate_job="$(submit_cpu_job bonh_evalval_acc eval_validate serial_requeue 02:00:00 96G)"
evalplus_prepare_job="$(submit_cpu_job bonh_eprep_acc evalplus_prepare serial_requeue 01:00:00 96G)"
wait_jobs evaluation_validation "$eval_validate_job" "$evalplus_prepare_job"

evalplus_run_job="$(submit_cpu_job bonh_eprun_acc evalplus_run serial_requeue 03:00:00 24G '0-191%40')"
wait_jobs evalplus_run "$evalplus_run_job"
evalplus_aggregate_job="$(submit_cpu_job bonh_epagg_acc evalplus_aggregate test 00:10:00 96G)"
wait_jobs evalplus_aggregate "$evalplus_aggregate_job"

wait_for_weight_analysis
weight_aggregate_job="$(submit_cpu_job bonh_weightagg_acc weight_aggregate test 00:10:00 96G)"
wait_jobs weight_aggregate "$weight_aggregate_job"

report_job="$(submit_cpu_job bonh_report_acc report serial_requeue 02:00:00 96G)"
wait_jobs report "$report_job"
audit_job="$(submit_cpu_job bonh_audit_acc final_audit serial_requeue 04:00:00 96G)"
wait_jobs final_audit "$audit_job"
email_job="$(submit_cpu_job bonh_email_acc final_email test 00:10:00 16G)"
wait_jobs final_email "$email_job"

log "supervisor_complete audit_job=$audit_job email_job=$email_job"
