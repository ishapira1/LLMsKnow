#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

POLL_SECONDS="${POLL_SECONDS:-30}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
GPU_PARTITION="${BONHAM_GPU_TEST_PARTITION:-gpu_test}"
GPU_GRES="${BONHAM_GPU_TEST_GRES:-gpu:nvidia_a100_3g.20gb}"
USER_NAME="${USER:-ishapira}"
ACCOUNTING_START="${BONHAM_ACCOUNTING_START:-2026-09-19}"
MAX_MEMORY_GIB="${LLAMA_DEVICE_MAX_MEMORY_GIB:-12}"
CAPABILITY_BATCH_SIZE="${LLAMA_CAPABILITY_BATCH_SIZE:-1}"
QWEN_JOBS="${QWEN_CAPABILITY_JOBS:-47309712:47310851}"
REGULAR_LLAMA_JOBS="${REGULAR_LLAMA_CAPABILITY_JOBS:-47311014:47306875}"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
}
[[ "$MAX_MEMORY_GIB" =~ ^[1-9][0-9]*$ ]] || {
  printf 'LLAMA_DEVICE_MAX_MEMORY_GIB must be a positive integer\n' >&2
  exit 2
}
[[ "$CAPABILITY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  printf 'LLAMA_CAPABILITY_BATCH_SIZE must be a positive integer\n' >&2
  exit 2
}

mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/gpu_eval_states"
supervisor_log="$LOG_ROOT/submit/llama_capabilities_sharded_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$supervisor_log") 2>&1

log() { printf '%s time=%s\n' "$*" "$(date -Is)" >&2; }

job_state() {
  local job_id="$1"
  sacct -X -n -j "$job_id" --parsable2 --format=State 2>/dev/null \
    | cut -d'|' -f1 | head -1
}

wait_terminal() {
  local label="$1" job_id state
  IFS=':' read -r -a jobs <<< "$2"
  while true; do
    local pending=0
    for job_id in "${jobs[@]}"; do
      state="$(job_state "$job_id")"
      case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ;;
        *) pending=1 ;;
      esac
    done
    if (( pending == 0 )); then
      log "terminal label=$label jobs=${jobs[*]}"
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

capability_state_complete() {
  local state_id="$1" index_path shard_count last_shard
  index_path="$RESULT_ROOT/evaluations/inputs/llama31_8b/capabilities/index.jsonl"
  [[ -s "$index_path" ]] || return 1
  shard_count="$(wc -l < "$index_path" | tr -d ' ')"
  [[ "$shard_count" =~ ^[1-9][0-9]*$ ]] || return 1
  last_shard="$((shard_count - 1))"
  [[ -f "$RESULT_ROOT/evaluations/results/llama31_8b/$state_id/capabilities/$(printf 'shard_%04d' "$last_shard")/COMPLETE" ]]
}

pair_complete() {
  local pair="$1" first second
  IFS=':' read -r first second <<< "$pair"
  local -a states=(
    unpruned n1_mechanism n2_selective random_n1
    random_n2 weak_prompt strong_prompt prompt_only_meandiff
  )
  capability_state_complete "${states[$first]}" && capability_state_complete "${states[$second]}"
}

wait_gpu_test_slot() {
  local active
  while true; do
    active="$(squeue -h -u "$USER_NAME" -p "$GPU_PARTITION" | wc -l | tr -d ' ')"
    # gpu_test permits two submitted jobs per user.  Each Llama recovery job
    # requests four of the eight permitted slices, so it is safe to fill one
    # free job slot while a one-slice Qwen recovery is still running.
    if (( active < 2 )); then return 0; fi
    log "waiting_for_gpu_test_slot active=$active"
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

submit_pair() {
  local pair="$1" name="bonh_lcap_shard_${1/:/}" existing state raw job_id
  if pair_complete "$pair"; then
    log "pair_already_complete pair=$pair"
    return 0
  fi
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
  raw="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$GPU_PARTITION" --job-name="$name" \
    --nodes=1 --ntasks=2 --cpus-per-task=4 --mem=192G --time=12:00:00 \
    --gres="$GPU_GRES:4" \
    --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=llama31_8b,STATE_INDICES=$pair,STATE_COUNT=2,GPUS_PER_STATE=2,CPUS_PER_STATE=4,MEM_PER_STATE=96G,EVALUATION_FAMILY_SET=capabilities,CAPABILITY_EVALUATION_BATCH_SIZE=$CAPABILITY_BATCH_SIZE,LLMSSYCOPH_DEVICE_MAX_MEMORY_GIB=$MAX_MEMORY_GIB" \
    --output="$LOG_ROOT/slurm/gpu_eval_states/%x_%j.out" \
    --error="$LOG_ROOT/slurm/gpu_eval_states/%x_%j.err" \
    "$BUNDLE_DIR/gpu_eval_states.sbatch")"
  job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'Unexpected sbatch response for %s: %s\n' "$name" "$raw" >&2
    return 2
  }
  log "submitted_job name=$name job_id=$job_id pair=$pair max_memory_gib=$MAX_MEMORY_GIB batch_size=$CAPABILITY_BATCH_SIZE"
  printf '%s\n' "$job_id"
}

wait_success() {
  local label="$1" job_id state pending
  shift
  while true; do
    pending=0
    for job_id in "$@"; do
      state="$(job_state "$job_id")"
      case "$state" in
        COMPLETED) ;;
        PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|'') pending=1 ;;
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

run_wave() {
  local label="$1" pair_a="$2" pair_b="$3" job_id
  local -a jobs=()
  if ! pair_complete "$pair_a"; then
    wait_gpu_test_slot
    job_id="$(submit_pair "$pair_a")"
    [[ -z "$job_id" ]] || jobs+=("$job_id")
  fi
  if ! pair_complete "$pair_b"; then
    wait_gpu_test_slot
    job_id="$(submit_pair "$pair_b")"
    [[ -z "$job_id" ]] || jobs+=("$job_id")
  fi
  if (( ${#jobs[@]} > 0 )); then
    wait_success "$label" "${jobs[@]}"
  fi
}

log "llama_capabilities_sharded_start commit=$(git -C "$REPO_DIR" rev-parse --short HEAD)"
wait_terminal qwen_capabilities "$QWEN_JOBS"

# The full-memory jobs are fallbacks for the same immutable coordinates.  Hold
# only pending jobs; a running job is useful and must finish instead of racing.
IFS=':' read -r -a regular_jobs <<< "$REGULAR_LLAMA_JOBS"
for job_id in "${regular_jobs[@]}"; do
  state="$(job_state "$job_id")"
  case "$state" in
    PENDING)
      scontrol hold "$job_id"
      log "held_pending_regular_fallback job_id=$job_id"
      ;;
    RUNNING|CONFIGURING|COMPLETING)
      log "regular_fallback_already_active job_id=$job_id state=$state"
      wait_success regular_llama_capability "$job_id"
      ;;
  esac
done

run_wave llama_capabilities_0_3 0:1 2:3
run_wave llama_capabilities_4_7 4:5 6:7

for state_id in unpruned n1_mechanism n2_selective random_n1 random_n2 weak_prompt strong_prompt prompt_only_meandiff; do
  capability_state_complete "$state_id" || {
    printf 'Llama capability state remains incomplete: %s\n' "$state_id" >&2
    exit 1
  }
done
log 'llama_capabilities_sharded_complete=1'
