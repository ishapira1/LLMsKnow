#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

: "${LLAMA_PIPELINE_JOB_ID:?LLAMA_PIPELINE_JOB_ID is required}"
: "${QWEN_PIPELINE_JOB_ID:?QWEN_PIPELINE_JOB_ID is required}"
: "${LLAMA_ANALYSIS_ARRAY_JOB_ID:?LLAMA_ANALYSIS_ARRAY_JOB_ID is required}"
: "${QWEN_ANALYSIS_ARRAY_JOB_ID:?QWEN_ANALYSIS_ARRAY_JOB_ID is required}"

POLL_SECONDS="${POLL_SECONDS:-30}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
GPU_PARTITION="${BONHAM_GPU_TEST_PARTITION:-gpu_test}"
GPU_GRES="${BONHAM_GPU_TEST_GRES:-gpu:nvidia_a100_3g.20gb}"
USER_NAME="${USER:-ishapira}"
mkdir -p "$LOG_ROOT/slurm/gpu_score_fallback"

job_state() {
  sacct -X -j "$1" -n -P -o State 2>/dev/null | head -1
}

score_count() {
  find "$RESULT_ROOT/scores/$1" -mindepth 2 -maxdepth 2 \
    -type f -name COMPLETE.json 2>/dev/null | wc -l | tr -d ' '
}

active_array_tasks() {
  { squeue -h -j "$1" -t R,CG 2>/dev/null || true; } \
    | wc -l | tr -d ' '
}

gpu_test_has_slot() {
  local active
  active="$(squeue -h -u "$USER_NAME" -p "$GPU_PARTITION" | wc -l | tr -d ' ')"
  (( active < 2 ))
}

submit_fallback() {
  local model_key="$1" short="$2" raw job_id
  raw="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$GPU_PARTITION" \
    --job-name="bonh_${short}_analysis_test" --time=12:00:00 \
    --nodes=1 --ntasks=2 --cpus-per-task=4 --mem=192G \
    --gres="$GPU_GRES:4" \
    --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$model_key,LANES=2,GPUS_PER_LANE=2,CPUS_PER_LANE=4,MEM_PER_LANE=96G,BLOCKS_PER_PASS=2,SCORE_IDS_COLON=n1_seed17_prune:source_all_prune:n1_seed29_prune:source_false_prune" \
    --output="$LOG_ROOT/slurm/gpu_score_fallback/%x_%j.out" \
    --error="$LOG_ROOT/slurm/gpu_score_fallback/%x_%j.err" \
    "$BUNDLE_DIR/gpu_score_multilane.sbatch")"
  job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'unexpected_sbatch_response model=%s response=%s\n' "$model_key" "$raw" >&2
    return 2
  }
  printf '%s\n' "$job_id"
}

llama_fallback=''
qwen_fallback=''
while [[ -z "$llama_fallback" || -z "$qwen_fallback" ]]; do
  llama_state="$(job_state "$LLAMA_PIPELINE_JOB_ID")"
  qwen_state="$(job_state "$QWEN_PIPELINE_JOB_ID")"
  combined="$llama_state:$qwen_state"
  printf 'time=%s llama_pipeline=%s qwen_pipeline=%s llama_analysis=%s qwen_analysis=%s\n' \
    "$(date -Is)" "$llama_state" "$qwen_state" \
    "${llama_fallback:-waiting}" "${qwen_fallback:-waiting}"
  case "$combined" in
    *FAILED*|*CANCELLED*|*OUT_OF_MEMORY*|*TIMEOUT*|*NODE_FAIL*)
      printf 'pipeline_failure=%s\n' "$combined" >&2
      exit 1
      ;;
  esac

  if [[ -z "$qwen_fallback" && "$qwen_state" == COMPLETED* ]]; then
    if (( $(score_count qwen25_7b) == 7 )); then
      qwen_fallback='complete'
    elif (( $(active_array_tasks "$QWEN_ANALYSIS_ARRAY_JOB_ID") == 0 )) && \
        gpu_test_has_slot; then
      scancel "$QWEN_ANALYSIS_ARRAY_JOB_ID" 2>/dev/null || true
      qwen_fallback="$(submit_fallback qwen25_7b qwen)"
      printf 'time=%s submitted_model=qwen25_7b fallback=%s\n' \
        "$(date -Is)" "$qwen_fallback"
    else
      printf 'time=%s waiting_model=qwen25_7b reason=active_analysis_or_gpu_slot\n' \
        "$(date -Is)"
    fi
  fi
  if [[ -z "$llama_fallback" && "$llama_state" == COMPLETED* ]]; then
    if (( $(score_count llama31_8b) == 7 )); then
      llama_fallback='complete'
    elif (( $(active_array_tasks "$LLAMA_ANALYSIS_ARRAY_JOB_ID") == 0 )) && \
        gpu_test_has_slot; then
      scancel "$LLAMA_ANALYSIS_ARRAY_JOB_ID" 2>/dev/null || true
      llama_fallback="$(submit_fallback llama31_8b llama)"
      printf 'time=%s submitted_model=llama31_8b fallback=%s\n' \
        "$(date -Is)" "$llama_fallback"
    else
      printf 'time=%s waiting_model=llama31_8b reason=active_analysis_or_gpu_slot\n' \
        "$(date -Is)"
    fi
  fi
  [[ -n "$llama_fallback" && -n "$qwen_fallback" ]] || sleep "$POLL_SECONDS"
done
printf 'time=%s llama_analysis=%s qwen_analysis=%s\n' \
  "$(date -Is)" "$llama_fallback" "$qwen_fallback"
