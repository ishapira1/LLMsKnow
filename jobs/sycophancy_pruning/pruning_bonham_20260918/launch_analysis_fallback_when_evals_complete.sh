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

wait_for_pipelines() {
  local llama_state qwen_state combined
  while true; do
    llama_state="$(job_state "$LLAMA_PIPELINE_JOB_ID")"
    qwen_state="$(job_state "$QWEN_PIPELINE_JOB_ID")"
    printf 'time=%s llama_pipeline=%s qwen_pipeline=%s\n' \
      "$(date -Is)" "$llama_state" "$qwen_state"
    if [[ "$llama_state" == COMPLETED* && "$qwen_state" == COMPLETED* ]]; then
      return 0
    fi
    combined="$llama_state:$qwen_state"
    case "$combined" in
      *FAILED*|*CANCELLED*|*OUT_OF_MEMORY*|*TIMEOUT*|*NODE_FAIL*)
        printf 'pipeline_failure=%s\n' "$combined" >&2
        return 1
        ;;
    esac
    sleep "$POLL_SECONDS"
  done
}

score_count() {
  find "$RESULT_ROOT/scores/$1" -mindepth 2 -maxdepth 2 \
    -type f -name COMPLETE.json 2>/dev/null | wc -l | tr -d ' '
}

wait_for_active_array_tasks() {
  local job_id="$1" active
  while true; do
    active="$(
      { squeue -h -j "$job_id" -t R,CG 2>/dev/null || true; } \
        | wc -l | tr -d ' '
    )"
    if (( active == 0 )); then return 0; fi
    printf 'time=%s waiting_for_active_analysis_tasks job=%s active=%s\n' \
      "$(date -Is)" "$job_id" "$active"
    sleep "$POLL_SECONDS"
  done
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

wait_for_pipelines

llama_missing=0
qwen_missing=0
(( $(score_count llama31_8b) == 7 )) || llama_missing=1
(( $(score_count qwen25_7b) == 7 )) || qwen_missing=1

if (( llama_missing == 1 )); then
  wait_for_active_array_tasks "$LLAMA_ANALYSIS_ARRAY_JOB_ID"
  scancel "$LLAMA_ANALYSIS_ARRAY_JOB_ID" 2>/dev/null || true
fi
if (( qwen_missing == 1 )); then
  wait_for_active_array_tasks "$QWEN_ANALYSIS_ARRAY_JOB_ID"
  scancel "$QWEN_ANALYSIS_ARRAY_JOB_ID" 2>/dev/null || true
fi

llama_fallback='complete'
qwen_fallback='complete'
while (( llama_missing == 1 || qwen_missing == 1 )); do
  if (( qwen_missing == 1 )) && gpu_test_has_slot; then
    qwen_fallback="$(submit_fallback qwen25_7b qwen)"
    qwen_missing=0
    printf 'time=%s submitted_model=qwen25_7b fallback=%s\n' \
      "$(date -Is)" "$qwen_fallback"
  fi
  if (( llama_missing == 1 )) && gpu_test_has_slot; then
    llama_fallback="$(submit_fallback llama31_8b llama)"
    llama_missing=0
    printf 'time=%s submitted_model=llama31_8b fallback=%s\n' \
      "$(date -Is)" "$llama_fallback"
  fi
  (( llama_missing == 0 && qwen_missing == 0 )) || sleep "$POLL_SECONDS"
done
printf 'time=%s llama_analysis=%s qwen_analysis=%s\n' \
  "$(date -Is)" "$llama_fallback" "$qwen_fallback"
