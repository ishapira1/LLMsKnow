#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

: "${LLAMA_CORE_MASK_JOB_ID:?LLAMA_CORE_MASK_JOB_ID is required}"
: "${QWEN_CORE_MASK_JOB_ID:?QWEN_CORE_MASK_JOB_ID is required}"
: "${LLAMA_EVAL_PREP_JOB_ID:?LLAMA_EVAL_PREP_JOB_ID is required}"
: "${QWEN_EVAL_PREP_JOB_ID:?QWEN_EVAL_PREP_JOB_ID is required}"

POLL_SECONDS="${POLL_SECONDS:-20}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
GPU_PARTITION="${BONHAM_GPU_TEST_PARTITION:-gpu_test}"
GPU_GRES="${BONHAM_GPU_TEST_GRES:-gpu:nvidia_a100_3g.20gb}"
USER_NAME="${USER:-ishapira}"
mkdir -p "$LOG_ROOT/slurm/gpu_model_pipeline"

job_state() {
  sacct -X -j "$1" -n -P -o State 2>/dev/null | head -1
}

wait_for_prerequisites() {
  local llama_mask qwen_mask llama_eval qwen_eval combined
  while true; do
    llama_mask="$(job_state "$LLAMA_CORE_MASK_JOB_ID")"
    qwen_mask="$(job_state "$QWEN_CORE_MASK_JOB_ID")"
    llama_eval="$(job_state "$LLAMA_EVAL_PREP_JOB_ID")"
    qwen_eval="$(job_state "$QWEN_EVAL_PREP_JOB_ID")"
    printf 'time=%s llama_mask=%s qwen_mask=%s llama_evalprep=%s qwen_evalprep=%s\n' \
      "$(date -Is)" "$llama_mask" "$qwen_mask" "$llama_eval" "$qwen_eval"
    if [[ "$llama_mask" == COMPLETED* && "$qwen_mask" == COMPLETED* && \
          "$llama_eval" == COMPLETED* && "$qwen_eval" == COMPLETED* ]]; then
      return 0
    fi
    combined="$llama_mask:$qwen_mask:$llama_eval:$qwen_eval"
    case "$combined" in
      *FAILED*|*CANCELLED*|*OUT_OF_MEMORY*|*TIMEOUT*|*NODE_FAIL*)
        printf 'prerequisite_failure=%s\n' "$combined" >&2
        return 1
        ;;
    esac
    sleep "$POLL_SECONDS"
  done
}

wait_for_gpu_test_clear() {
  local active
  while true; do
    active="$(squeue -h -u "$USER_NAME" -p "$GPU_PARTITION" | wc -l | tr -d ' ')"
    if (( active == 0 )); then return 0; fi
    printf 'time=%s waiting_gpu_test_clear=%s\n' "$(date -Is)" "$active"
    sleep "$POLL_SECONDS"
  done
}

submit_pipeline() {
  local model_key="$1" short="$2" raw job_id
  raw="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$GPU_PARTITION" \
    --job-name="bonh_${short}_pipeline" --time=12:00:00 \
    --nodes=1 --ntasks=4 --cpus-per-task=4 --mem=192G \
    --gres="$GPU_GRES:4" \
    --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$model_key,EVALUATION_BATCH_SIZE=4" \
    --output="$LOG_ROOT/slurm/gpu_model_pipeline/%x_%j.out" \
    --error="$LOG_ROOT/slurm/gpu_model_pipeline/%x_%j.err" \
    "$BUNDLE_DIR/gpu_model_pipeline.sbatch")"
  job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'unexpected_sbatch_response model=%s response=%s\n' "$model_key" "$raw" >&2
    return 2
  }
  printf '%s\n' "$job_id"
}

wait_for_prerequisites
wait_for_gpu_test_clear
llama_pipeline="$(submit_pipeline llama31_8b llama)"
qwen_pipeline="$(submit_pipeline qwen25_7b qwen)"
printf 'time=%s llama_pipeline=%s qwen_pipeline=%s\n' \
  "$(date -Is)" "$llama_pipeline" "$qwen_pipeline"
