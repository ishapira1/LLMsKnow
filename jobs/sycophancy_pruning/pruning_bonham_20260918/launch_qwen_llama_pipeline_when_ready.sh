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

gpu_test_has_slot() {
  local active
  active="$(squeue -h -u "$USER_NAME" -p "$GPU_PARTITION" | wc -l | tr -d ' ')"
  (( active < 2 ))
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

llama_pipeline="$(squeue -h -u "$USER_NAME" -n bonh_llama_pipeline -o '%A' | head -1)"
qwen_pipeline="$(squeue -h -u "$USER_NAME" -n bonh_qwen_pipeline -o '%A' | head -1)"
while [[ -z "$llama_pipeline" || -z "$qwen_pipeline" ]]; do
  llama_mask="$(job_state "$LLAMA_CORE_MASK_JOB_ID")"
  qwen_mask="$(job_state "$QWEN_CORE_MASK_JOB_ID")"
  llama_eval="$(job_state "$LLAMA_EVAL_PREP_JOB_ID")"
  qwen_eval="$(job_state "$QWEN_EVAL_PREP_JOB_ID")"
  combined="$llama_mask:$qwen_mask:$llama_eval:$qwen_eval"
  printf 'time=%s llama_mask=%s qwen_mask=%s llama_evalprep=%s qwen_evalprep=%s llama_pipeline=%s qwen_pipeline=%s\n' \
    "$(date -Is)" "$llama_mask" "$qwen_mask" "$llama_eval" "$qwen_eval" \
    "${llama_pipeline:-none}" "${qwen_pipeline:-none}"
  case "$combined" in
    *FAILED*|*CANCELLED*|*OUT_OF_MEMORY*|*TIMEOUT*|*NODE_FAIL*)
      printf 'prerequisite_failure=%s\n' "$combined" >&2
      exit 1
      ;;
  esac
  if [[ -z "$qwen_pipeline" && "$qwen_mask" == COMPLETED* && "$qwen_eval" == COMPLETED* ]] && \
      gpu_test_has_slot; then
    qwen_pipeline="$(submit_pipeline qwen25_7b qwen)"
    printf 'time=%s submitted_model=qwen25_7b pipeline=%s\n' \
      "$(date -Is)" "$qwen_pipeline"
  fi
  if [[ -z "$llama_pipeline" && "$llama_mask" == COMPLETED* && "$llama_eval" == COMPLETED* ]] && \
      gpu_test_has_slot; then
    llama_pipeline="$(submit_pipeline llama31_8b llama)"
    printf 'time=%s submitted_model=llama31_8b pipeline=%s\n' \
      "$(date -Is)" "$llama_pipeline"
  fi
  [[ -n "$llama_pipeline" && -n "$qwen_pipeline" ]] || sleep "$POLL_SECONDS"
done
printf 'time=%s llama_pipeline=%s qwen_pipeline=%s\n' \
  "$(date -Is)" "$llama_pipeline" "$qwen_pipeline"
