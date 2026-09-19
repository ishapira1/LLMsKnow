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

latest_job_id_by_name() {
  sacct -S 2026-09-19 -X -n --name "$1" --parsable2 --format=JobIDRaw 2>/dev/null \
    | cut -d'|' -f1 | grep -E '^[0-9]+$' | sort -n | tail -1 || true
}

source_model_complete() {
  local model="$1" shard_count state observed
  local -a states=(
    unpruned n1_mechanism n2_selective random_n1
    random_n2 weak_prompt strong_prompt prompt_only_meandiff
  )
  shard_count="$(wc -l < "$RESULT_ROOT/evaluations/inputs/$model/source_attribution/index.jsonl" | tr -d ' ')"
  for state in "${states[@]}"; do
    observed="$(find "$RESULT_ROOT/evaluations/results/$model/$state/source_attribution" \
      -type f -name COMPLETE 2>/dev/null | wc -l | tr -d ' ')"
    (( observed >= shard_count )) || return 1
  done
}

primary_gpu_tail_complete() {
  local gemma_pipeline_id
  source_model_complete qwen25_7b || return 1
  source_model_complete llama31_8b || return 1
  gemma_pipeline_id="$(latest_job_id_by_name bonh_gemma_pipeline)"
  [[ -n "$gemma_pipeline_id" && "$(job_state "$gemma_pipeline_id")" == COMPLETED* ]]
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
llama_attempts=0
qwen_attempts=0

fallback_terminal_incomplete() {
  local job_id="$1" state
  [[ -n "$job_id" ]] || return 1
  state="$(job_state "$job_id")"
  case "$state" in
    COMPLETED*|FAILED*|CANCELLED*|OUT_OF_MEMORY*|TIMEOUT*|NODE_FAIL*) return 0 ;;
    *) return 1 ;;
  esac
}

maybe_recover_failed_fallback() {
  local model_key="$1" fallback_name="$2" attempts_name="$3"
  local fallback_id="${!fallback_name}" attempts="${!attempts_name}"
  [[ -n "$fallback_id" ]] || return 0
  fallback_terminal_incomplete "$fallback_id" || return 0
  if (( $(score_count "$model_key") == 7 )); then
    return 0
  fi
  attempts=$((attempts + 1))
  if (( attempts >= 2 )); then
    printf 'fallback_failure model=%s job=%s attempts=%s\n' \
      "$model_key" "$fallback_id" "$attempts" >&2
    exit 1
  fi
  printf 'time=%s recovering_incomplete_fallback model=%s job=%s\n' \
    "$(date -Is)" "$model_key" "$fallback_id"
  printf -v "$fallback_name" '%s' ''
  printf -v "$attempts_name" '%s' "$attempts"
}

maybe_submit_model() {
  local model_key="$1" short="$2" array_job_id="$3" fallback_name="$4"
  local fallback_id="${!fallback_name}"
  [[ -z "$fallback_id" ]] || return 1
  (( $(score_count "$model_key") < 7 )) || return 1
  (( $(active_array_tasks "$array_job_id") == 0 )) || return 1
  primary_gpu_tail_complete || return 1
  gpu_test_has_slot || return 1
  scancel "$array_job_id" 2>/dev/null || true
  fallback_id="$(submit_fallback "$model_key" "$short")"
  printf -v "$fallback_name" '%s' "$fallback_id"
  printf 'time=%s submitted_model=%s fallback=%s\n' \
    "$(date -Is)" "$model_key" "$fallback_id"
  return 0
}

# Stay alive through score completion, not merely submission.  Once either
# packed evaluation pipeline releases a gpu_test slot, keep that slot occupied
# by whichever model still needs analysis.  The second model's scores are
# independent of its evaluation and can therefore run concurrently with it.
while :; do
  llama_state="$(job_state "$LLAMA_PIPELINE_JOB_ID")"
  qwen_state="$(job_state "$QWEN_PIPELINE_JOB_ID")"
  llama_scores="$(score_count llama31_8b)"
  qwen_scores="$(score_count qwen25_7b)"
  combined="$llama_state:$qwen_state"
  printf 'time=%s llama_pipeline=%s qwen_pipeline=%s llama_scores=%s qwen_scores=%s llama_analysis=%s qwen_analysis=%s\n' \
    "$(date -Is)" "$llama_state" "$qwen_state" \
    "$llama_scores" "$qwen_scores" \
    "${llama_fallback:-waiting}" "${qwen_fallback:-waiting}"
  case "$combined" in
    *FAILED*|*CANCELLED*|*OUT_OF_MEMORY*|*TIMEOUT*|*NODE_FAIL*)
      printf 'pipeline_failure=%s\n' "$combined" >&2
      exit 1
      ;;
  esac

  (( llama_scores == 7 && qwen_scores == 7 )) && break

  maybe_recover_failed_fallback qwen25_7b qwen_fallback qwen_attempts
  maybe_recover_failed_fallback llama31_8b llama_fallback llama_attempts

  # Prefer the model whose paper pipeline has already finished.  If its
  # regular analysis array is running, immediately offer the free slot to the
  # other model rather than leaving expensive capacity idle.
  submitted=0
  if [[ "$qwen_state" == COMPLETED* ]]; then
    maybe_submit_model qwen25_7b qwen "$QWEN_ANALYSIS_ARRAY_JOB_ID" qwen_fallback && submitted=1 || true
  fi
  if (( submitted == 0 )) && [[ "$llama_state" == COMPLETED* ]]; then
    maybe_submit_model llama31_8b llama "$LLAMA_ANALYSIS_ARRAY_JOB_ID" llama_fallback && submitted=1 || true
  fi
  if (( submitted == 0 )); then
    maybe_submit_model qwen25_7b qwen "$QWEN_ANALYSIS_ARRAY_JOB_ID" qwen_fallback && submitted=1 || true
  fi
  if (( submitted == 0 )); then
    maybe_submit_model llama31_8b llama "$LLAMA_ANALYSIS_ARRAY_JOB_ID" llama_fallback && submitted=1 || true
  fi

  sleep "$POLL_SECONDS"
done
printf 'time=%s llama_scores=7 qwen_scores=7 llama_analysis=%s qwen_analysis=%s\n' \
  "$(date -Is)" "${llama_fallback:-regular_array}" "${qwen_fallback:-regular_array}"
