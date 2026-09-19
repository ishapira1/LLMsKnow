#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

POLL_SECONDS="${POLL_SECONDS:-60}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
GPU_TEST_PARTITION="${BONHAM_GPU_TEST_PARTITION:-gpu_test}"
GPU_TEST_GRES="${BONHAM_GPU_TEST_GRES:-gpu:nvidia_a100_3g.20gb}"
ACCOUNTING_START="${BONHAM_ACCOUNTING_START:-2026-09-19}"
USER_NAME="${USER:-ishapira}"
MODEL_KEY=gemma4_12b
SUPPLEMENT_FIRST=191
SUPPLEMENT_LAST=205

mkdir -p \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/gpu_multilane" \
  "$LOG_ROOT/slurm/gpu_score_multilane" \
  "$LOG_ROOT/slurm/gpu_model_pipeline" \
  "$LOG_ROOT/slurm/allocate_model_manifests" \
  "$LOG_ROOT/slurm/eval_prepare_model" \
  "$LOG_ROOT/slurm/steering_prepare" \
  "$LOG_ROOT/slurm/build_masks"
supervisor_log="$LOG_ROOT/submit/gemma_exact_$(date +%Y%m%dT%H%M%S).log"
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

wait_job() {
  local label="$1" job_id="$2" state
  while true; do
    state="$(job_state "$job_id")"
    case "$state" in
      COMPLETED) log "stage_complete label=$label job_id=$job_id"; return 0 ;;
      PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|'')
        sleep "$POLL_SECONDS" ;;
      *) log "stage_failure label=$label job_id=$job_id state=$state"; return 1 ;;
    esac
  done
}

gpu_test_job_count() {
  squeue -h -u "$USER_NAME" -p "$GPU_TEST_PARTITION" | wc -l | tr -d ' '
}

gpu_test_clear() {
  (( $(gpu_test_job_count) == 0 ))
}

supplement_receipts_complete() {
  local index stem root
  root="$RESULT_ROOT/n1_screen/$MODEL_KEY"
  for ((index = SUPPLEMENT_FIRST; index <= SUPPLEMENT_LAST; index++)); do
    stem="$(printf 'shard_%04d' "$index")"
    if [[ -f "$root/$stem/COMPLETE" ]]; then
      continue
    fi
    compgen -G "$root/$stem.partial.*/COMPLETE" >/dev/null || return 1
  done
}

score_receipts_complete() {
  [[ $(find "$RESULT_ROOT/scores/$MODEL_KEY" -mindepth 2 -maxdepth 2 \
      -type f -name COMPLETE.json 2>/dev/null | wc -l | tr -d ' ') == 7 ]]
}

submit_cpu() {
  local name="$1" stage="$2" memory="$3" time_limit="$4" existing state raw
  existing="$(job_id_by_name "$name")"
  state="$(job_state "$existing")"
  case "$state" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|COMPLETED)
      log "reuse_cpu_job name=$name job_id=$existing state=$state"
      printf '%s\n' "$existing"
      return 0
      ;;
  esac
  raw="$(sbatch --parsable --account="$ACCOUNT" --partition=test \
    --job-name="$name" --time="$time_limit" --cpus-per-task=8 --mem="$memory" \
    --export="ALL,STAGE=$stage,MODEL_KEY=$MODEL_KEY,BONHAM_BUNDLE_DIR=$BUNDLE_DIR" \
    --output="$LOG_ROOT/slurm/$stage/%x_%j.out" \
    --error="$LOG_ROOT/slurm/$stage/%x_%j.err" \
    "$BUNDLE_DIR/cpu_stage.sbatch")"
  printf '%s\n' "${raw%%;*}"
}

submit_screen_test() {
  local raw
  raw="$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_TEST_PARTITION" \
    --job-name=bonh_gem_suppscr --nodes=1 --ntasks=2 --cpus-per-task=4 \
    --mem=192G --time=04:00:00 --gres="$GPU_TEST_GRES:4" \
    --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,STAGE=n1_screen,MODEL_KEY=$MODEL_KEY,PACK_START=$SUPPLEMENT_FIRST,PACK_END=$SUPPLEMENT_LAST,LANES=2,GPUS_PER_LANE=2,CPUS_PER_LANE=4,MEM_PER_LANE=96G,SCREEN_BATCH_SIZE=4,SCREEN_INPUT_DIR=$RESULT_ROOT/inputs/n1_screen_model_supplement_shards/$MODEL_KEY" \
    --output="$LOG_ROOT/slurm/gpu_multilane/%x_%j.out" \
    --error="$LOG_ROOT/slurm/gpu_multilane/%x_%j.err" \
    "$BUNDLE_DIR/gpu_multilane.sbatch")"
  printf '%s\n' "${raw%%;*}"
}

wait_or_promote_supplement() {
  local job_id state partition
  while ! supplement_receipts_complete; do
    job_id="$(job_id_by_name bonh_gem_suppscr)"
    state="$(job_state "$job_id")"
    partition="$(job_partition "$job_id")"
    # The supplement occupies the entire four-slice capacity left by one
    # Qwen/Llama packed job.  Wait for a fully clear test partition so this
    # lower-priority recovery cannot consume the second submitted-job slot and
    # block the paper-critical source-attribution handoff.
    if [[ "$state" == PENDING && "$partition" != "$GPU_TEST_PARTITION" ]] && gpu_test_clear; then
      log "promote_supplement job_id=$job_id partition=$partition"
      scancel "$job_id"
      job_id="$(submit_screen_test)"
      log "submitted_supplement_test job_id=$job_id"
    elif [[ "$state" =~ ^(FAILED|OUT_OF_MEMORY|NODE_FAIL|TIMEOUT|PREEMPTED)$ ]]; then
      if gpu_test_clear; then
        job_id="$(submit_screen_test)"
        log "recovered_supplement_test job_id=$job_id prior_state=$state"
      else
        log "waiting_to_recover_supplement prior_state=$state"
      fi
    elif [[ "$state" == COMPLETED ]] && ! supplement_receipts_complete; then
      log "supplement_job_completed_without_all_receipts job_id=$job_id"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
  log "supplement_receipts_complete=1"
}

submit_score() {
  local target="$1" raw
  if [[ "$target" == test ]]; then
    raw="$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_TEST_PARTITION" \
      --job-name=bonh_gemma_scores --nodes=1 --ntasks=4 --cpus-per-task=4 \
      --mem=384G --time=12:00:00 --gres="$GPU_TEST_GRES:8" \
      --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$MODEL_KEY,LANES=4,GPUS_PER_LANE=2,CPUS_PER_LANE=4,MEM_PER_LANE=96G,BLOCKS_PER_PASS=2" \
      --output="$LOG_ROOT/slurm/gpu_score_multilane/%x_%j.out" \
      --error="$LOG_ROOT/slurm/gpu_score_multilane/%x_%j.err" \
      "$BUNDLE_DIR/gpu_score_multilane.sbatch")"
  else
    raw="$(sbatch --parsable --account="$ACCOUNT" --partition=gpu_h200,gpu_requeue \
      --job-name=bonh_gemma_scores --nodes=1 --ntasks=4 --cpus-per-task=4 \
      --mem=384G --time=24:00:00 --gres=gpu:nvidia_h200:4 \
      --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$MODEL_KEY,LANES=4,GPUS_PER_LANE=1,CPUS_PER_LANE=4,MEM_PER_LANE=96G,BLOCKS_PER_PASS=2" \
      --output="$LOG_ROOT/slurm/gpu_score_multilane/%x_%j.out" \
      --error="$LOG_ROOT/slurm/gpu_score_multilane/%x_%j.err" \
      "$BUNDLE_DIR/gpu_score_multilane.sbatch")"
  fi
  printf '%s\n' "${raw%%;*}"
}

wait_or_promote_scores() {
  local job_id state partition
  job_id="$(job_id_by_name bonh_gemma_scores)"
  state="$(job_state "$job_id")"
  case "$state" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|COMPLETED)
      log "reuse_score_job job_id=$job_id state=$state"
      ;;
    *)
      job_id="$(submit_score regular)"
      log "submitted_score_fallback job_id=$job_id"
      ;;
  esac
  while ! score_receipts_complete; do
    state="$(job_state "$job_id")"
    partition="$(job_partition "$job_id")"
    if [[ "$state" == PENDING && "$partition" != "$GPU_TEST_PARTITION" ]] && gpu_test_clear; then
      log "promote_scores job_id=$job_id partition=$partition"
      scancel "$job_id"
      job_id="$(submit_score test)"
      log "submitted_scores_test job_id=$job_id"
    elif [[ "$state" =~ ^(FAILED|OUT_OF_MEMORY|NODE_FAIL|TIMEOUT|PREEMPTED)$ ]]; then
      if gpu_test_clear; then
        job_id="$(submit_score test)"
        log "recovered_scores_test job_id=$job_id prior_state=$state"
      else
        log "waiting_to_recover_scores prior_state=$state"
      fi
    elif [[ "$state" == COMPLETED ]] && ! score_receipts_complete; then
      log "score_job_completed_without_all_receipts job_id=$job_id"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
  log "score_receipts_complete=1"
}

submit_pipeline() {
  local target="$1" raw
  if [[ "$target" == test ]]; then
    raw="$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_TEST_PARTITION" \
      --job-name=bonh_gemma_pipeline --nodes=1 --ntasks=4 --cpus-per-task=4 \
      --mem=384G --time=12:00:00 --gres="$GPU_TEST_GRES:8" \
      --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$MODEL_KEY,GPUS_PER_LANE=2,CPUS_PER_LANE=4,MEM_PER_LANE=96G,EVALUATION_BATCH_SIZE=4" \
      --output="$LOG_ROOT/slurm/gpu_model_pipeline/%x_%j.out" \
      --error="$LOG_ROOT/slurm/gpu_model_pipeline/%x_%j.err" \
      "$BUNDLE_DIR/gpu_model_pipeline.sbatch")"
  else
    raw="$(sbatch --parsable --account="$ACCOUNT" --partition=gpu_h200,gpu_requeue \
      --job-name=bonh_gemma_pipeline --nodes=1 --ntasks=4 --cpus-per-task=4 \
      --mem=384G --time=24:00:00 --gres=gpu:nvidia_h200:4 \
      --export="ALL,BONHAM_BUNDLE_DIR=$BUNDLE_DIR,MODEL_KEY=$MODEL_KEY,GPUS_PER_LANE=1,CPUS_PER_LANE=4,MEM_PER_LANE=96G,EVALUATION_BATCH_SIZE=4" \
      --output="$LOG_ROOT/slurm/gpu_model_pipeline/%x_%j.out" \
      --error="$LOG_ROOT/slurm/gpu_model_pipeline/%x_%j.err" \
      "$BUNDLE_DIR/gpu_model_pipeline.sbatch")"
  fi
  printf '%s\n' "${raw%%;*}"
}

wait_or_promote_pipeline() {
  local job_id state partition
  job_id="$(job_id_by_name bonh_gemma_pipeline)"
  state="$(job_state "$job_id")"
  case "$state" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|COMPLETED)
      log "reuse_pipeline_job job_id=$job_id state=$state"
      ;;
    *)
      job_id="$(submit_pipeline regular)"
      log "submitted_pipeline_fallback job_id=$job_id"
      ;;
  esac
  while true; do
    state="$(job_state "$job_id")"
    partition="$(job_partition "$job_id")"
    case "$state" in
      COMPLETED) log "gemma_pipeline_complete job_id=$job_id"; return 0 ;;
      PENDING)
        if [[ "$partition" != "$GPU_TEST_PARTITION" ]] && gpu_test_clear; then
          log "promote_pipeline job_id=$job_id partition=$partition"
          scancel "$job_id"
          job_id="$(submit_pipeline test)"
          log "submitted_pipeline_test job_id=$job_id"
        fi
        ;;
      RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|'') ;;
      *) log "gemma_pipeline_failure job_id=$job_id state=$state"; return 1 ;;
    esac
    sleep "$POLL_SECONDS"
  done
}

log "gemma_exact_supervisor_start commit=$(git -C "$REPO_DIR" rev-parse --short HEAD)"
wait_or_promote_supplement

manifest_receipt="$RESULT_ROOT/manifests/$MODEL_KEY/MANIFESTS_COMPLETE.json"
if [[ ! -f "$manifest_receipt" ]]; then
  allocation_job="$(submit_cpu bonh_gem_alloc_exact allocate_model_manifests 96G 02:00:00)"
  wait_job gemma_exact_allocation "$allocation_job"
fi
[[ -f "$manifest_receipt" ]] || { log "missing_manifest_receipt=1"; exit 1; }

if [[ ! -f "$RESULT_ROOT/evaluations/inputs/$MODEL_KEY/COMPLETE.json" ]]; then
  eval_prepare_job="$(submit_cpu bonh_gem_evalprep eval_prepare_model 96G 02:00:00)"
  wait_job gemma_eval_prepare "$eval_prepare_job"
fi
if [[ ! -f "$RESULT_ROOT/steering/$MODEL_KEY/inputs/paired_prompts.COMPLETE.json" ]]; then
  steering_prepare_job="$(submit_cpu bonh_gem_steerprep steering_prepare 96G 02:00:00)"
  wait_job gemma_steering_prepare "$steering_prepare_job"
fi

if ! score_receipts_complete; then
  wait_or_promote_scores
fi

mask_receipt="$RESULT_ROOT/masks/$MODEL_KEY/TARGET_MASKS_COMPLETE.json"
if [[ ! -f "$mask_receipt" ]]; then
  mask_job="$(job_id_by_name bonh_gemma4_12b_mask_a)"
  if [[ -z "$mask_job" || "$(job_state "$mask_job")" =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL)$ ]]; then
    mask_job="$(submit_cpu bonh_gem_mask_exact build_masks 120G 04:00:00)"
  fi
  wait_job gemma_masks "$mask_job"
fi
[[ -f "$mask_receipt" ]] || { log "missing_mask_receipt=1"; exit 1; }

wait_or_promote_pipeline
log "gemma_exact_supervisor_complete=1"
