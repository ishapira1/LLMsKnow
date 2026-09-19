#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"
require_runtime

POLL_SECONDS="${POLL_SECONDS:-30}"
ACCOUNT="${BONHAM_ACCOUNT:-barak_lab}"
ACCOUNTING_START="${BONHAM_ACCOUNTING_START:-2026-09-19}"
USER_NAME="${USER:-ishapira}"
RECEIPT="$RESULT_ROOT/reports/early_qwen_llama/COMPLETE.json"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
}

mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/early_report"
supervisor_log="$LOG_ROOT/submit/early_qwen_llama_report_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$supervisor_log") 2>&1
log() { printf '%s time=%s\n' "$*" "$(date -Is)" >&2; }

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

paper_core_complete() {
  local model family
  for model in qwen25_7b llama31_8b; do
    for family in generalization useful_assertions source_attribution; do
      family_complete "$model" "$family" || return 1
    done
  done
}

job_state() {
  local job_id="$1"
  sacct -X -n -j "$job_id" --parsable2 --format=State 2>/dev/null \
    | cut -d'|' -f1 | head -1
}

latest_job() {
  {
    squeue -h -u "$USER_NAME" -n bonh_early_report -o '%A' 2>/dev/null || true
    sacct -S "$ACCOUNTING_START" -X -n --name bonh_early_report \
      --parsable2 --format=JobIDRaw 2>/dev/null | cut -d'|' -f1 || true
  } | grep -E '^[0-9]+$' | sort -n | tail -1 || true
}

submit_report() {
  local raw job_id
  raw="$(sbatch --parsable --account="$ACCOUNT" --partition=serial_requeue \
    --job-name=bonh_early_report --time=02:00:00 --cpus-per-task=8 --mem=96G \
    --export="ALL,STAGE=early_report,MODEL_KEY=shared,BONHAM_BUNDLE_DIR=$BUNDLE_DIR" \
    --output="$LOG_ROOT/slurm/early_report/%x_%j.out" \
    --error="$LOG_ROOT/slurm/early_report/%x_%j.err" \
    "$BUNDLE_DIR/cpu_stage.sbatch")"
  job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'Unexpected early-report sbatch response: %s\n' "$raw" >&2
    return 2
  }
  printf '%s\n' "$job_id"
}

log "early_qwen_llama_report_supervisor_start commit=$(git -C "$REPO_DIR" rev-parse --short HEAD)"
while [[ ! -f "$RECEIPT" ]] && ! paper_core_complete; do
  log 'waiting_for_qwen_llama_paper_core=1'
  sleep "$POLL_SECONDS"
done
if [[ -f "$RECEIPT" ]]; then
  log "early_report_already_complete receipt=$RECEIPT"
  exit 0
fi

job_id="$(latest_job)"
state="$(job_state "$job_id")"
case "$state" in
  PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)
    log "reuse_early_report_job job_id=$job_id state=$state"
    ;;
  COMPLETED)
    [[ -f "$RECEIPT" ]] || {
      printf 'Completed early-report job lacks its authenticated receipt: %s\n' "$job_id" >&2
      exit 1
    }
    ;;
  *)
    job_id="$(submit_report)"
    log "submitted_early_report job_id=$job_id"
    ;;
esac

while [[ ! -f "$RECEIPT" ]]; do
  state="$(job_state "$job_id")"
  case "$state" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|'')
      sleep "$POLL_SECONDS"
      ;;
    COMPLETED)
      printf 'Early-report job completed without its authenticated receipt: %s\n' "$job_id" >&2
      exit 1
      ;;
    *)
      printf 'Early-report job failed: %s state=%s\n' "$job_id" "$state" >&2
      exit 1
      ;;
  esac
done
log "early_qwen_llama_report_complete receipt=$RECEIPT job_id=$job_id"
