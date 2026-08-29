#!/usr/bin/env bash
set -Eeuo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export CONFIDENCE_RESISTANCE_BUNDLE_DIR="$BUNDLE_DIR"
source "$BUNDLE_DIR/common.sh"
require_runner
DRY_RUN="${DRY_RUN:-1}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { printf 'DRY_RUN must be 0 or 1\n' >&2; exit 2; }
if [[ "$DRY_RUN" == 0 ]]; then
  [[ "${OPENAI_CONFIRM_SPEND:-0}" == 1 ]] || { printf 'Set OPENAI_CONFIRM_SPEND=1 after reviewing the frozen request count.\n' >&2; exit 2; }
  [[ "${PRICING_RECHECKED:-0}" == 1 ]] || { printf 'Recheck official OpenAI Batch pricing and set PRICING_RECHECKED=1.\n' >&2; exit 2; }
fi
mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/prepare" "$LOG_ROOT/slurm/gpt" \
  "$LOG_ROOT/slurm/hf_neutral" "$LOG_ROOT/slurm/freeze_targets" \
  "$LOG_ROOT/slurm/hf_endorsed" "$LOG_ROOT/slurm/analysis" "$LOG_ROOT/slurm/confirmation"
submit_log="$LOG_ROOT/submit/submit_$(date -u +%Y%m%dT%H%M%SZ).log"

submit() {
  label="$1"
  script="$2"
  dependency="${3:-}"
  command=(sbatch --parsable --export=ALL,CONFIDENCE_RESISTANCE_BUNDLE_DIR="$BUNDLE_DIR",REPO_ROOT="$REPO_ROOT",RESULT_ROOT="$RESULT_ROOT",LOG_ROOT="$LOG_ROOT",HF_CACHE_DIR="$HF_CACHE_DIR")
  [[ -n "$dependency" ]] && command+=(--dependency="$dependency")
  command+=("$script")
  printf 'stage=%s command=' "$label" | tee -a "$submit_log" >&2
  printf '%q ' "${command[@]}" | tee -a "$submit_log" >&2
  printf '\n' | tee -a "$submit_log" >&2
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'dry_%s\n' "$label"
  else
    "${command[@]}" | tee -a "$submit_log"
  fi
}

prepare_job="$(submit prepare "$BUNDLE_DIR/prepare.sbatch")"
gpt_job="$(submit gpt "$BUNDLE_DIR/gpt_pipeline.sbatch" "afterok:$prepare_job")"
hf_neutral_job="$(submit hf_neutral "$BUNDLE_DIR/hf_neutral_array.sbatch" "afterok:$prepare_job")"
freeze_job="$(submit freeze_hf_targets "$BUNDLE_DIR/freeze_hf_targets.sbatch" "afterok:$hf_neutral_job")"
hf_endorsed_job="$(submit hf_endorsed "$BUNDLE_DIR/hf_endorsed_array.sbatch" "afterok:$freeze_job")"
discovery_job="$(submit discovery "$BUNDLE_DIR/discovery_analysis.sbatch" "afterok:$gpt_job:$hf_endorsed_job")"
{
  printf 'experiment=%s\ndry_run=%s\nresult_root=%s\nlog_root=%s\n' \
    cross_model_confidence_resistance_20260828 "$DRY_RUN" "$RESULT_ROOT" "$LOG_ROOT"
  printf 'prepare_job=%s\ngpt_job=%s\nhf_neutral_job=%s\nfreeze_job=%s\nhf_endorsed_job=%s\ndiscovery_job=%s\n' \
    "$prepare_job" "$gpt_job" "$hf_neutral_job" "$freeze_job" "$hf_endorsed_job" "$discovery_job"
  printf 'confirmation_submission=manual_after_discovery_review\n'
} | tee -a "$submit_log"
