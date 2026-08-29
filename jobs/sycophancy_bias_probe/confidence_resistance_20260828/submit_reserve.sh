#!/usr/bin/env bash
set -Eeuo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export CONFIDENCE_RESISTANCE_BUNDLE_DIR="$BUNDLE_DIR"
source "$BUNDLE_DIR/common.sh"
require_runner
DRY_RUN="${DRY_RUN:-1}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { printf 'DRY_RUN must be 0 or 1\n' >&2; exit 2; }
coverage_audit="$RESULT_ROOT/audit/measurement_coverage.json"
if [[ -f "$coverage_audit" ]]; then
  coverage_passed="$("$PYTHON_BIN" -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["passed"])))' "$coverage_audit")"
  [[ "$coverage_passed" == 0 ]] || { printf 'Initial coverage already passes; reserve inference is unnecessary.\n'; exit 0; }
elif [[ "$DRY_RUN" != 1 || "${ALLOW_RESERVE_DRY_RUN_WITHOUT_AUDIT:-0}" != 1 ]]; then
  printf 'Run the initial measurement coverage audit before submitting reserve inference.\n' >&2
  exit 2
fi
if [[ "$DRY_RUN" == 0 ]]; then
  [[ "${OPENAI_CONFIRM_SPEND:-0}" == 1 ]] || { printf 'Set OPENAI_CONFIRM_SPEND=1 after reviewing reserve request counts.\n' >&2; exit 2; }
  [[ "${PRICING_RECHECKED:-0}" == 1 ]] || { printf 'Recheck official OpenAI Batch pricing and set PRICING_RECHECKED=1.\n' >&2; exit 2; }
fi
mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/reserve_neutral" \
  "$LOG_ROOT/slurm/reserve_freeze" "$LOG_ROOT/slurm/reserve_endorsed" \
  "$LOG_ROOT/slurm/reserve_gpt" "$LOG_ROOT/slurm/reserve_select"
submit_log="$LOG_ROOT/submit/reserve_submit_$(date -u +%Y%m%dT%H%M%SZ).log"

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

gpt_job="$(submit reserve_gpt "$BUNDLE_DIR/gpt_reserve_pipeline.sbatch")"
hf_neutral_job="$(submit reserve_hf_neutral "$BUNDLE_DIR/hf_reserve_neutral_array.sbatch")"
freeze_job="$(submit reserve_freeze "$BUNDLE_DIR/freeze_hf_reserve_targets.sbatch" "afterok:$hf_neutral_job")"
hf_endorsed_job="$(submit reserve_hf_endorsed "$BUNDLE_DIR/hf_reserve_endorsed_array.sbatch" "afterok:$freeze_job")"
select_job="$(submit reserve_select "$BUNDLE_DIR/select_reserve.sbatch" "afterok:$gpt_job:$hf_endorsed_job")"
{
  printf 'experiment=%s\ndry_run=%s\nresult_root=%s\nlog_root=%s\n' \
    cross_model_confidence_resistance_20260828 "$DRY_RUN" "$RESULT_ROOT" "$LOG_ROOT"
  printf 'reserve_gpt_job=%s\nreserve_hf_neutral_job=%s\nreserve_freeze_job=%s\nreserve_hf_endorsed_job=%s\nreserve_select_job=%s\n' \
    "$gpt_job" "$hf_neutral_job" "$freeze_job" "$hf_endorsed_job" "$select_job"
} | tee -a "$submit_log"
