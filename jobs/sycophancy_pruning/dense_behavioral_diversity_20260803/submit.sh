#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BUNDLE_DIR/common.sh"
DRY_RUN="${DRY_RUN:-1}"
SUBMISSION_RECORD="$RESULT_ROOT/submission_record.json"
mkdir -p "$LOG_ROOT/submit"

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ -e "$SUBMISSION_RECORD" ]]; then
    printf 'Refusing duplicate submission: %s\n' "$SUBMISSION_RECORD" >&2
    exit 2
  fi
  cluster="$(scontrol show config | sed -n 's/^ClusterName *= *//p' | head -n 1)"
  [[ "$cluster" == "odyssey" ]]
  if [[ -e "$RESULT_ROOT" && -n "$(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Refusing colliding non-empty result root: %s\n' "$RESULT_ROOT" >&2
    exit 2
  fi
  mkdir -p "$RESULT_ROOT"
fi

timestamp="$(date +%Y%m%dT%H%M%S%z)"
submit_log="$LOG_ROOT/submit/submit_${timestamp}_pid_$$.log"

submit() {
  local label="$1" script="$2" array="$3" dependency="$4" resource="${5:-cpu}"
  mkdir -p "$LOG_ROOT/slurm/$label"
  local command=(
    sbatch --parsable
    --job-name="syco_dense_${label}_0803"
    --export="ALL,DENSE_BUNDLE_DIR=$BUNDLE_DIR,DENSE_STAGE=$label"
    --output="$LOG_ROOT/slurm/$label/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/$label/%x.%A_%a.err"
  )
  [[ -n "$array" ]] && command+=(--array="$array")
  [[ -n "$dependency" ]] && command+=(--dependency="afterok:$dependency")
  if [[ "$resource" == "h200" ]]; then
    command+=(--partition=gpu_h200,seas_gpu,gpu --gres=gpu:1)
  fi
  command+=("$script")
  printf '%s_command=' "$label" >&2; printf '%q ' "${command[@]}" >&2; printf '\n' >&2
  if [[ "$DRY_RUN" == "1" ]]; then printf 'dry_%s\n' "$label"; else "${command[@]}"; fi
}

main() {
  printf 'experiment=dense_behavioral_diversity_20260803\ndry_run=%s\nresult_root=%s\n' "$DRY_RUN" "$RESULT_ROOT"
  preflight_job="$(submit preflight "$BUNDLE_DIR/cpu_stage.sbatch" "" "")"
  candidates_job="$(submit build_candidates "$BUNDLE_DIR/cpu_stage.sbatch" "" "$preflight_job")"
  sample_job="$(submit sample "$BUNDLE_DIR/gpu_stage.sbatch" "0-3%4" "$candidates_job" h200)"
  manifests_job="$(submit build_manifests "$BUNDLE_DIR/cpu_stage.sbatch" "" "$sample_job")"
  manifests_mail_job="$(submit email_manifests "$BUNDLE_DIR/cpu_stage.sbatch" "" "$manifests_job")"
  score_job="$(submit score "$BUNDLE_DIR/gpu_stage.sbatch" "0-7%4" "$manifests_mail_job" h200)"
  masks_job="$(submit build_masks "$BUNDLE_DIR/cpu_stage.sbatch" "0-3%4" "$score_job")"
  tune_base_job="$(submit tune_base "$BUNDLE_DIR/gpu_stage.sbatch" "" "$masks_job" h200)"
  replicate_behavior_job="$(submit replicate_behavior "$BUNDLE_DIR/cpu_stage.sbatch" "" "$tune_base_job")"
  tune_grid_job="$(submit tune_grid "$BUNDLE_DIR/gpu_stage.sbatch" "0-35%4" "$replicate_behavior_job" h200)"
  tune_matched_job="$(submit tune_matched "$BUNDLE_DIR/gpu_stage.sbatch" "0-3%4" "$tune_grid_job" h200)"
  utility_base_job="$(submit utility_base "$BUNDLE_DIR/gpu_stage.sbatch" "" "$tune_matched_job" h200)"
  replicate_utility_job="$(submit replicate_utility "$BUNDLE_DIR/cpu_stage.sbatch" "" "$utility_base_job")"
  utility_grid_job="$(submit utility_grid "$BUNDLE_DIR/gpu_stage.sbatch" "0-35%4" "$replicate_utility_job" h200)"
  utility_matched_job="$(submit utility_matched "$BUNDLE_DIR/gpu_stage.sbatch" "0-3%4" "$utility_grid_job" h200)"
  select_job="$(submit select "$BUNDLE_DIR/cpu_stage.sbatch" "" "$utility_matched_job")"
  screening_mail_job="$(submit email_screening "$BUNDLE_DIR/cpu_stage.sbatch" "" "$select_job")"
  states_job="$(submit build_states "$BUNDLE_DIR/cpu_stage.sbatch" "" "$screening_mail_job")"
  final_suite_job="$(submit final_suite "$BUNDLE_DIR/gpu_stage.sbatch" "0-6%4" "$states_job" h200)"
  elephant_job="$(submit score_elephant "$BUNDLE_DIR/cpu_stage.sbatch" "" "$final_suite_job")"
  aggregate_job="$(submit aggregate "$BUNDLE_DIR/cpu_stage.sbatch" "" "$elephant_job")"
  final_mail_job="$(submit email_final "$BUNDLE_DIR/cpu_stage.sbatch" "" "$aggregate_job")"

  if [[ "$DRY_RUN" != "1" ]]; then
    temporary="$SUBMISSION_RECORD.tmp.$$"
    "$PYTHON_BIN" - "$timestamp" "$submit_log" "$preflight_job" "$candidates_job" "$sample_job" "$manifests_job" "$score_job" "$masks_job" "$tune_grid_job" "$utility_grid_job" "$select_job" "$states_job" "$final_suite_job" "$aggregate_job" "$final_mail_job" > "$temporary" <<'PY'
import json,sys
names=("submitted_at","submit_log","preflight","build_candidates","sample","build_manifests","score","build_masks","tune_grid","utility_grid","select","build_states","final_suite","aggregate","final_mail")
payload={"status":"submitted","experiment":"dense_behavioral_diversity_20260803","gpu_concurrency_cap":4,"stale_lock_cleanup":False,**dict(zip(names,sys.argv[1:]))}
print(json.dumps(payload,indent=2,sort_keys=True))
PY
    mv "$temporary" "$SUBMISSION_RECORD"
    "$PYTHON_BIN" "$BUNDLE_DIR/email_status.py" --stage submitted --result-root "$RESULT_ROOT" --email-to "$EMAIL_TO"
    for job in "$sample_job" "$score_job" "$tune_base_job" "$tune_grid_job" "$tune_matched_job" "$utility_base_job" "$utility_grid_job" "$utility_matched_job" "$final_suite_job"; do
      scontrol top "$job" >/dev/null 2>&1 || true
    done
  fi
  printf 'sample_job=%s\nscore_job=%s\ntune_grid_job=%s\nutility_grid_job=%s\nfinal_suite_job=%s\nfinal_mail_job=%s\n' "$sample_job" "$score_job" "$tune_grid_job" "$utility_grid_job" "$final_suite_job" "$final_mail_job"
}

main "$@" 2>&1 | tee -a "$submit_log"
