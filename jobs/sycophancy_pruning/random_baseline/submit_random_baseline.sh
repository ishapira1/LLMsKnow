#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DRY_RUN="${DRY_RUN:-1}"
PHASE="${PHASE:-validation}"
SUBMISSION_ATTEMPT="${SUBMISSION_ATTEMPT:-initial}"
[[ "$SUBMISSION_ATTEMPT" =~ ^[A-Za-z0-9_-]+$ ]] || {
  printf 'Unsafe SUBMISSION_ATTEMPT\n' >&2; exit 2;
}
mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm"/{preflight,mask_builder,mask_audit,smoke,parity,core,aggregate_core,broad,packaging}
submit_log="$LOG_ROOT/submit/submit_${PHASE}_$(date +%Y%m%dT%H%M%S)_${DRY_RUN}.log"
exec >>"$submit_log" 2>&1

if [[ "$DRY_RUN" != 1 && "$DRY_RUN" != 0 ]]; then
  printf 'DRY_RUN must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$PHASE" != validation && "$PHASE" != campaign ]]; then
  printf 'PHASE must be validation or campaign\n' >&2
  exit 2
fi
dry_marker="$LOG_ROOT/submit/DRY_RUN_${PHASE}_COMPLETE"
current_commit="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ "$DRY_RUN" == 0 && ( ! -f "$dry_marker" || "$(tr -d '[:space:]' < "$dry_marker")" != "$current_commit" ) ]]; then
  printf 'Refusing submission: run PHASE=%s DRY_RUN=1 first.\n' "$PHASE" >&2
  exit 2
fi

submit_job() {
  local stage="$1" script="$2" array="$3" dependency="$4" name="$5"
  local log_stage="$6"
  local command=(sbatch --parsable --job-name="$name"
    --output="$LOG_ROOT/slurm/$log_stage/%A_%a.out"
    --error="$LOG_ROOT/slurm/$log_stage/%A_%a.err"
    --export="ALL,STAGE=$stage,RANDOM_BASELINE_BUNDLE_DIR=$BUNDLE_DIR,REPO_DIR=$REPO_DIR,RESULT_ROOT=$RESULT_ROOT,LOG_ROOT=$LOG_ROOT,HF_CACHE_DIR=$HF_CACHE_DIR,PYTHON_BIN=$PYTHON_BIN,SYCOBENCH_SOURCE=$SYCOBENCH_SOURCE")
  if [[ -n "$array" ]]; then command+=(--array="$array"); fi
  if [[ -n "$dependency" ]]; then command+=(--dependency="afterok:$dependency"); fi
  command+=("$script")
  printf '%q ' "${command[@]}" >&2; printf '\n' >&2
  if [[ "$DRY_RUN" == 1 ]]; then
    local checksum
    checksum="$(printf '%s' "$name" | cksum)"
    checksum="${checksum%% *}"
    printf '%s\n' "$((900000 + checksum % 90000))"
  else
    local submitted
    submitted="$("${command[@]}")"
    printf '%s\n' "${submitted%%;*}"
  fi
}

if [[ "$PHASE" == validation ]]; then
  preflight_job="$(submit_job preflight "$BUNDLE_DIR/cpu_stage.sbatch" "" "" rb_preflight_20260803 preflight | tail -1)"
  builder_job="$(submit_job build_masks "$BUNDLE_DIR/gpu_stage.sbatch" '0-1%2' "$preflight_job" rb_masks_20260803 mask_builder | tail -1)"
  audit_job="$(submit_job audit_masks "$BUNDLE_DIR/cpu_stage.sbatch" '0-1%2' "$builder_job" rb_audit_20260803 mask_audit | tail -1)"
  audit_email_job="$(submit_job audit_complete "$BUNDLE_DIR/cpu_stage.sbatch" "" "$audit_job" rb_auditmail_20260803 mask_audit | tail -1)"
  smoke_job="$(submit_job smoke "$BUNDLE_DIR/gpu_stage.sbatch" '0-1%2' "$audit_email_job" rb_smoke_20260803 smoke | tail -1)"
  parity_job="$(submit_job parity "$BUNDLE_DIR/gpu_stage.sbatch" '0-1%2' "$smoke_job" rb_parity_20260803 parity | tail -1)"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%s\n' "$current_commit" > "$dry_marker"
    printf 'dry_run_complete=1\nvalidation_final_job=%s\n' "$parity_job"
  else
    validation_milestone="validation_submission"
    if [[ "$SUBMISSION_ATTEMPT" != initial ]]; then
      validation_milestone="validation_submission_${SUBMISSION_ATTEMPT}"
    fi
    send_progress_email "$validation_milestone" "[random_baseline] validation phase submitted ($SUBMISSION_ATTEMPT)" \
      "Validation submitted. Preflight=$preflight_job, masks=$builder_job, audit=$audit_job, smoke=$smoke_job, parity=$parity_job."
    printf 'validation_final_job=%s\n' "$parity_job"
  fi
  exit 0
fi

if [[ "$DRY_RUN" == 0 ]]; then
  for model in llama qwen; do
    for artifact in "$RESULT_ROOT/audit/${model}_masks.json" \
                    "$RESULT_ROOT/audit/${model}_gpu_smoke.json" \
                    "$RESULT_ROOT/audit/${model}_batch_parity.json"; do
      [[ -f "$artifact" ]] || { printf 'Validation artifact missing: %s\n' "$artifact" >&2; exit 1; }
    done
    selected_batch="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch_size"])' "$RESULT_ROOT/audit/${model}_batch_parity.json")"
    [[ "$selected_batch" -eq 1 ]] || { printf 'Unexpected batch decision for %s\n' "$model" >&2; exit 1; }
  done
fi

core_job="$(submit_job core "$BUNDLE_DIR/gpu_stage.sbatch" '0-21%4' "" rb_core_20260803 core | tail -1)"
core_agg_job="$(submit_job aggregate_core "$BUNDLE_DIR/cpu_stage.sbatch" '0-1%2' "$core_job" rb_coreagg_20260803 aggregate_core | tail -1)"

llama_email_job="$(submit_job llama_core_email "$BUNDLE_DIR/cpu_stage.sbatch" "" "${core_agg_job}_0" rb_llamamail_20260803 aggregate_core | tail -1)"
qwen_email_job="$(submit_job qwen_core_email "$BUNDLE_DIR/cpu_stage.sbatch" "" "${core_agg_job}_1" rb_qwenmail_20260803 aggregate_core | tail -1)"

broad_dependency="$llama_email_job:$qwen_email_job"
broad_job="$(submit_job broad "$BUNDLE_DIR/gpu_stage.sbatch" '0-143%4' "$broad_dependency" rb_broad_20260803 broad | tail -1)"
broad_email_job="$(submit_job broad_complete "$BUNDLE_DIR/cpu_stage.sbatch" "" "$broad_job" rb_broadmail_20260803 broad | tail -1)"
package_job="$(submit_job package "$BUNDLE_DIR/cpu_stage.sbatch" "" "$broad_email_job" rb_package_20260803 packaging | tail -1)"

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'dry_run_complete=1\nphase1_final_job=%s\n' "$package_job"
  printf '%s\n' "$current_commit" > "$dry_marker"
else
  send_progress_email submission "[random_baseline] campaign submitted" \
    "Validated campaign submitted. Core=$core_job, broad=$broad_job, package=$package_job. Concurrency is capped at four GPUs."
  printf 'phase1_final_job=%s\n' "$package_job"
fi
