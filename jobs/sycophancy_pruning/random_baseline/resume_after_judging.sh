#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DRY_RUN="${DRY_RUN:-1}"
labels="$RESULT_ROOT/judging/feedback_labels.jsonl"
packet="$RESULT_ROOT/judging/feedback_packet.jsonl"
mkdir -p "$LOG_ROOT/submit" "$LOG_ROOT/slurm/final"
if [[ "$DRY_RUN" == 0 && ! -f "$LOG_ROOT/submit/RESUME_DRY_RUN_COMPLETE" ]]; then
  printf 'Refusing final submission: run DRY_RUN=1 first.\n' >&2; exit 2
fi
if [[ ! -f "$packet" ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY RUN: packet does not exist yet: %s\n' "$packet"
  else
    printf 'Missing packet: %s\n' "$packet" >&2; exit 1
  fi
fi
if [[ ! -f "$labels" ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY RUN: labels must be uploaded to %s\n' "$labels"
  else
    printf 'Missing labels: %s\n' "$labels" >&2; exit 1
  fi
fi
command=(sbatch --parsable --job-name=rb_final_20260803
  --output="$LOG_ROOT/slurm/final/%A_0.out" --error="$LOG_ROOT/slurm/final/%A_0.err"
  --export="ALL,STAGE=final,RANDOM_BASELINE_BUNDLE_DIR=$BUNDLE_DIR,REPO_DIR=$REPO_DIR,RESULT_ROOT=$RESULT_ROOT,LOG_ROOT=$LOG_ROOT,HF_CACHE_DIR=$HF_CACHE_DIR,PYTHON_BIN=$PYTHON_BIN"
  "$BUNDLE_DIR/cpu_stage.sbatch")
printf '%q ' "${command[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  touch "$LOG_ROOT/submit/RESUME_DRY_RUN_COMPLETE"
  printf 'dry_run_complete=1\n'
else
  final_job="$("${command[@]}")"
  printf 'final_job=%s\n' "$final_job"
fi
