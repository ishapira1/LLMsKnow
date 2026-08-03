#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DRY_RUN="${DRY_RUN:-1}"
RECOVERY_STAGE="${RECOVERY_STAGE:?Set RECOVERY_STAGE}"
RECOVERY_ARRAY="${RECOVERY_ARRAY:?Set the exact failed array task or range}"
START_AFTER_JOB_ID="${START_AFTER_JOB_ID:-}"
RECOVERY_ID="${RECOVERY_ID:?Set a unique directory-safe RECOVERY_ID}"
[[ "$RECOVERY_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { printf 'Unsafe RECOVERY_ID\n' >&2; exit 2; }
[[ "$RECOVERY_ARRAY" =~ ^[0-9]+(-[0-9]+)?(%[0-9]+)?$ ]] || {
  printf 'Unsafe RECOVERY_ARRAY\n' >&2; exit 2;
}
case "$RECOVERY_STAGE" in
  build_masks|smoke|parity|core|broad) script="$BUNDLE_DIR/gpu_stage.sbatch" ;;
  audit_masks|aggregate_core|package|final) script="$BUNDLE_DIR/cpu_stage.sbatch" ;;
  *) printf 'Unsupported recovery stage: %s\n' "$RECOVERY_STAGE" >&2; exit 2 ;;
esac
recovery_root="$RESULT_ROOT/recovery/$RECOVERY_ID"
if [[ -e "$recovery_root" ]]; then
  printf 'Recovery ID already exists: %s\n' "$recovery_root" >&2; exit 1
fi
mkdir -p "$LOG_ROOT/submit/recovery" "$LOG_ROOT/slurm/recovery/$RECOVERY_STAGE"
command=(sbatch --parsable --job-name="rb_rec_${RECOVERY_STAGE:0:8}"
  --array="$RECOVERY_ARRAY"
  --output="$LOG_ROOT/slurm/recovery/$RECOVERY_STAGE/%A_%a.out"
  --error="$LOG_ROOT/slurm/recovery/$RECOVERY_STAGE/%A_%a.err"
  --export="ALL,STAGE=$RECOVERY_STAGE,RANDOM_BASELINE_BUNDLE_DIR=$BUNDLE_DIR,REPO_DIR=$REPO_DIR,RESULT_ROOT=$RESULT_ROOT,LOG_ROOT=$LOG_ROOT,HF_CACHE_DIR=$HF_CACHE_DIR,PYTHON_BIN=$PYTHON_BIN,SYCOBENCH_SOURCE=$SYCOBENCH_SOURCE,ALLOW_STALE_LOCK_CLEANUP=$ALLOW_STALE_LOCK_CLEANUP"
)
if [[ -n "$START_AFTER_JOB_ID" ]]; then
  [[ "$START_AFTER_JOB_ID" =~ ^[0-9]+$ ]] || { printf 'Unsafe START_AFTER_JOB_ID\n' >&2; exit 2; }
  command+=(--dependency="afterany:$START_AFTER_JOB_ID")
fi
command+=("$script")
printf '%q ' "${command[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'dry_run_complete=1\n'
  exit 0
fi
mkdir -p "$recovery_root"
job_id="$("${command[@]}")"
"$PYTHON_BIN" -c 'import json,sys,time,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":"submitted","stage":sys.argv[2],"array":sys.argv[3],"job_id":sys.argv[4],"submitted_at_epoch":int(time.time())},indent=2)+"\n")' \
  "$recovery_root/submission.json" "$RECOVERY_STAGE" "$RECOVERY_ARRAY" "${job_id%%;*}"
printf 'recovery_job=%s\n' "${job_id%%;*}"
