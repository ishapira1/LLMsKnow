#!/bin/bash
set -euo pipefail

BUNDLE_NAME="full_allfamilies_paraphrase_sharded_20260618"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
source jobs/sycophancy_bias_probe/storage_common.sh
configure_sycophancy_bias_storage "$BUNDLE_NAME"

SUBMIT_LOG_DIR="$LOG_ROOT/submit"
SUBMISSION_ENV_FILE="${1:-$SUBMIT_LOG_DIR/latest_submission.env}"

if [[ ! -f "$SUBMISSION_ENV_FILE" ]]; then
  printf '%s\n' "Missing submission metadata file: $SUBMISSION_ENV_FILE" >&2
  exit 1
fi

set -a
source "$SUBMISSION_ENV_FILE"
set +a

printf '[status-20260618] bundle=%s\n' "$BUNDLE_NAME"
printf '[status-20260618] submission_env=%s\n' "$SUBMISSION_ENV_FILE"
printf '[status-20260618] log_root=%s\n' "$LOG_ROOT"
printf '[status-20260618] storage_root=%s\n' "$SYCOPHANCY_STORAGE_ROOT"
printf '[status-20260618] submitted_at=%s\n' "${SUBMITTED_AT:-unknown}"
printf '[status-20260618] sampling_job_id=%s probe_job_id=%s\n' "${SAMPLING_JOB_ID:-unknown}" "${PROBE_JOB_ID:-unknown}"
printf '[status-20260618] sampling_task_matrix=%s\n' "${SAMPLING_TASK_MATRIX:-missing}"
printf '[status-20260618] probe_task_matrix=%s\n' "${PROBE_TASK_MATRIX:-missing}"
printf '[status-20260618] submit_log_file=%s\n' "${SUBMIT_LOG_FILE:-missing}"

if command -v squeue >/dev/null 2>&1; then
  printf '[status-20260618] squeue active rows:\n'
  squeue -j "${SAMPLING_JOB_ID:-0},${PROBE_JOB_ID:-0}" -o "%.18i %.10T %.20j %.8u %.10M %.9l %.6D %R" || true
fi

if command -v sacct >/dev/null 2>&1; then
  printf '[status-20260618] sacct sampling summary:\n'
  sacct -j "${SAMPLING_JOB_ID:-0}" --format=JobIDRaw,JobName,State,ExitCode,Elapsed -n -P || true
  printf '[status-20260618] sacct probe summary:\n'
  sacct -j "${PROBE_JOB_ID:-0}" --format=JobIDRaw,JobName,State,ExitCode,Elapsed -n -P || true
fi

STATUS_PYTHON="${STATUS_PYTHON:-python}"
"$STATUS_PYTHON" - "${SAMPLING_TASK_MATRIX:-}" "${PROBE_TASK_MATRIX:-}" <<'PY'
import csv
import json
import sys
from pathlib import Path


def summarize(label: str, path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        print(f"[status-20260618] {label}: missing task matrix {path}")
        return

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    status_counts = {}
    missing = 0
    first_problem = None
    for row in rows:
        run_dir = Path(row["run_dir"])
        status_path = run_dir / "meta" / "status.json"
        if not status_path.exists():
            missing += 1
            if first_problem is None:
                first_problem = f"missing_status::{run_dir}"
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            key = f"unreadable_status::{type(exc).__name__}"
            status_counts[key] = status_counts.get(key, 0) + 1
            if first_problem is None:
                first_problem = f"{key}::{run_dir}"
            continue
        status_value = str(payload.get("status", "unknown") or "unknown")
        status_counts[status_value] = status_counts.get(status_value, 0) + 1
        if status_value != "completed" and first_problem is None:
            first_problem = f"{status_value}::{run_dir}"

    print(
        f"[status-20260618] {label}:"
        f" expected={len(rows)}"
        f" missing_status={missing}"
        + "".join(f" {key}={status_counts[key]}" for key in sorted(status_counts))
    )
    if first_problem is not None:
        print(f"[status-20260618] {label}: first_noncompleted={first_problem}")


summarize("sampling_runs", sys.argv[1])
summarize("probe_runs", sys.argv[2])
PY
