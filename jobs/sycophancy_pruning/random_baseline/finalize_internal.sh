#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
labels="${1:?labels path required}"
[[ -f "$labels" ]] || { printf 'Missing feedback labels: %s\n' "$labels" >&2; exit 1; }
"$PYTHON_BIN" "$BUNDLE_DIR/package_feedback.py" aggregate --result-root "$RESULT_ROOT" \
  --private-key "$RESULT_ROOT/judging/feedback_private_key.jsonl" --labels "$labels"
"$PYTHON_BIN" "$BUNDLE_DIR/random_baseline.py" aggregate-final --result-root "$RESULT_ROOT"
send_progress_email final_report_complete "[random_baseline] final report complete" \
  "Final inference, feedback labels, broad-suite audit, and completion hashes are complete."
"$PYTHON_BIN" "$BUNDLE_DIR/random_baseline.py" verify --result-root "$RESULT_ROOT"
"$PYTHON_BIN" "$BUNDLE_DIR/export_report.py" --result-root "$RESULT_ROOT" \
  --artifact-root "$REPO_DIR/artifacts/pruning/random_baseline"
