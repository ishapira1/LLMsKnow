#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p jobs/sycophancy_bias_probe/logs

discover_run_dirs() {
  local include_backups="${INCLUDE_BACKUPS:-1}"
  local include_smoke="${INCLUDE_SMOKE:-0}"
  local -a run_dirs=()
  local model_path=""
  local run_dir=""
  local run_name=""

  while IFS= read -r model_path; do
    [[ -n "$model_path" ]] || continue
    run_dir="$(cd "$(dirname "$model_path")/../../.." && pwd)"
    run_name="$(basename "$run_dir")"

    if [[ "$include_backups" != "1" && "$run_dir" == *"/_backup/"* ]]; then
      continue
    fi
    if [[ "$include_smoke" != "1" && "$run_name" == smoke_* ]]; then
      continue
    fi

    run_dirs+=("$run_dir")
  done < <(
    find results/sycophancy_bias_probe \
      -path '*/probes/chosen_probe/probe_bias_incorrect_suggestion/model.pkl' \
      | sort
  )

  if (( ${#run_dirs[@]} == 0 )); then
    printf '%s\n' "No probe_bias_incorrect_suggestion runs were discovered." >&2
    exit 1
  fi

  printf '%s\n' "${run_dirs[@]}"
}

RUN_DIRS=()
while IFS= read -r run_dir; do
  RUN_DIRS+=("$run_dir")
done < <(discover_run_dirs)

printf '[submit] discovered %s incorrect-suggestion backfill tasks\n' "${#RUN_DIRS[@]}"
for idx in "${!RUN_DIRS[@]}"; do
  printf '  [%s] %s\n' "$idx" "${RUN_DIRS[$idx]}"
done

SBATCH_CMD=(
  sbatch
  "--array=0-$((${#RUN_DIRS[@]} - 1))"
  jobs/sycophancy_bias_probe/backfill_incorrect_suggestion_cross_family_20260407_seas.sbatch
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[submit] dry run: '
  printf '%q ' "${SBATCH_CMD[@]}" "$@"
  printf '\n'
  exit 0
fi

"${SBATCH_CMD[@]}" "$@"
