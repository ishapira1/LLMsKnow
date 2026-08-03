#!/usr/bin/env bash
#SBATCH --job-name=rb_gpu_20260803
#SBATCH --partition=gpu_h200,seas_gpu,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=18:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -Eeuo pipefail
if [[ -n "${RANDOM_BASELINE_BUNDLE_DIR:-}" ]]; then
  source "$RANDOM_BASELINE_BUNDLE_DIR/common.sh"
elif [[ -n "${REPO_DIR:-}" ]]; then
  source "$REPO_DIR/jobs/sycophancy_pruning/random_baseline/common.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

STAGE="${STAGE:?Set STAGE}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="llama"
LABEL="$STAGE"

on_exit() {
  local status="$?"
  finish_task_log "$status"
  if [[ "$status" -ne 0 ]]; then failure_email "$status" "$STAGE" "$MODEL"; fi
  exit "$status"
}
trap on_exit EXIT

case "$STAGE" in
  build_masks|parity|smoke)
    if [[ "$TASK_ID" -eq 1 ]]; then MODEL="qwen"; fi
    ;;
  core)
    if [[ "$TASK_ID" -ge 11 ]]; then MODEL="qwen"; fi
    ;;
  broad)
    if [[ "$TASK_ID" -ge 72 ]]; then MODEL="qwen"; fi
    ;;
  *) printf 'Unknown GPU stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac

start_task_log "$STAGE" "$MODEL" "$LABEL"
snapshot="$(model_snapshot "$MODEL")"
registry="$RESULT_ROOT/registry/$MODEL.json"

case "$STAGE" in
  build_masks)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/random_baseline.py" build-masks
      --model "$MODEL" --result-root "$RESULT_ROOT" --model-snapshot "$snapshot")
    ;;
  parity)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/multi_state_eval.py" parity
      --model "$MODEL" --result-root "$RESULT_ROOT")
    ;;
  smoke)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/multi_state_eval.py" smoke
      --model "$MODEL" --model-snapshot "$snapshot" --registry "$registry"
      --result-root "$RESULT_ROOT")
    ;;
  core)
    local_task="$((TASK_ID % 11))"
    if [[ "$local_task" -eq 0 ]]; then
      states="base,learned"
    else
      start="$(((local_task - 1) * 4))"
      states="$($PYTHON_BIN -c 'import random_baseline as r,sys; s=[f"{f}__seed_{x}" for f in r.CONTROL_FAMILIES for x in r.SEEDS]; print(",".join(s[int(sys.argv[1]):int(sys.argv[1])+4]))' "$start")"
    fi
    command=("$PYTHON_BIN" "$BUNDLE_DIR/multi_state_eval.py" run --model "$MODEL"
      --model-snapshot "$snapshot" --registry "$registry" --result-root "$RESULT_ROOT"
      --states "$states" --benchmark core)
    ;;
  broad)
    local_task="$((TASK_ID % 72))"
    benchmark_index="$((local_task / 12))"
    state_index="$((local_task % 12))"
    benchmarks=(sycobench alpaca_wikitext mmlu icl feedback elephant)
    benchmark="${benchmarks[$benchmark_index]}"
    state="$($PYTHON_BIN -c 'import random_baseline as r,sys; s=["base","learned"]+[f"{f}__seed_{x}" for f in r.CONTROL_FAMILIES for x in r.BROAD_SEEDS]; print(s[int(sys.argv[1])])' "$state_index")"
    command=("$PYTHON_BIN" "$BUNDLE_DIR/multi_state_eval.py" run --model "$MODEL"
      --model-snapshot "$snapshot" --registry "$registry" --result-root "$RESULT_ROOT"
      --states "$state" --benchmark "$benchmark" --sycobench-source "$SYCOBENCH_SOURCE")
    ;;
esac
print_command "${command[@]}"
"${command[@]}"
