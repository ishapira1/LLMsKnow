#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

ENV_PYTHON="${ENV_PYTHON:-python}"
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolLM2-135M-Instruct}"
DATASETS_CSV="${DATASETS_CSV:-arc_challenge,commonsense_qa}"
RUN_NAME="${RUN_NAME:-local_tiny_cpu_smoke}"
OUT_DIR="${OUT_DIR:-results/sycophancy_pruning_verification}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-}}}"

cmd=(
  "$ENV_PYTHON" run_sycophancy_pruning.py
  --model "$MODEL_ID"
  --datasets "$DATASETS_CSV"
  --run_name "$RUN_NAME"
  --out_dir "$OUT_DIR"
  --sparsities "${SPARSITIES_CSV:-0,1e-5}"
  --device cpu
  --torch_dtype float32
  --max_questions_per_dataset "${MAX_QUESTIONS_PER_DATASET:-2}"
  --max_calibration_records "${MAX_CALIBRATION_RECORDS:-4}"
  --max_preservation_records "${MAX_PRESERVATION_RECORDS:-8}"
  --max_eval_records "${MAX_EVAL_RECORDS:-8}"
  --wrong_control_min_examples "${WRONG_CONTROL_MIN_EXAMPLES:-50}"
)

if [[ -n "$HF_CACHE_DIR" ]]; then
  cmd+=(--hf_cache_dir "$HF_CACHE_DIR")
fi

printf '[local-tiny-cpu-smoke] %q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}" "$@"
