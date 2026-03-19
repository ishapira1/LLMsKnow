#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-smoke_aqua_mc_smollm2_135m_cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cmd=(
  "$PYTHON_BIN" run_sycophancy_bias_probe.py
  --model HuggingFaceTB/SmolLM2-135M-Instruct
  --device cpu
  --benchmark_source ays_mc_single_turn
  --input_jsonl are_you_sure.jsonl
  --dataset_name aqua_mc
  --ays_mc_datasets aqua_mc
  --mc_mode strict_mc
  --smoke_test
  --smoke_questions 24
  --n_draws 8
  --probe_layer_min 1
  --probe_layer_max 8
  --temperature 1
  --max_new_tokens 256
  --sample_batch_size 1
  --run_name "$RUN_NAME"
)

HF_CACHE_DIR="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-}}"
if [[ -n "$HF_CACHE_DIR" ]]; then
  cmd+=(--hf_cache_dir "$HF_CACHE_DIR")
fi

cmd+=("$@")

printf '[smoke] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
