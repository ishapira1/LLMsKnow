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
DEVICE="${DEVICE:-auto}"
RUN_NAME="${RUN_NAME:-smoke_aqua_mc_mistral7b_auto_q12_l4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
OUT_DIR="results/sycophancy_bias_probe"
SHOW_HELP=0
HF_CACHE_DIR="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${TRANSFORMERS_CACHE:-}}}"
if [[ -z "$HF_CACHE_DIR" && -n "${HF_HOME:-}" ]]; then
  HF_CACHE_DIR="${HF_HOME%/}/hub"
fi
if [[ -n "$HF_CACHE_DIR" ]]; then
  export HF_HUB_CACHE="$HF_CACHE_DIR"
  export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
  export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
fi

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      SHOW_HELP=1
      ;;
    --model=*)
      MODEL_ID="${arg#--model=}"
      ;;
    --out_dir=*)
      OUT_DIR="${arg#--out_dir=}"
      ;;
    --device=*)
      DEVICE="${arg#--device=}"
      ;;
    --run_name=*)
      RUN_NAME="${arg#--run_name=}"
      ;;
  esac
done

args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --model)
      if (( i + 1 < ${#args[@]} )); then
        MODEL_ID="${args[$((i + 1))]}"
      fi
      ;;
    --out_dir)
      if (( i + 1 < ${#args[@]} )); then
        OUT_DIR="${args[$((i + 1))]}"
      fi
      ;;
    --device)
      if (( i + 1 < ${#args[@]} )); then
        DEVICE="${args[$((i + 1))]}"
      fi
      ;;
    --run_name)
      if (( i + 1 < ${#args[@]} )); then
        RUN_NAME="${args[$((i + 1))]}"
      fi
      ;;
  esac
done

cmd=(
  "$PYTHON_BIN" run_sycophancy_bias_probe.py
  --model "$MODEL_ID"
  --device "$DEVICE"
  --benchmark_source ays_mc_single_turn
  --input_jsonl are_you_sure.jsonl
  --dataset_name aqua_mc
  --ays_mc_datasets aqua_mc
  --mc_mode strict_mc
  --smoke_test
  --smoke_questions 12
  --override_sampling_cache
  --n_draws 4
  --probe_layer_min 1
  --probe_layer_max 4
  --temperature 1
  --max_new_tokens 256
  --sample_batch_size 1
  --run_name "$RUN_NAME"
  --out_dir "$OUT_DIR"
)

if [[ -n "$HF_CACHE_DIR" ]]; then
  cmd+=(--hf_cache_dir "$HF_CACHE_DIR")
fi

cmd+=("$@")

printf '[smoke] %q ' "${cmd[@]}"
printf '\n'

if (( SHOW_HELP )); then
  exec "${cmd[@]}"
fi

"${cmd[@]}"

integrity_cmd=(
  "$PYTHON_BIN" -m llmssycoph.integrity
  --model "$MODEL_ID"
  --out_dir "$OUT_DIR"
  --run_name "$RUN_NAME"
)

printf '[integrity] %q ' "${integrity_cmd[@]}"
printf '\n'

exec "${integrity_cmd[@]}"
