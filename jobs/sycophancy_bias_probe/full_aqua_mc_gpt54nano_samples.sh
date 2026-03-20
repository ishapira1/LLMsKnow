#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-gpt-5.4-nano}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-results/sycophancy_bias_probe}"
DATASET_NAME="${DATASET_NAME:-commonsense_qa}"
AYS_MC_DATASETS="${AYS_MC_DATASETS:-$DATASET_NAME}"
MAX_QUESTIONS="${MAX_QUESTIONS:-500}"
RUN_NAME="${RUN_NAME:-full_${DATASET_NAME}_gpt54nano_auto_q${MAX_QUESTIONS}_samples}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"

HF_CACHE_DIR="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${TRANSFORMERS_CACHE:-}}}"
if [[ -z "$HF_CACHE_DIR" && -n "${HF_HOME:-}" ]]; then
  HF_CACHE_DIR="${HF_HOME%/}/hub"
fi
if [[ -n "$HF_CACHE_DIR" ]]; then
  export HF_HUB_CACHE="$HF_CACHE_DIR"
  export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
  export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
fi

if [[ -z "${OPENAI_API_KEY_FOR_PROJECT:-${OPENAI_API_KEY:-}}" ]]; then
  printf '%s\n' "Missing OPENAI_API_KEY_FOR_PROJECT (or OPENAI_API_KEY) in environment or .env" >&2
  exit 1
fi

SHOW_HELP=0
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
    --dataset_name=*|--dataset_type=*)
      DATASET_NAME="${arg#*=}"
      ;;
    --ays_mc_datasets=*)
      AYS_MC_DATASETS="${arg#*=}"
      ;;
    --max_questions=*)
      MAX_QUESTIONS="${arg#--max_questions=}"
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
    --dataset_name|--dataset_type)
      if (( i + 1 < ${#args[@]} )); then
        DATASET_NAME="${args[$((i + 1))]}"
      fi
      ;;
    --ays_mc_datasets)
      if (( i + 1 < ${#args[@]} )); then
        AYS_MC_DATASETS="${args[$((i + 1))]}"
      fi
      ;;
    --max_questions)
      if (( i + 1 < ${#args[@]} )); then
        MAX_QUESTIONS="${args[$((i + 1))]}"
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
  --dataset_name "$DATASET_NAME"
  --ays_mc_datasets "$AYS_MC_DATASETS"
  --instruction_policy answer_only
  --mc_mode strict_mc
  --sampling_only
  --test_frac 0
  --probe_val_frac 0
  --max_questions "$MAX_QUESTIONS"
  --n_draws 1
  --temperature 0.1
  --top_p 1.0
  --max_new_tokens 256
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --probe_layer_min 1
  --probe_layer_max 32
  --run_name "$RUN_NAME"
  --out_dir "$OUT_DIR"
)

if [[ -n "$HF_CACHE_DIR" ]]; then
  cmd+=(--hf_cache_dir "$HF_CACHE_DIR")
fi

cmd+=("$@")

printf '[full-gpt54nano-samples] %q ' "${cmd[@]}"
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
