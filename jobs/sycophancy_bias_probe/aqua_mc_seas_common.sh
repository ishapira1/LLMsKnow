#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' "[warning] No active Slurm job detected. Submit one of the seas .sbatch files instead." >&2
fi

if type module >/dev/null 2>&1; then
  module load python/3.10.9-fasrc01
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

DEFAULT_ENV_PYTHON="/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python"
if [[ -n "${ENV_PYTHON:-}" ]]; then
  PYTHON_BIN="$ENV_PYTHON"
elif [[ -x "$DEFAULT_ENV_PYTHON" ]]; then
  PYTHON_BIN="$DEFAULT_ENV_PYTHON"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "Missing python interpreter: $PYTHON_BIN" >&2
    exit 1
  fi
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '%s\n' "Missing python interpreter on PATH: $PYTHON_BIN" >&2
    exit 1
  fi
fi

printf '%s\n' "[env] python=$PYTHON_BIN"

HF_CACHE_DIR="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${TRANSFORMERS_CACHE:-}}}"
if [[ -z "$HF_CACHE_DIR" && -n "${HF_HOME:-}" ]]; then
  HF_CACHE_DIR="${HF_HOME%/}/hub"
fi
if [[ -z "$HF_CACHE_DIR" ]]; then
  printf '%s\n' "Missing HF cache dir. Set HUGGINGFACE_HUB_CACHE, HF_HUB_CACHE, TRANSFORMERS_CACHE, or HF_HOME in .env." >&2
  exit 1
fi
if [[ "$HF_CACHE_DIR" == /home/* ]]; then
  printf '%s\n' "Refusing to run: HF cache points to /home ($HF_CACHE_DIR)" >&2
  exit 1
fi

HF_DATASETS_DIR="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
HF_HOME_DIR="${HF_HOME:-$(dirname "$HF_CACHE_DIR")/hf_home}"

export HF_HUB_CACHE="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS_DIR"
export HF_HOME="$HF_HOME_DIR"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HOME"

MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-results/sycophancy_bias_probe}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"

cmd=(
  "$PYTHON_BIN" run_sycophancy_bias_probe.py
  --model "$MODEL_ID"
  --device "$DEVICE"
  --hf_cache_dir "$HF_CACHE_DIR"
  --benchmark_source ays_mc_single_turn
  --input_jsonl are_you_sure.jsonl
  --dataset_name aqua_mc
  --ays_mc_datasets aqua_mc
  --mc_mode strict_mc
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --out_dir "$OUT_DIR"
)

cmd+=("$@")

printf '[aqua-mc-seas] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
