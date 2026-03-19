#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
cd "$REPO_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' "[warning] No active Slurm job detected. Submit one of the seas .sbatch files instead." >&2
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

module load python/3.10.9-fasrc01

ENV_PYTHON="${ENV_PYTHON:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
  printf '%s\n' "Missing python interpreter: $ENV_PYTHON" >&2
  exit 1
fi
printf '%s\n' "[env] python=$ENV_PYTHON"
"$ENV_PYTHON" -c "import sys, numpy; print('[env] sys.executable=', sys.executable); print('[env] numpy=', numpy.__version__)"

if [[ ! -f .env ]]; then
  printf '%s\n' "Missing .env in $REPO_DIR" >&2
  exit 1
fi
set -a
source .env
set +a

HF_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-}"
if [[ -z "$HF_CACHE_DIR" ]]; then
  printf '%s\n' "HUGGINGFACE_HUB_CACHE must be set in .env" >&2
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
  "$ENV_PYTHON" run_sycophancy_bias_probe.py
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
