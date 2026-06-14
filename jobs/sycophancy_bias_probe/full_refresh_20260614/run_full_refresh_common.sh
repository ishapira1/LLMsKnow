#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
cd "$REPO_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' "[warning] No active Slurm job detected. Submit one of the 20260614 .sbatch files instead." >&2
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

HF_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-}}"
if [[ -z "$HF_CACHE_DIR" ]]; then
  printf '%s\n' "HUGGINGFACE_HUB_CACHE or HF_HUB_CACHE must be set in .env" >&2
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

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-commonsense_qa}"
AYS_MC_DATASETS="${AYS_MC_DATASETS:-$DATASET_NAME}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-results/sycophancy_bias_probe}"
RUN_NAME="${RUN_NAME:-full_${DATASET_NAME}_refresh_20260614}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
SPLIT_SEED="${SPLIT_SEED:-5}"
SEED="${SEED:-5}"
PROBE_SEED="${PROBE_SEED:-5}"
PROBE_LAYER_MIN="${PROBE_LAYER_MIN:-1}"
PROBE_LAYER_MAX="${PROBE_LAYER_MAX:-999}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

cmd=(
  "$ENV_PYTHON" run_sycophancy_bias_probe.py
  --model "$MODEL_ID"
  --device "$DEVICE"
  --hf_cache_dir "$HF_CACHE_DIR"
  --benchmark_source ays_mc_single_turn
  --input_jsonl are_you_sure.jsonl
  --dataset_name "$DATASET_NAME"
  --ays_mc_datasets "$AYS_MC_DATASETS"
  --mc_mode strict_mc
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --split_seed "$SPLIT_SEED"
  --seed "$SEED"
  --probe_seed "$PROBE_SEED"
  --probe_layer_min "$PROBE_LAYER_MIN"
  --probe_layer_max "$PROBE_LAYER_MAX"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --run_name "$RUN_NAME"
  --fresh_run
  --out_dir "$OUT_DIR"
)

if [[ -n "$MAX_QUESTIONS" ]]; then
  cmd+=(--max_questions "$MAX_QUESTIONS")
fi

cmd+=("$@")

printf '[full-refresh-20260614] dataset=%s model=%s run_name=%s\n' \
  "$DATASET_NAME" "$MODEL_ID" "$RUN_NAME"
printf '[full-refresh-20260614] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
