#!/bin/bash
#SBATCH --job-name=probe_displacement_mini_qwen25_7b_20260602
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

module load python/3.10.9-fasrc01

ENV_PYTHON="${ENV_PYTHON:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
  printf '%s\n' "Missing python interpreter: $ENV_PYTHON" >&2
  exit 1
fi

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

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HOME" jobs/sycophancy_bias_probe/logs

RUN_DIR="${RUN_DIR:-results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/arc_challenge/full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun}"
PROBE_NAME="${PROBE_NAME:-probe_no_bias}"
SPLIT="${SPLIT:-test}"
MAX_QUESTIONS="${MAX_QUESTIONS:-64}"
DEVICE="${DEVICE:-auto}"
USE_DEVICE_MAP_AUTO="${USE_DEVICE_MAP_AUTO:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/analysis/probe_displacement_decomposition_mini_q${MAX_QUESTIONS}}"

cmd=(
  "$ENV_PYTHON" scripts/analyze_probe_displacement.py
  --run_dir "$RUN_DIR"
  --output_dir "$OUTPUT_DIR"
  --split "$SPLIT"
  --probe_name "$PROBE_NAME"
  --max_questions "$MAX_QUESTIONS"
  --device "$DEVICE"
  --hf_cache_dir "$HF_CACHE_DIR"
)

if [[ "$USE_DEVICE_MAP_AUTO" == "1" ]]; then
  cmd+=(--device_map_auto)
fi

printf '[probe-displacement-mini] run_dir=%s probe=%s split=%s max_questions=%s output_dir=%s\n' \
  "$RUN_DIR" "$PROBE_NAME" "$SPLIT" "$MAX_QUESTIONS" "$OUTPUT_DIR"
printf '[probe-displacement-mini] %q ' "${cmd[@]}" "$@"
printf '\n'

exec "${cmd[@]}" "$@"
