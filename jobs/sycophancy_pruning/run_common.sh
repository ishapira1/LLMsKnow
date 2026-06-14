#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

if [[ -n "${HUGGINGFACE_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

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

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HOME" jobs/sycophancy_pruning/logs

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASETS_CSV="${DATASETS_CSV:-arc_challenge,commonsense_qa}"
RUN_NAME="${RUN_NAME:-sycophancy_pruning_qwen25_two_dataset}"
SPARSITIES_CSV="${SPARSITIES_CSV:-0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3}"
OUT_DIR="${OUT_DIR:-results/sycophancy_pruning}"
DEVICE="${DEVICE:-cuda}"
PRUNE_FAMILY="${PRUNE_FAMILY:-incorrect_suggestion}"
SPLIT_SEED="${SPLIT_SEED:-5}"
SEED="${SEED:-5}"
PRESERVE_EXCLUDE_FRACTION="${PRESERVE_EXCLUDE_FRACTION:-0.01}"
SYC_REDUCTION_TARGET="${SYC_REDUCTION_TARGET:-0.30}"
PRESERVATION_LOSS_BUDGET="${PRESERVATION_LOSS_BUDGET:-0.10}"
NEUTRAL_ACCURACY_DROP_BUDGET="${NEUTRAL_ACCURACY_DROP_BUDGET:-0.05}"
WRONG_CONTROL_MIN_EXAMPLES="${WRONG_CONTROL_MIN_EXAMPLES:-50}"

MAX_QUESTIONS_PER_DATASET="${MAX_QUESTIONS_PER_DATASET:-}"
MAX_CALIBRATION_RECORDS="${MAX_CALIBRATION_RECORDS:-}"
MAX_PRESERVATION_RECORDS="${MAX_PRESERVATION_RECORDS:-}"
MAX_EVAL_RECORDS="${MAX_EVAL_RECORDS:-}"
SAVE_ALL_SWEEP_MASKS="${SAVE_ALL_SWEEP_MASKS:-0}"
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-0}"

printf '%s\n' "[env] repo=$REPO_DIR"
printf '%s\n' "[env] python=$ENV_PYTHON"
"$ENV_PYTHON" -c "import sys, torch; print('[env] sys.executable=', sys.executable); print('[env] torch=', torch.__version__); print('[env] cuda_available=', torch.cuda.is_available())"
printf '%s\n' "[env] HF_HUB_CACHE=$HF_HUB_CACHE"
printf '%s\n' "[env] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

cmd=(
  "$ENV_PYTHON" run_sycophancy_pruning.py
  --model "$MODEL_ID"
  --datasets "$DATASETS_CSV"
  --prune_family "$PRUNE_FAMILY"
  --run_name "$RUN_NAME"
  --out_dir "$OUT_DIR"
  --sparsities "$SPARSITIES_CSV"
  --split_seed "$SPLIT_SEED"
  --seed "$SEED"
  --device "$DEVICE"
  --hf_cache_dir "$HF_CACHE_DIR"
  --preserve_exclude_fraction "$PRESERVE_EXCLUDE_FRACTION"
  --syc_reduction_target "$SYC_REDUCTION_TARGET"
  --preservation_loss_budget "$PRESERVATION_LOSS_BUDGET"
  --neutral_accuracy_drop_budget "$NEUTRAL_ACCURACY_DROP_BUDGET"
  --wrong_control_min_examples "$WRONG_CONTROL_MIN_EXAMPLES"
)

if [[ -n "$MAX_QUESTIONS_PER_DATASET" ]]; then
  cmd+=(--max_questions_per_dataset "$MAX_QUESTIONS_PER_DATASET")
fi
if [[ -n "$MAX_CALIBRATION_RECORDS" ]]; then
  cmd+=(--max_calibration_records "$MAX_CALIBRATION_RECORDS")
fi
if [[ -n "$MAX_PRESERVATION_RECORDS" ]]; then
  cmd+=(--max_preservation_records "$MAX_PRESERVATION_RECORDS")
fi
if [[ -n "$MAX_EVAL_RECORDS" ]]; then
  cmd+=(--max_eval_records "$MAX_EVAL_RECORDS")
fi
if [[ "$SAVE_ALL_SWEEP_MASKS" == "1" ]]; then
  cmd+=(--save_all_sweep_masks)
fi
if [[ "$DEVICE_MAP_AUTO" == "1" ]]; then
  cmd+=(--device_map_auto)
fi

printf '[sycophancy-pruning] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}" "$@"
