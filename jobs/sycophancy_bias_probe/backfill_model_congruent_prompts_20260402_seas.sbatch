#!/bin/bash
#SBATCH --job-name=backfill_model_congruent_20260402
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=08:00:00
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
SPLITS_CSV="${SPLITS_CSV:-test}"
PROBE_NAMES_CSV="${PROBE_NAMES_CSV:-probe_no_bias}"
MAX_RECORDS="${MAX_RECORDS:-}"
FORCE_BACKFILL="${FORCE_BACKFILL:-0}"

cmd=(
  "$ENV_PYTHON" scripts/backfill_model_congruent_prompts.py
  --run_dir "$RUN_DIR"
  --hf_cache_dir "$HF_CACHE_DIR"
)

IFS=',' read -r -a splits <<< "$SPLITS_CSV"
for split_name in "${splits[@]}"; do
  split_name="${split_name//[[:space:]]/}"
  if [[ -n "$split_name" ]]; then
    cmd+=(--split "$split_name")
  fi
done

IFS=',' read -r -a probe_names <<< "$PROBE_NAMES_CSV"
for probe_name in "${probe_names[@]}"; do
  probe_name="${probe_name//[[:space:]]/}"
  if [[ -n "$probe_name" ]]; then
    cmd+=(--probe_name "$probe_name")
  fi
done

if [[ -n "$MAX_RECORDS" ]]; then
  cmd+=(--max_records "$MAX_RECORDS")
fi

if [[ "$FORCE_BACKFILL" == "1" ]]; then
  cmd+=(--force)
fi

printf '[model-congruent-backfill] run_dir=%s splits=%s probes=%s max_records=%s\n' \
  "$RUN_DIR" "$SPLITS_CSV" "$PROBE_NAMES_CSV" "${MAX_RECORDS:-all}"
printf '[model-congruent-backfill] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
