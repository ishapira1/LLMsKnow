#!/bin/bash
#SBATCH --job-name=backfill_incorrect_suggestion_cross_family_20260407
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/%x.%A_%a.err
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

discover_run_dirs() {
  local include_backups="${INCLUDE_BACKUPS:-1}"
  local include_smoke="${INCLUDE_SMOKE:-0}"
  local -a run_dirs=()
  local model_path=""
  local run_dir=""
  local run_name=""

  while IFS= read -r model_path; do
    [[ -n "$model_path" ]] || continue
    run_dir="$(cd "$(dirname "$model_path")/../../.." && pwd)"
    run_name="$(basename "$run_dir")"

    if [[ "$include_backups" != "1" && "$run_dir" == *"/_backup/"* ]]; then
      continue
    fi
    if [[ "$include_smoke" != "1" && "$run_name" == smoke_* ]]; then
      continue
    fi

    run_dirs+=("$run_dir")
  done < <(
    find results/sycophancy_bias_probe \
      -path '*/probes/chosen_probe/probe_bias_incorrect_suggestion/model.pkl' \
      | sort
  )

  if (( ${#run_dirs[@]} == 0 )); then
    printf '%s\n' "No probe_bias_incorrect_suggestion runs were discovered." >&2
    exit 1
  fi

  printf '%s\n' "${run_dirs[@]}"
}

RUN_DIRS=()
while IFS= read -r run_dir; do
  RUN_DIRS+=("$run_dir")
done < <(discover_run_dirs)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_DIRS[@]} )); then
  printf '%s\n' "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID for ${#RUN_DIRS[@]} discovered tasks" >&2
  exit 1
fi

RUN_DIR="${RUN_DIRS[$TASK_ID]}"
PROBE_NAME="probe_bias_incorrect_suggestion"
TEMPLATE_TYPES_CSV="${TEMPLATE_TYPES_CSV:-neutral,doubt_correct,suggest_correct,model_congruent_suggestion}"
SPLITS_CSV="${SPLITS_CSV:-train,val,test}"
FORCE_BACKFILL="${FORCE_BACKFILL:-1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"

cmd=(
  "$ENV_PYTHON" scripts/backfill_neutral_probe_scores.py
  --run_dir "$RUN_DIR"
  --probe_name "$PROBE_NAME"
  --hf_cache_dir "$HF_CACHE_DIR"
)

if [[ -n "$OUTPUT_SUBDIR" ]]; then
  cmd+=(--output_subdir "$OUTPUT_SUBDIR")
fi

IFS=',' read -r -a template_types <<< "$TEMPLATE_TYPES_CSV"
for template_type in "${template_types[@]}"; do
  template_type="${template_type//[[:space:]]/}"
  if [[ -n "$template_type" ]]; then
    cmd+=(--template_type "$template_type")
  fi
done

IFS=',' read -r -a splits <<< "$SPLITS_CSV"
for split_name in "${splits[@]}"; do
  split_name="${split_name//[[:space:]]/}"
  if [[ -n "$split_name" ]]; then
    cmd+=(--split "$split_name")
  fi
done

if [[ "$FORCE_BACKFILL" == "1" ]]; then
  cmd+=(--force)
fi

printf '[incorrect-suggestion-cross-family-backfill] task=%s/%s run_dir=%s templates=%s splits=%s force=%s\n' \
  "$((TASK_ID + 1))" "${#RUN_DIRS[@]}" "$RUN_DIR" "$TEMPLATE_TYPES_CSV" "$SPLITS_CSV" "$FORCE_BACKFILL"
printf '[incorrect-suggestion-cross-family-backfill] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
