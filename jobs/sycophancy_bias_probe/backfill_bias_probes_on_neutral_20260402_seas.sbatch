#!/bin/bash
#SBATCH --job-name=backfill_bias_probes_on_neutral_20260402
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
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

declare -a TASKS=(
  "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas|probe_bias_doubt_correct"
  "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas|probe_bias_incorrect_suggestion"
  "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas|probe_bias_suggest_correct"
  "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas|probe_bias_doubt_correct"
  "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas|probe_bias_incorrect_suggestion"
  "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas|probe_bias_suggest_correct"
  "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas|probe_bias_doubt_correct"
  "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas|probe_bias_incorrect_suggestion"
  "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas|probe_bias_suggest_correct"
  "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/arc_challenge/full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun|probe_bias_doubt_correct"
  "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/arc_challenge/full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun|probe_bias_incorrect_suggestion"
  "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/arc_challenge/full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun|probe_bias_suggest_correct"
)

TASK_SPEC="${TASK_SPEC:-}"
if [[ -n "$TASK_SPEC" ]]; then
  RUN_DIR="${TASK_SPEC%%|*}"
  PROBE_NAME="${TASK_SPEC##*|}"
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  if (( TASK_ID < 0 || TASK_ID >= ${#TASKS[@]} )); then
    printf '%s\n' "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID for ${#TASKS[@]} configured tasks" >&2
    exit 1
  fi
  RUN_DIR="${TASKS[$TASK_ID]%%|*}"
  PROBE_NAME="${TASKS[$TASK_ID]##*|}"
fi

TEMPLATE_TYPES_CSV="${TEMPLATE_TYPES_CSV:-neutral}"
SPLITS_CSV="${SPLITS_CSV:-train,val,test}"
FORCE_BACKFILL="${FORCE_BACKFILL:-0}"

cmd=(
  "$ENV_PYTHON" scripts/backfill_neutral_probe_scores.py
  --run_dir "$RUN_DIR"
  --probe_name "$PROBE_NAME"
  --hf_cache_dir "$HF_CACHE_DIR"
)

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

printf '[bias-probe-neutral-backfill] run_dir=%s probe_name=%s templates=%s splits=%s\n' \
  "$RUN_DIR" "$PROBE_NAME" "$TEMPLATE_TYPES_CSV" "$SPLITS_CSV"
printf '[bias-probe-neutral-backfill] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
