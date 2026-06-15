#!/bin/bash
#SBATCH --job-name=full_allfamilies_paraphrase_20260614
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_20260614/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_20260614/%x.%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
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

mkdir -p \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_HOME" \
  jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_20260614

BIAS_TYPES_CSV="${BIAS_TYPES_CSV:-incorrect_suggestion,incorrect_suggestion_strong,doubt_correct,doubt_correct_strong,suggest_correct,suggest_correct_strong,suggest_random,suggest_random_strong}"
PARAPHRASE_ARTIFACT_PATH="${PARAPHRASE_ARTIFACT_PATH:-data/ad_hoc/paraphrase_robustness_test_stems_v1}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-results/sycophancy_bias_probe}"
SPLIT_SEED="${SPLIT_SEED:-5}"
SEED="${SEED:-5}"
PROBE_SEED="${PROBE_SEED:-5}"
PROBE_LAYER_MIN="${PROBE_LAYER_MIN:-1}"
PROBE_LAYER_MAX="${PROBE_LAYER_MAX:-999}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [[ ! -d "$PARAPHRASE_ARTIFACT_PATH" && ! -f "$PARAPHRASE_ARTIFACT_PATH" ]]; then
  printf '%s\n' "Missing paraphrase artifact path: $PARAPHRASE_ARTIFACT_PATH" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0)
    TASK_LABEL="commonsense_qa_llama31_8b"
    MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    DATASET_NAME="commonsense_qa"
    AYS_MC_DATASETS="commonsense_qa"
    RUN_NAME="full_commonsense_qa_llama31_8b_allfamilies_paraphrase_20260614_seas"
    ;;
  1)
    TASK_LABEL="commonsense_qa_qwen25_7b"
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
    DATASET_NAME="commonsense_qa"
    AYS_MC_DATASETS="commonsense_qa"
    RUN_NAME="full_commonsense_qa_qwen25_7b_allfamilies_paraphrase_20260614_seas"
    ;;
  2)
    TASK_LABEL="arc_challenge_llama31_8b"
    MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    DATASET_NAME="arc_challenge"
    AYS_MC_DATASETS="arc_challenge"
    RUN_NAME="full_arc_challenge_llama31_8b_allfamilies_paraphrase_20260614_seas"
    ;;
  3)
    TASK_LABEL="arc_challenge_qwen25_7b"
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
    DATASET_NAME="arc_challenge"
    AYS_MC_DATASETS="arc_challenge"
    RUN_NAME="full_arc_challenge_qwen25_7b_allfamilies_paraphrase_20260614_seas"
    ;;
  *)
    printf '%s\n' "Unsupported SLURM_ARRAY_TASK_ID=$TASK_ID" >&2
    exit 1
    ;;
esac

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
  --bias_types "$BIAS_TYPES_CSV"
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --split_seed "$SPLIT_SEED"
  --seed "$SEED"
  --probe_seed "$PROBE_SEED"
  --probe_layer_min "$PROBE_LAYER_MIN"
  --probe_layer_max "$PROBE_LAYER_MAX"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --paraphrase_artifact_path "$PARAPHRASE_ARTIFACT_PATH"
  --run_name "$RUN_NAME"
  --fresh_run
  --out_dir "$OUT_DIR"
)

printf '[full-allfamilies-paraphrase-20260614] task=%s model=%s dataset=%s\n' \
  "$TASK_LABEL" "$MODEL_ID" "$DATASET_NAME"
printf '[full-allfamilies-paraphrase-20260614] bias_types=%s\n' "$BIAS_TYPES_CSV"
printf '[full-allfamilies-paraphrase-20260614] paraphrase_artifact_path=%s\n' "$PARAPHRASE_ARTIFACT_PATH"
printf '[full-allfamilies-paraphrase-20260614] %q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
