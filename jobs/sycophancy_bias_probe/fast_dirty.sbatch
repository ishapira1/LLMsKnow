#!/bin/bash
#SBATCH --job-name=syco_fast_dirty
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=syco_fast_dirty.%j.out
#SBATCH --error=syco_fast_dirty.%j.err

set -euo pipefail

REPO_DIR="/n/home12/ishapira/LLMsKnow"
cd "$REPO_DIR"

module load python/3.10.9-fasrc01
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate itai_ml_env

set -a
source .env
set +a

: "${HUGGINGFACE_HUB_CACHE:?HUGGINGFACE_HUB_CACHE must be set in .env}"
if [[ "$HUGGINGFACE_HUB_CACHE" == /home/* ]]; then
  echo "Refusing to run: HUGGINGFACE_HUB_CACHE points to /home ($HUGGINGFACE_HUB_CACHE)"
  exit 1
fi

export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_DATASETS_CACHE="${HUGGINGFACE_HUB_CACHE}/datasets"
export HF_HOME="$(dirname "$HUGGINGFACE_HUB_CACHE")/hf_home"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HOME"

python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --device auto \
  --hf_cache_dir "$HUGGINGFACE_HUB_CACHE" \
  --smoke_test \
  --smoke_questions 24 \
  --n_draws 2 \
  --probe_layer_min 1 \
  --probe_layer_max 8 \
  --max_new_tokens 24 \
  --run_name fast_dirty_mistral7b_instruct
