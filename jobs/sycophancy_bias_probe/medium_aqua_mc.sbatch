#!/bin/bash
#SBATCH --job-name=medium_aqua_mc_mistral7b_instruct_v0_2_v9
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="/n/home12/ishapira/LLMsKnow"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

module load python/3.10.9-fasrc01
ENV_PYTHON="/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python"
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "Missing python interpreter: $ENV_PYTHON"
  exit 1
fi
echo "[env] python=$ENV_PYTHON"
"$ENV_PYTHON" -c "import sys, numpy; print('[env] sys.executable=', sys.executable); print('[env] numpy=', numpy.__version__)"

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

"$ENV_PYTHON" run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --device auto \
  --hf_cache_dir "$HUGGINGFACE_HUB_CACHE" \
  --benchmark_source ays_mc_single_turn \
  --input_jsonl are_you_sure.jsonl \
  --dataset_name aqua_mc \
  --ays_mc_datasets aqua_mc \
  --mc_mode strict_mc \
  --temperature 0.1 \
  --max_questions 300 \
  --n_draws 16 \
  --probe_layer_min 1 \
  --probe_layer_max 16 \
  --max_new_tokens 256 \
  --run_name medium_aqua_mc_mistral7b_instruct_V9
