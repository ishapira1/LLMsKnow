#!/usr/bin/env bash
#SBATCH --job-name=syco_conf_neu_0828
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=18:00:00
#SBATCH --array=0-3
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/hf_neutral/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/hf_neutral/%x.%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
set -Eeuo pipefail
source "${CONFIDENCE_RESISTANCE_BUNDLE_DIR:?required}/common.sh"
require_runner
require_h200
models=(llama llama qwen qwen)
datasets=(arc_challenge commonsense_qa arc_challenge commonsense_qa)
task_id="${SLURM_ARRAY_TASK_ID:?required}"
model_key="${models[$task_id]}"
dataset="${datasets[$task_id]}"
start_task_log hf_neutral "${dataset}_${model_key}" canonical_token_neutral "$model_key" "$dataset"
trap 'status=$?; finish_task_log "$status"; exit "$status"' EXIT
command=("$PYTHON_BIN" "$RUNNER" score-hf-neutral --output-root "$RESULT_ROOT" --model-key "$model_key" --dataset "$dataset" --hf-cache-dir "$HF_CACHE_DIR")
print_command "${command[@]}"
"${command[@]}"
