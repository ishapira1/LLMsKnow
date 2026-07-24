#!/bin/bash
#SBATCH --job-name=syco_qwen_md_pilot_20260724
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/slurm/dose/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/slurm/dose/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/common.sh

mapfile -t LAYERS < <(pilot_layers)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#LAYERS[@]} )); then
  printf '[pilot-dose] invalid task_id=%s layer_count=%s\n' "$TASK_ID" "${#LAYERS[@]}" >&2
  exit 1
fi
LAYER="${LAYERS[$TASK_ID]}"

start_structured_task_log "dose_layer_$LAYER"
"$ENV_PYTHON" "$RUNTIME_CONTRACT_PATH" \
  --require-cuda \
  --hf-cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --model-config Qwen/Qwen2.5-7B-Instruct

test -d "$SOURCE_RUN_DIR"
test -s "$DIRECTIONS_PATH"
test -s "$(dirname "$DIRECTIONS_PATH")/manifest.json"
test -s "$CELL_ROOT/selected_patch_layers.json"

cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" run-layer
  --source-run-dir "$SOURCE_RUN_DIR"
  --directions-path "$DIRECTIONS_PATH"
  --output-root "$CELL_ROOT"
  --layer "$LAYER"
  --split val
  --protocol dose-tune
  "--alphas=$PILOT_ALPHAS"
  --max-questions "$PILOT_MAX_QUESTIONS"
  --random-control-seeds "$PILOT_CONTROL_SEEDS"
  --no-include-transported-probe
  --max-batch-size "$PILOT_MAX_BATCH_SIZE"
  --progress-every 10
  --device cuda
  --hf-cache-dir "$HF_HUB_CACHE"
  --torch-dtype auto
)
print_command "${cmd[@]}"
"${cmd[@]}"

SHARD_DIR="$CELL_ROOT/layers/$(printf 'layer_%03d' "$LAYER")/val"
test -s "$SHARD_DIR/item_results_dose_tune.jsonl"
test -s "$SHARD_DIR/manifest_dose_tune.json"
printf '[pilot-dose] completed layer=%s shard_dir=%s\n' "$LAYER" "$SHARD_DIR"

