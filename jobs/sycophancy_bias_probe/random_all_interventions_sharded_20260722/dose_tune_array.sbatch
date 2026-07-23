#!/bin/bash
#SBATCH --job-name=syco_int_dose_20260722
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --array=0-11
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/dose_tune/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/dose_tune/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/common.sh

TOP_K="${TOP_K_PATCH_LAYERS:-3}"
mapfile -t SELECTED_INDICES < <(selected_base_indices)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS=$(( ${#SELECTED_INDICES[@]} * TOP_K ))
if (( TASK_ID >= TOTAL_TASKS )); then
  printf '[dose-tune] skipping task_id=%s total_tasks=%s\n' "$TASK_ID" "$TOTAL_TASKS"
  exit 0
fi
COMBO_OFFSET=$((TASK_ID / TOP_K))
CANDIDATE_INDEX=$((TASK_ID % TOP_K))
resolve_task_context "${SELECTED_INDICES[$COMBO_OFFSET]}"
start_structured_task_log "dose_tune_candidate_$CANDIDATE_INDEX" probe_bias_random_all

LAYER_SELECTION="$CELL_ROOT/selected_patch_layers.json"
test -s "$DIRECTIONS_PATH"
test -s "$LAYER_SELECTION"
CANDIDATE_COUNT="$($ENV_PYTHON -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidate_layers"]))' "$LAYER_SELECTION")"
if (( CANDIDATE_INDEX >= CANDIDATE_COUNT )); then
  printf '[dose-tune] clean skip candidate_index=%s candidate_count=%s\n' \
    "$CANDIDATE_INDEX" "$CANDIDATE_COUNT"
  exit 0
fi
VAL_MAX_QUESTIONS="${VAL_MAX_QUESTIONS:-}"
ALPHAS="${ALPHAS:--4,-2,-1,-0.5,0,0.5,1,2,4}"
STEERING_CONTROL_SEEDS="${STEERING_CONTROL_SEEDS:-0,1,2,3,4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" run-dose-candidate
  --source-run-dir "$SOURCE_RUN_DIR"
  --directions-path "$DIRECTIONS_PATH"
  --layer-selection-json "$LAYER_SELECTION"
  --candidate-index "$CANDIDATE_INDEX"
  --output-root "$CELL_ROOT"
  "--alphas=$ALPHAS"
  --random-control-seeds "$STEERING_CONTROL_SEEDS"
  --max-batch-size "$MAX_BATCH_SIZE"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
)
if [[ -n "$VAL_MAX_QUESTIONS" ]]; then
  cmd+=(--max-questions "$VAL_MAX_QUESTIONS")
fi
print_command "${cmd[@]}"
"${cmd[@]}"

SELECTED_LAYER="$($ENV_PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["candidate_layers"][int(sys.argv[2])]))' "$LAYER_SELECTION" "$CANDIDATE_INDEX")"
SHARD_DIR="$CELL_ROOT/layers/$(printf 'layer_%03d' "$SELECTED_LAYER")/val"
test -s "$SHARD_DIR/item_results_dose_tune.jsonl"
test -s "$SHARD_DIR/manifest_dose_tune.json"
printf '[dose-tune] completed layer=%s shard_dir=%s\n' "$SELECTED_LAYER" "$SHARD_DIR"
