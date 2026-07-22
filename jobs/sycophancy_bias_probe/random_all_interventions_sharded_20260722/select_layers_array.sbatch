#!/bin/bash
#SBATCH --job-name=syco_int_lsel_20260722
#SBATCH --partition=shared,sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/select_layers/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/select_layers/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/common.sh

mapfile -t SELECTED_INDICES < <(selected_base_indices)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#SELECTED_INDICES[@]} )); then
  printf '[select-layers] skipping task_id=%s selected_count=%s\n' "$TASK_ID" "${#SELECTED_INDICES[@]}"
  exit 0
fi
resolve_task_context "${SELECTED_INDICES[$TASK_ID]}"
start_structured_task_log select_layers probe_bias_random_all

TOP_K="${TOP_K_PATCH_LAYERS:-3}"
cmd=(
  "$ENV_PYTHON" run_random_all_intervention.py select-layers
  --output-root "$CELL_ROOT"
  --split val
  --top-k "$TOP_K"
)
print_command "${cmd[@]}"
"${cmd[@]}"

test -s "$CELL_ROOT/selected_patch_layers.json"
test -s "$CELL_ROOT/validation_layer_selection.csv"
printf '[select-layers] completed selection=%s\n' "$CELL_ROOT/selected_patch_layers.json"
