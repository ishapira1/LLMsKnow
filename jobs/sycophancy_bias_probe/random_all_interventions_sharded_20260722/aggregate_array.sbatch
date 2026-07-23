#!/bin/bash
#SBATCH --job-name=syco_int_agg_20260722
#SBATCH --partition=shared,sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/aggregate/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/aggregate/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/common.sh

mapfile -t SELECTED_INDICES < <(selected_base_indices)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#SELECTED_INDICES[@]} )); then
  printf '[aggregate] skipping task_id=%s selected_count=%s\n' "$TASK_ID" "${#SELECTED_INDICES[@]}"
  exit 0
fi
resolve_task_context "${SELECTED_INDICES[$TASK_ID]}"
start_structured_task_log aggregate probe_bias_random_all

N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-5}"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" aggregate
  --output-root "$CELL_ROOT"
  --n-bootstrap "$N_BOOTSTRAP"
  --seed "$BOOTSTRAP_SEED"
)
print_command "${cmd[@]}"
"${cmd[@]}"

test -s "$CELL_ROOT/aggregate/summary_all_splits.csv"
test -s "$CELL_ROOT/aggregate/primary_bootstrap_all_splits.csv"
test -s "$CELL_ROOT/aggregate/causal_contrasts_all_splits.csv"
test -s "$CELL_ROOT/aggregate/probe_moderator_all_splits.csv"
test -s "$CELL_ROOT/aggregate/hidden_truth_results_all_splits.csv"
test -s "$CELL_ROOT/aggregate/manifest_all_splits.json"
printf '[aggregate] completed aggregate_dir=%s\n' "$CELL_ROOT/aggregate"
