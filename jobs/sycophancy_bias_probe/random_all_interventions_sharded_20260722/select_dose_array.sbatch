#!/bin/bash
#SBATCH --job-name=syco_int_dsel_20260722
#SBATCH --partition=shared,sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/select_dose/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/select_dose/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/common.sh

mapfile -t SELECTED_INDICES < <(selected_base_indices)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#SELECTED_INDICES[@]} )); then
  printf '[select-dose] skipping task_id=%s selected_count=%s\n' "$TASK_ID" "${#SELECTED_INDICES[@]}"
  exit 0
fi
resolve_task_context "${SELECTED_INDICES[$TASK_ID]}"
start_structured_task_log select_dose probe_bias_random_all

NEUTRAL_COST="${MAX_NEUTRAL_ACCURACY_COST:-0.02}"
CORRECT_COST="${MAX_CORRECT_SUGGESTION_ACCURACY_COST:-0.02}"
CORRECT_PROBABILITY_COST="${MAX_CORRECT_SUGGESTION_PROBABILITY_COST:-0.02}"
GENUINE_MARGIN_COST="${MAX_GENUINE_AGREEMENT_RELATIVE_MARGIN_COST:-0.10}"
MIN_DOSE_RESPONSE_SPEARMAN="${MIN_DOSE_RESPONSE_SPEARMAN:-0.70}"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" select-dose
  --output-root "$CELL_ROOT"
  --split val
  --max-neutral-accuracy-cost "$NEUTRAL_COST"
  --max-correct-suggestion-accuracy-cost "$CORRECT_COST"
  --max-correct-suggestion-probability-cost "$CORRECT_PROBABILITY_COST"
  --max-genuine-agreement-relative-margin-cost "$GENUINE_MARGIN_COST"
  --min-dose-response-spearman "$MIN_DOSE_RESPONSE_SPEARMAN"
)
print_command "${cmd[@]}"
"${cmd[@]}"

test -s "$SELECTION_JSON"
test -s "$CELL_ROOT/validation_dose_selection.csv"
printf '[select-dose] completed selection=%s\n' "$SELECTION_JSON"
