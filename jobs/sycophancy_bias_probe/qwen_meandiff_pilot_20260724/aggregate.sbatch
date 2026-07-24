#!/bin/bash
#SBATCH --job-name=syco_qwen_md_agg_20260724
#SBATCH --partition=shared,sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/slurm/aggregate/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/slurm/aggregate/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/common.sh
start_structured_task_log aggregate

select_cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" select-dose
  --output-root "$CELL_ROOT"
  --split val
)
print_command "${select_cmd[@]}"
set +e
"${select_cmd[@]}"
SELECT_STATUS=$?
set -e
if (( SELECT_STATUS != 0 )) && [[ ! -s "$CELL_ROOT/validation_dose_selection.csv" ]]; then
  printf '[pilot-aggregate] dose selection failed before producing diagnostics status=%s\n' "$SELECT_STATUS" >&2
  exit "$SELECT_STATUS"
fi
if (( SELECT_STATUS != 0 )); then
  printf '[pilot-aggregate] scientifically meaningful no-go: no candidate passed every prespecified gate\n'
fi

aggregate_cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" aggregate
  --output-root "$CELL_ROOT"
  --split val
  --n-bootstrap 1000
  --seed 5
)
print_command "${aggregate_cmd[@]}"
"${aggregate_cmd[@]}"

summary_cmd=(
  "$ENV_PYTHON"
  "$REPO_DIR/jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/summarize_pilot.py"
  summarize
  --output-root "$CELL_ROOT"
)
print_command "${summary_cmd[@]}"
"${summary_cmd[@]}"

test -s "$CELL_ROOT/validation_dose_selection.csv"
test -s "$CELL_ROOT/aggregate/summary_val.csv"
test -s "$CELL_ROOT/aggregate/manifest_val.json"
test -s "$CELL_ROOT/aggregate/pilot_dose_response.csv"
test -s "$CELL_ROOT/aggregate/plots/pilot_meandiff_dose_response_by_layer.png"
printf '[pilot-aggregate] completed output_root=%s selection_status=%s\n' \
  "$CELL_ROOT" "$SELECT_STATUS"

