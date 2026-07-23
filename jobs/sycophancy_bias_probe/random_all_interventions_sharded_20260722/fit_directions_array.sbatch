#!/bin/bash
#SBATCH --job-name=syco_int_fit_20260722
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/fit/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/fit/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/common.sh

mapfile -t SELECTED_INDICES < <(selected_base_indices)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#SELECTED_INDICES[@]} )); then
  printf '[fit] skipping task_id=%s selected_count=%s\n' "$TASK_ID" "${#SELECTED_INDICES[@]}"
  exit 0
fi
resolve_task_context "${SELECTED_INDICES[$TASK_ID]}"
start_structured_task_log fit_directions probe_bias_random_all

FIT_MAX_QUESTIONS="${FIT_MAX_QUESTIONS:-}"
FIT_SEED="${FIT_SEED:-5}"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" fit-directions
  --source-run-dir "$SOURCE_RUN_DIR"
  --output-dir "$DIRECTIONS_DIR"
  --fit-split train
  --seed "$FIT_SEED"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
  --n-control-directions 20
)
if [[ -n "$FIT_MAX_QUESTIONS" ]]; then
  cmd+=(--max-questions "$FIT_MAX_QUESTIONS")
fi
print_command "${cmd[@]}"
"${cmd[@]}"

test -s "$DIRECTIONS_PATH"
test -s "$DIRECTIONS_DIR/manifest.json"
printf '[fit] completed directions=%s manifest=%s\n' \
  "$DIRECTIONS_PATH" "$DIRECTIONS_DIR/manifest.json"
