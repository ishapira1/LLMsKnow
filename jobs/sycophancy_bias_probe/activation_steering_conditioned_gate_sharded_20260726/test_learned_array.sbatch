#!/bin/bash
#SBATCH --job-name=syco_cond_test_20260726
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/test/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/test/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_INDEX=$((TASK_ID / 2))
LAYER_INDEX=$((TASK_ID % 2))
resolve_conditioned_model "$MODEL_INDEX"
start_structured_task_log test_learned

SELECTION="$INTERVENTION_ROOT/validation_selection.json"
PROJECTION="$INTERVENTION_ROOT/compute_projection.json"
SELECTED="$(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); rows=[r for r in d["selections"] if r["model_name"]==sys.argv[2]]; print(int(len(rows)==1 and bool(rows[0]["selected"])))' \
    "$SELECTION" "$MODEL_IDENTIFIER"
)"
if [[ "$SELECTED" != "1" ]]; then
  printf '[task] skipped: model has no eligible validation candidate\n'
  exit 0
fi
"$ENV_PYTHON" \
  jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/runtime_contract.py \
  --require-cuda
read -r PRIMARY_FAMILY LAYER MAGNITUDE < <(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); rows=[r for r in d["selections"] if r["model_name"]==sys.argv[2]]; assert len(rows)==1 and rows[0]["selected"]; s=rows[0]["selected"]; i=int(sys.argv[3]); print(s["conditioning_family"],s["layer"] if i==0 else s["neighbor_layer"],s["ratio_magnitude"])' \
    "$SELECTION" "$MODEL_IDENTIFIER" "$LAYER_INDEX"
)
read -r MAXIMUM NEIGHBOR_FULL < <(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["gpu_submission_authorized"]; print(d["test_questions_per_model"],int(d["neighbor_full_curve"]))' \
    "$PROJECTION"
)
RATIOS="-0.2,-0.1,-0.05,0,0.05,0.1,0.2"
if (( LAYER_INDEX == 1 && NEIGHBOR_FULL == 0 )); then
  RATIOS="-$MAGNITUDE,0,$MAGNITUDE"
fi
OUTPUT_DIR="$MODEL_ROOT/test_learned/layer_$(printf '%03d' "$LAYER")"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-conditioned
  --config "$CONFIG_PATH"
  --source-run-dir "$ARC_SOURCE"
  --question-manifest "$COHORT_MANIFEST"
  --directions-path "$CONDITIONED_DIRECTIONS"
  --output-dir "$OUTPUT_DIR"
  --split test
  --layers "$LAYER"
  --primary-family "$PRIMARY_FAMILY"
  --direction-families "$PRIMARY_FAMILY"
  --position-modes boundary_only,suffix_energy_matched
  --ratios="$RATIOS"
  --minimum-neutral-correct 100
  --maximum-live-questions "$MAXIMUM"
  --device cuda
  --hf-cache-dir "$HF_HUB_CACHE"
  --torch-dtype bfloat16
  --progress-every 10
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT_DIR/question_results.jsonl"
