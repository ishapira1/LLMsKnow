#!/bin/bash
#SBATCH --job-name=syco_cond_suffix_20260726
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/sensitivity/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/sensitivity/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
resolve_conditioned_model "${SLURM_ARRAY_TASK_ID:-0}"
start_structured_task_log suffix_sensitivity

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
read -r PRIMARY_FAMILY LAYER < <(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); rows=[r for r in d["selections"] if r["model_name"]==sys.argv[2]]; s=rows[0]["selected"]; print(s["conditioning_family"],s["layer"])' \
    "$SELECTION" "$MODEL_IDENTIFIER"
)
MAXIMUM="$("$ENV_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["test_questions_per_model"])' "$PROJECTION")"
LEARNED_MANIFEST="$MODEL_ROOT/test_learned/layer_$(printf '%03d' "$LAYER")/manifest.json"
SAFE_RATIO="$(
  "$ENV_PYTHON" -c \
    'import json,math,sys; d=json.load(open(sys.argv[1])); k=int(d["maximum_suffix_token_count"]); print(0.2/math.sqrt(k))' \
    "$LEARNED_MANIFEST"
)"
OUTPUT_DIR="$MODEL_ROOT/suffix_same_per_position"
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
  --position-modes suffix_same_per_position
  --ratios="-$SAFE_RATIO,0,$SAFE_RATIO"
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
