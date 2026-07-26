#!/bin/bash
#SBATCH --job-name=syco_cond_val_20260726
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/validation/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/validation/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
resolve_conditioned_model "${SLURM_ARRAY_TASK_ID:-0}"
start_structured_task_log validation
"$ENV_PYTHON" \
  jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/runtime_contract.py \
  --require-cuda

CPU_DECISION="$AUDIT_OUTPUT_DIR/decision.json"
PROJECTION="$INTERVENTION_ROOT/compute_projection.json"
read -r PRIMARY_FAMILY LAYERS < <(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); m=d["models"][sys.argv[2]]; assert d["gpu_stage_authorized"]; print(m["primary_family"],",".join(map(str,m["nominated_layers_with_neighbors"])))' \
    "$CPU_DECISION" "$MODEL_IDENTIFIER"
)
read -r DIRECTION_FAMILIES MAXIMUM < <(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["gpu_submission_authorized"]; primary=sys.argv[2]; f=[primary,"global_wc"]+(["global_wn"] if d["include_global_wn"] else []); print(",".join(f),d["validation_questions_per_model"])' \
    "$PROJECTION" "$PRIMARY_FAMILY"
)
OUTPUT_DIR="$MODEL_ROOT/validation"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-conditioned
  --config "$CONFIG_PATH"
  --source-run-dir "$ARC_SOURCE"
  --question-manifest "$COHORT_MANIFEST"
  --directions-path "$CONDITIONED_DIRECTIONS"
  --output-dir "$OUTPUT_DIR"
  --split val
  --layers "$LAYERS"
  --primary-family "$PRIMARY_FAMILY"
  --direction-families "$DIRECTION_FAMILIES"
  --position-modes boundary_only,suffix_energy_matched
  --ratios=-0.2,-0.1,-0.05,0,0.05,0.1,0.2
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
