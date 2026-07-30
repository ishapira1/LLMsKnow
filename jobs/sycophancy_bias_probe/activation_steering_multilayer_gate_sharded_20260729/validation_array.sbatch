#!/bin/bash
#SBATCH --job-name=syco_ml_val_20260729
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_multilayer_gate_sharded_20260729/slurm/validation/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_multilayer_gate_sharded_20260729/slurm/validation/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_multilayer_gate_sharded_20260729/common.sh
resolve_multilayer_model "${SLURM_ARRAY_TASK_ID:-0}"
start_structured_task_log validation
"$ENV_PYTHON" \
  jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/runtime_contract.py \
  --require-cuda
test -s "$MODEL_ROOT/bf16_gate/manifest.json"
"$ENV_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="complete" and d["alpha_zero_noop_exact"] and d["nonfinite_failures"]==0' \
  "$MODEL_ROOT/bf16_gate/manifest.json"

OUTPUT_DIR="$MODEL_ROOT/validation"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-multilayer
  --config "$CONFIG_PATH"
  --source-run-dir "$ARC_SOURCE"
  --question-manifest "$COHORT_MANIFEST"
  --directions-path "$CONDITIONED_DIRECTIONS"
  --output-dir "$OUTPUT_DIR"
  --split val
  --selected-layer "$SELECTED_LAYER"
  --primary-family "$PRIMARY_FAMILY"
  --layer-modes all_nonterminal,selected_single
  --position-modes boundary_only,suffix_energy_matched
  --ratios=-0.2,-0.1,-0.05,0,0.05,0.1,0.2
  --minimum-neutral-correct 100
  --maximum-live-questions 120
  --device cuda
  --hf-cache-dir "$HF_HUB_CACHE"
  --torch-dtype bfloat16
  --progress-every 10
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT_DIR/question_results.jsonl"
