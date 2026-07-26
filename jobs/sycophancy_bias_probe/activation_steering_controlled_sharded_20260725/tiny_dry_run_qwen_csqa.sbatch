#!/bin/bash
#SBATCH --job-name=syco_ctrl_tiny_20260725
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/tiny/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/tiny/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
QUESTION_MANIFEST="${QUESTION_MANIFEST:-configs/experiments/activation_steering_preflight_8_20260725.jsonl}"
export QUESTION_MANIFEST
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_cell 1
start_structured_task_log tiny_dry_run
validate_runtime_contract --require-cuda

LAYERS="${LAYERS:-17,18}"
CONTROL_SEEDS="${CONTROL_SEEDS:-0,1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
TINY_ROOT="$INTERVENTION_ROOT/tiny/qwen25_7b/commonsense_qa"
TINY_DIRECTIONS="$TINY_ROOT/directions"
REAL_MODEL_GATE="$TINY_ROOT/real_model_bf16_gate.json"
mkdir -p "$TINY_ROOT"

export ACTIVATION_STEERING_REAL_SOURCE_RUN="$SOURCE_RUN_DIR"
export ACTIVATION_STEERING_REAL_MANIFEST="$PREFLIGHT_MANIFEST"
export ACTIVATION_STEERING_REAL_CONFIG="$CONFIG_PATH"
export ACTIVATION_STEERING_REAL_LAYER="${ACTIVATION_STEERING_REAL_LAYER:-17}"
export ACTIVATION_STEERING_REAL_REPORT="$REAL_MODEL_GATE"
real_gate_cmd=(
  "$ENV_PYTHON" -m unittest -v
  tests.test_controlled_activation_steering_real_model
)
print_command "${real_gate_cmd[@]}"
"${real_gate_cmd[@]}"
test -s "$REAL_MODEL_GATE"

fit_cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" fit-directions
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --question-manifest "$PREFLIGHT_MANIFEST"
  --output-dir "$TINY_DIRECTIONS"
  --layers "$LAYERS"
  --control-seeds "$CONTROL_SEEDS"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
  --progress-every 1
)
print_command "${fit_cmd[@]}"
"${fit_cmd[@]}"

ALPHAS="$(
  "$ENV_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); d=p["diagnostics"]; ratio_one=max(x["wn_raw_norm"]/x["median_residual_norm"] for x in d); ratio=128*ratio_one; grid=[-128,-64,-32,-16,-8,-4,-2,-1,-.5,-.25,0,.25,.5,1,2,4,8,16,32,64,128]; end=128;
while ratio < 4 and end < 8192: end*=2; grid=[-end]+grid+[end]; ratio*=2
small=.25
while ratio_one >= 1 and ratio_one*small > .25 and small > 2**-12: small/=2; grid.extend([-small,small])
print(",".join(str(x) for x in sorted(set(grid))))' \
  "$TINY_DIRECTIONS/manifest.json"
)"
printf '[tiny] calibrated_alphas=%s\n' "$ALPHAS"

for split in train val test; do
  run_cmd=(
    "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" tiny-dry-run
    --config "$CONFIG_PATH"
    --source-run-dir "$SOURCE_RUN_DIR"
    --question-manifest "$PREFLIGHT_MANIFEST"
    --directions-path "$TINY_DIRECTIONS/directions.npz"
    --output-dir "$TINY_ROOT/$split"
    --split "$split"
    --layers "$LAYERS"
    "--alphas=$ALPHAS"
    --control-seeds "$CONTROL_SEEDS"
    --learned-directions wn,cn,wc,sw
    --max-batch-size "$MAX_BATCH_SIZE"
    --score-fixed-probe
    --generation-diagnostics
    --device cuda
    --hf-cache-dir "$HF_CACHE_DIR"
    --torch-dtype auto
    --progress-every 1
  )
  print_command "${run_cmd[@]}"
  "${run_cmd[@]}"
done

test -s "$TINY_ROOT/test/question_results.jsonl"
test -s "$TINY_ROOT/test/noop_sentinels.jsonl"
projection_cmd=(
  "$ENV_PYTHON" "$REPO_DIR/scripts/summarize_activation_steering_tiny.py"
  --manifest "$TINY_ROOT/train/manifest.json"
  --manifest "$TINY_ROOT/val/manifest.json"
  --manifest "$TINY_ROOT/test/manifest.json"
  --real-model-gate "$REAL_MODEL_GATE"
  --output "$TINY_ROOT/compute_projection.json"
)
print_command "${projection_cmd[@]}"
"${projection_cmd[@]}"
test -s "$TINY_ROOT/compute_projection.json"
