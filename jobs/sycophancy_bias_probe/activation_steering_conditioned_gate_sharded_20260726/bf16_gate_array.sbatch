#!/bin/bash
#SBATCH --job-name=syco_cond_bf16_20260726
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/bf16/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/bf16/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
resolve_conditioned_model "${SLURM_ARRAY_TASK_ID:-0}"
start_structured_task_log bf16_gate
"$ENV_PYTHON" \
  jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/runtime_contract.py \
  --require-cuda

CPU_DECISION="$AUDIT_OUTPUT_DIR/decision.json"
read -r PRIMARY_FAMILY LAYER < <(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); m=d["models"][sys.argv[2]]; assert d["gpu_stage_authorized"] and m["primary_family"]; print(m["primary_family"],m["nominated_layer"])' \
    "$CPU_DECISION" "$MODEL_IDENTIFIER"
)
BENCHMARK_MANIFEST="$INTERVENTION_ROOT/cohorts/${MODEL_KEY}_bf16_8.jsonl"
if [[ ! -e "$BENCHMARK_MANIFEST" ]]; then
  "$ENV_PYTHON" -c \
    'import json,sys,pathlib; rows=[json.loads(x) for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if x.strip() and json.loads(x)["split"]=="val"][:8]; pathlib.Path(sys.argv[2]).write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))' \
    "$COHORT_MANIFEST" "$BENCHMARK_MANIFEST"
fi
OUTPUT_DIR="$MODEL_ROOT/bf16_gate"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-conditioned
  --config "$CONFIG_PATH"
  --source-run-dir "$ARC_SOURCE"
  --question-manifest "$BENCHMARK_MANIFEST"
  --directions-path "$CONDITIONED_DIRECTIONS"
  --output-dir "$OUTPUT_DIR"
  --split val
  --layers "$LAYER"
  --primary-family "$PRIMARY_FAMILY"
  --direction-families "$PRIMARY_FAMILY"
  --position-modes boundary_only,suffix_energy_matched
  --ratios=-0.05,0,0.05
  --minimum-neutral-correct 0
  --maximum-live-questions 8
  --device cuda
  --hf-cache-dir "$HF_HUB_CACHE"
  --torch-dtype bfloat16
  --progress-every 2
)
print_command "${cmd[@]}"
"${cmd[@]}"
"$ENV_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="complete"; assert d["alpha_zero_noop_exact"] is True; assert d["nonfinite_failures"]==0' \
  "$OUTPUT_DIR/manifest.json"
