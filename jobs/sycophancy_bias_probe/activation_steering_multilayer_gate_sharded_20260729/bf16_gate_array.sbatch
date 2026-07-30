#!/bin/bash
#SBATCH --job-name=syco_ml_bf16_20260729
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_multilayer_gate_sharded_20260729/slurm/bf16/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_multilayer_gate_sharded_20260729/slurm/bf16/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_multilayer_gate_sharded_20260729/common.sh
resolve_multilayer_model "${SLURM_ARRAY_TASK_ID:-0}"
start_structured_task_log bf16_gate
"$ENV_PYTHON" \
  jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/runtime_contract.py \
  --require-cuda

BF16_MANIFEST="$INTERVENTION_ROOT/cohorts/${MODEL_KEY}_bf16_8.jsonl"
if [[ ! -e "$BF16_MANIFEST" ]]; then
  "$ENV_PYTHON" -c \
    'import json,sys,pathlib; src=pathlib.Path(sys.argv[1]); rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip() and json.loads(x)["split"]=="val"][:8]; out=pathlib.Path(sys.argv[2]); out.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows)); assert len(rows)==8' \
    "$COHORT_MANIFEST" "$BF16_MANIFEST"
fi
OUTPUT_DIR="$MODEL_ROOT/bf16_gate"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-multilayer
  --config "$CONFIG_PATH"
  --source-run-dir "$ARC_SOURCE"
  --question-manifest "$BF16_MANIFEST"
  --directions-path "$CONDITIONED_DIRECTIONS"
  --output-dir "$OUTPUT_DIR"
  --split val
  --selected-layer "$SELECTED_LAYER"
  --primary-family "$PRIMARY_FAMILY"
  --layer-modes all_nonterminal,selected_single
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
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="complete"; assert d["alpha_zero_noop_exact"] is True; assert d["nonfinite_failures"]==0; assert len(d["all_nonterminal_layers"])>20' \
  "$OUTPUT_DIR/manifest.json"
