#!/usr/bin/env bash
#SBATCH --account=barak_lab
#SBATCH --partition=serial_requeue
#SBATCH --job-name=bonham_cpu_0918
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
#SBATCH --output=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/cpu/%x_%A_%a.out
#SBATCH --error=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/cpu/%x_%A_%a.err

set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${STAGE:?STAGE is required}"
task="${SLURM_ARRAY_TASK_ID:-0}"
model_key="${MODEL_KEY:-shared}"
PYTHON_BIN="$CPU_PYTHON_BIN"; export PYTHON_BIN
start_task_log "$STAGE" "$model_key" "array_task_$task"
trap 'status=$?; finish_task_log "$status"' EXIT
trap 'printf "received_usr1_requeue=1\n"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
cd "$REPO_DIR"

case "$STAGE" in
  capability_sources)
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
    command=("$PYTHON_BIN" "$BUNDLE_DIR/prepare_capability_sources.py" --output-root "$CAPABILITY_SOURCE_ROOT" --hf-cache "$HF_CACHE_DIR")
    ;;
  source_freeze)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" prepare-static --result-root "$RESULT_ROOT" --suite-source-bindings "$SUITE_SOURCE_BINDINGS" --shard-size 200)
    ;;
  prepare_screens)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" prepare-screens --result-root "$RESULT_ROOT" --shard-size 200)
    ;;
  extend_n1_screens)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" extend-n1-screens --result-root "$RESULT_ROOT" --shard-size 200)
    ;;
  prepare_model_n1_extension)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" prepare-model-n1-extension --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --shard-size 200)
    ;;
  prepare_model_n1_supplement)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" prepare-model-n1-supplement --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --shard-size 200)
    ;;
  allocate_manifests)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" allocate-manifests --result-root "$RESULT_ROOT")
    ;;
  allocate_model_manifests)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" allocate-manifests --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY")
    ;;
  build_masks)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-masks --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY")
    ;;
  build_core_masks)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-masks --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --scope core)
    ;;
  build_states)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-mask-states --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY")
    ;;
  steering_prepare)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/steering.py" prepare --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY")
    ;;
  eval_prepare)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" prepare --result-root "$RESULT_ROOT" --suite-source-bindings "$SUITE_SOURCE_BINDINGS" --external-utility-root "$CAPABILITY_SOURCE_ROOT")
    ;;
  eval_prepare_model)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" prepare --result-root "$RESULT_ROOT" --suite-source-bindings "$SUITE_SOURCE_BINDINGS" --external-utility-root "$CAPABILITY_SOURCE_ROOT" --model-key "$MODEL_KEY")
    ;;
  source_attribution_prepare)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" prepare-source-attribution --result-root "$RESULT_ROOT")
    ;;
  source_attribution_prepare_model)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" prepare-source-attribution --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY")
    ;;
  weight_model)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/weight_analysis.py" analyze-model --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY")
    ;;
  weight_aggregate)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/weight_analysis.py" aggregate --result-root "$RESULT_ROOT")
    ;;
  eval_validate)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" validate-complete --result-root "$RESULT_ROOT")
    ;;
  evalplus_prepare)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evalplus.py" prepare --result-root "$RESULT_ROOT")
    ;;
  evalplus_run)
    state_index="$((task / 8))"; within="$((task % 8))"
    models=(llama31_8b qwen25_7b gemma4_12b)
    states=(unpruned n1_mechanism n2_selective random_n1 random_n2 weak_prompt strong_prompt prompt_only_meandiff)
    benchmarks=(humaneval mbpp)
    run_model="${models[$((state_index / 8))]}"
    state_id="${states[$((state_index % 8))]}"
    benchmark="${benchmarks[$((within / 4))]}"; shard="$((within % 4))"
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evalplus.py" run-shard --result-root "$RESULT_ROOT" --model-key "$run_model" --state-id "$state_id" --benchmark "$benchmark" --shard "$shard" --image "$EVALPLUS_IMAGE" --image-sha256 "$EVALPLUS_IMAGE_SHA256" --singularity-binary /usr/bin/singularity --timeout-seconds 7200)
    ;;
  evalplus_aggregate)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evalplus.py" aggregate --result-root "$RESULT_ROOT" --image-sha256 "$EVALPLUS_IMAGE_SHA256")
    ;;
  report)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/reporting.py" --result-root "$RESULT_ROOT")
    ;;
  early_report)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/reporting.py" --result-root "$RESULT_ROOT" --early-qwen-llama)
    ;;
  final_audit)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/audit.py" --result-root "$RESULT_ROOT")
    ;;
  final_email)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/completion_email.py" --result-root "$RESULT_ROOT" --recipient itaishapira@g.harvard.edu)
    ;;
  *) printf 'unknown Bonham CPU stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
print_command "${command[@]}"
"${command[@]}"
