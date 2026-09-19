#!/usr/bin/env bash
#SBATCH --account=barak_lab
#SBATCH --job-name=bonham_gpu_0918
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
#SBATCH --output=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu/%x_%A_%a.out
#SBATCH --error=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu/%x_%A_%a.err

set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${STAGE:?STAGE is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
task="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON_BIN="$(python_for_model "$MODEL_KEY")"; export PYTHON_BIN
start_task_log "$STAGE" "$MODEL_KEY" "array_task_$task"
trap 'status=$?; finish_task_log "$status"' EXIT
trap 'printf "received_usr1_requeue=1\n"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
cd "$REPO_DIR"

case "$STAGE" in
  model_smoke)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" model-smoke --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR")
    ;;
  neutral_screen|n1_screen|source_screen)
    shard="$task"
    if [[ "$STAGE" == neutral_screen ]]; then
      input_dir="$RESULT_ROOT/inputs/neutral_screen_shards"
    elif [[ "$STAGE" == n1_screen ]]; then
      input_dir="$RESULT_ROOT/inputs/n1_screen_shards"
    else
      input_dir="$RESULT_ROOT/inputs/source_screen_shards/$MODEL_KEY"
    fi
    task_path="$input_dir/shard_$(printf '%04d' "$shard").jsonl"
    if [[ ! -f "$task_path" ]]; then
      printf 'skip_reason=shard_not_materialized\ntask_path=%s\n' "$task_path"
      exit 0
    fi
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" run-screen --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR" --stage "$STAGE" --task-shard "$task_path" --shard-index "$shard" --output "$RESULT_ROOT/$STAGE/$MODEL_KEY/shard_$(printf '%04d' "$shard")" --batch-size 4)
    ;;
  score_component)
    score_ids=(n1_seed5_prune n1_seed17_prune n1_seed29_prune general_preserve selective_preserve source_all_prune source_false_prune)
    score_id="${score_ids[$task]}"
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" score-component --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --score-id "$score_id" --hf-cache "$HF_CACHE_DIR")
    ;;
  random_masks)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-random-mask --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR")
    ;;
  steering_extract)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/steering.py" extract --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR")
    ;;
  steering_develop)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/steering.py" develop --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR")
    ;;
  eval_generalization|eval_useful_assertions|eval_source_attribution|eval_capabilities)
    states=(unpruned n1_mechanism n2_selective random_n1 random_n2 weak_prompt strong_prompt prompt_only_meandiff)
    if [[ "$STAGE" == eval_generalization ]]; then
      family=generalization; stride=60
    elif [[ "$STAGE" == eval_useful_assertions ]]; then
      family=useful_assertions; stride=120
    elif [[ "$STAGE" == eval_source_attribution ]]; then
      family=source_attribution; stride=60
    else
      family=capabilities; stride=80
    fi
    state_index="$((task / stride))"; shard="$((task % stride))"
    state_id="${states[$state_index]}"
    task_path="$RESULT_ROOT/evaluations/inputs/$MODEL_KEY/$family/shard_$(printf '%04d' "$shard").jsonl"
    if [[ ! -f "$task_path" ]]; then
      printf 'skip_reason=shard_not_materialized\ntask_path=%s\n' "$task_path"
      exit 0
    fi
    command=("$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" run-shard --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --state-id "$state_id" --family "$family" --shard "$shard" --hf-cache "$HF_CACHE_DIR" --batch-size 4)
    ;;
  *) printf 'unknown Bonham GPU stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
print_command "${command[@]}"
"${command[@]}"
