#!/bin/bash
#SBATCH --job-name=syco_prune_sample_20260722
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=18:00:00
#SBATCH --array=0-11
#SBATCH --output=jobs/sycophancy_pruning/logs/paper_global_sharded_20260722/slurm/sampling/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_pruning/logs/paper_global_sharded_20260722/slurm/sampling/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
setup_environment

SEEDS=(5 17 29)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= 12 )); then
  printf '%s\n' "sampling task id out of range: $TASK_ID" >&2
  exit 2
fi
MODEL_INDEX=$((TASK_ID / 6))
DATASET_INDEX=$(((TASK_ID / 3) % 2))
SEED_INDEX=$((TASK_ID % 3))
MODEL_ID="$(model_id_for_index "$MODEL_INDEX")"
MODEL_REVISION="$(model_revision_for_index "$MODEL_INDEX")"
DATASET="$(dataset_for_index "$DATASET_INDEX")"
CALIBRATION_SEED="${SEEDS[$SEED_INDEX]}"
require_sha "MODEL_REVISION" "$MODEL_REVISION"

MODEL_SLUG="$(slugify "$MODEL_ID")"
RUN_NAME="$(sampling_run_name "$MODEL_SLUG" "$DATASET" "$CALIBRATION_SEED")"
RUN_DIR="$(sampling_run_dir "$MODEL_ID" "$DATASET" "$CALIBRATION_SEED")"
ALLOW_STALE_LOCK_CLEANUP="${ALLOW_STALE_LOCK_CLEANUP:-0}"
if [[ "$ALLOW_STALE_LOCK_CLEANUP" == "1" && -f "$RUN_DIR/.run.lock" ]]; then
  printf '[sampling] removing stale lock by explicit request: %s\n' "$RUN_DIR/.run.lock"
  rm -f "$RUN_DIR/.run.lock"
fi

TASK_LABEL="${MODEL_SLUG}_${DATASET}_seed${CALIBRATION_SEED}"
STRUCTURED_DIR="jobs/sycophancy_pruning/logs/$BUNDLE_NAME/by_task/$TASK_LABEL/sampling/job_${SLURM_JOB_ID:-local}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.err" >&2)

cmd=(
  "$ENV_PYTHON" run_sycophancy_bias_probe.py
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --device cuda
  --hf_cache_dir "$HF_CACHE_DIR"
  --benchmark_source ays_mc_single_turn
  --input_jsonl are_you_sure.jsonl
  --dataset_name "$DATASET"
  --ays_mc_datasets "$DATASET"
  --instruction_policy answer_only
  --bias_types incorrect_suggestion,incorrect_suggestion_strong,suggest_correct,suggest_correct_strong
  --behavior_generation
  --temperature 0
  --top_p 1
  --n_draws 1
  --sample_batch_size "${SAMPLE_BATCH_SIZE:-1}"
  --max_new_tokens "${MAX_NEW_TOKENS:-32}"
  --split_seed 5
  --test_frac 0.2
  --probe_val_frac 0.2
  --seed "$CALIBRATION_SEED"
  --sampling_only
  --run_name "$RUN_NAME"
  --out_dir "$SAMPLING_ROOT"
)
if [[ "${DEVICE_MAP_AUTO:-0}" == "1" ]]; then
  cmd+=(--device_map_auto)
fi
if [[ -n "${MAX_QUESTIONS:-}" ]]; then
  cmd+=(--max_questions "$MAX_QUESTIONS")
fi

printf -v command_text '%q ' "${cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
printf '[sampling] model=%s revision=%s dataset=%s calibration_seed=%s run_dir=%s\n' \
  "$MODEL_ID" "$MODEL_REVISION" "$DATASET" "$CALIBRATION_SEED" "$RUN_DIR"
START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
