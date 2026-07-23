#!/bin/bash
#SBATCH --job-name=weight_pruning_tokens
#SBATCH --partition=shared,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/token_snapshots/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/token_snapshots/%x.%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
require_slurm_array_task
setup_environment

MODEL_INDEX="$SLURM_ARRAY_TASK_ID"
MODEL_ID="$(model_id_for_index "$MODEL_INDEX")"
MODEL_REVISION="$(model_revision_for_index "$MODEL_INDEX")"
require_sha "MODEL_REVISION" "$MODEL_REVISION"
MODEL_SLUG="$(slugify "$MODEL_ID")"
BUNDLE_DIR="$(manifest_bundle_dir "$MODEL_ID" "$MODEL_REVISION")"
PRUNE_MANIFEST="$BUNDLE_DIR/seed_5/smoke/pruning.jsonl"
PRESERVE_MANIFEST="$BUNDLE_DIR/seed_5/smoke/preservation.jsonl"
OUTPUT_DIR="$BUNDLE_DIR/token_snapshots"
require_file "$PRUNE_MANIFEST"
require_file "$PRESERVE_MANIFEST"

TASK_LABEL="${MODEL_SLUG}_token_snapshot"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/$MODEL_SLUG/token_snapshots/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.err" >&2)

prune_cmd=(
  "$ENV_PYTHON" scripts/snapshot_pruning_tokenization.py
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --manifest "$PRUNE_MANIFEST"
  --output "$OUTPUT_DIR/raw_chat_seed5_smoke_pruning.json"
  --max-rows 4
  --hf-cache-dir "$HF_CACHE_DIR"
)
preserve_cmd=(
  "$ENV_PYTHON" scripts/snapshot_pruning_tokenization.py
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --manifest "$PRESERVE_MANIFEST"
  --output "$OUTPUT_DIR/raw_chat_seed5_smoke_preservation.json"
  --max-rows 16
  --hf-cache-dir "$HF_CACHE_DIR"
)
printf -v prune_command_text '%q ' "${prune_cmd[@]}"
printf -v preserve_command_text '%q ' "${preserve_cmd[@]}"
command_text="$prune_command_text && $preserve_command_text"
print_task_start "$TASK_LABEL" "$command_text"
START_EPOCH="$(date +%s)"
set +e
"${prune_cmd[@]}" && "${preserve_cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
