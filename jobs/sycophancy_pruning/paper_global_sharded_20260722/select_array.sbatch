#!/bin/bash
#SBATCH --job-name=weight_pruning_select
#SBATCH --partition=shared,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/selection/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/selection/%x.%A_%a.err
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
MANIFEST_SIZE="${MANIFEST_SIZE:-main}"
BUNDLE_DIR="$(manifest_bundle_dir "$MODEL_ID" "$MODEL_REVISION")"
PRUNE_MANIFEST="$BUNDLE_DIR/seed_5/${MANIFEST_SIZE}/pruning.jsonl"
PRESERVE_MANIFEST="$BUNDLE_DIR/seed_5/${MANIFEST_SIZE}/preservation.jsonl"
EVALUATION_MANIFEST="$BUNDLE_DIR/evaluation/fixed_seed_5_heldout.jsonl"
MANIFEST_IDENTITY="$(pruning_manifest_identity \
  "$PRUNE_MANIFEST" "$PRESERVE_MANIFEST" "$EVALUATION_MANIFEST")"
GRID_ROOT="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY"
OUTPUT_DIR="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/selection/$MANIFEST_IDENTITY"

TASK_LABEL="${MODEL_SLUG}_${MANIFEST_SIZE}_selection"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/$MODEL_SLUG/selection_${MANIFEST_SIZE}/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.err" >&2)

cmd=(
  "$ENV_PYTHON" scripts/collect_pruning_grid.py
  --grid-root "$GRID_ROOT"
  --output-dir "$OUTPUT_DIR"
  --split val
  --calibration-seed 5
  --artifact-identity "$MANIFEST_IDENTITY"
)
printf -v command_text '%q ' "${cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
