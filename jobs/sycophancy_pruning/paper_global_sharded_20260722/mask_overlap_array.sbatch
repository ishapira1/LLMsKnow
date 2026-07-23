#!/bin/bash
#SBATCH --job-name=weight_pruning_overlap
#SBATCH --partition=shared,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/overlap/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/overlap/%x.%A_%a.err
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
EVALUATION_MANIFEST="$BUNDLE_DIR/evaluation/fixed_seed_5_heldout.jsonl"
MANIFEST_IDENTITY_5="$(pruning_manifest_identity \
  "$BUNDLE_DIR/seed_5/$MANIFEST_SIZE/pruning.jsonl" \
  "$BUNDLE_DIR/seed_5/$MANIFEST_SIZE/preservation.jsonl" \
  "$EVALUATION_MANIFEST")"
MANIFEST_IDENTITY_17="$(pruning_manifest_identity \
  "$BUNDLE_DIR/seed_17/$MANIFEST_SIZE/pruning.jsonl" \
  "$BUNDLE_DIR/seed_17/$MANIFEST_SIZE/preservation.jsonl" \
  "$EVALUATION_MANIFEST")"
MANIFEST_IDENTITY_29="$(pruning_manifest_identity \
  "$BUNDLE_DIR/seed_29/$MANIFEST_SIZE/pruning.jsonl" \
  "$BUNDLE_DIR/seed_29/$MANIFEST_SIZE/preservation.jsonl" \
  "$EVALUATION_MANIFEST")"
REPLICATION_IDENTITY="$(combined_identity \
  "$MANIFEST_IDENTITY_5" "$MANIFEST_IDENTITY_17" "$MANIFEST_IDENTITY_29")"
SELECTION_JSON="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/selection/$MANIFEST_IDENTITY_5/selected_configuration.json"
require_file "$SELECTION_JSON"

SELECTION_STATUS="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["status"])' "$SELECTION_JSON")"
TASK_LABEL="${MODEL_SLUG}_${MANIFEST_SIZE}_mask_overlap"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/$MODEL_SLUG/mask_overlap_${MANIFEST_SIZE}/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.err" >&2)

if [[ "$SELECTION_STATUS" != "selected" ]]; then
  print_task_start "$TASK_LABEL" "no feasible mask; overlap not applicable"
  printf '[overlap] model=%s status=no_feasible_mask; retaining base model\n' "$MODEL_ID"
  print_task_end "$TASK_LABEL" 0 0
  exit 0
fi

P_VALUE="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["selected_p"])' "$SELECTION_JSON")"
Q_VALUE="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["selected_q"])' "$SELECTION_JSON")"
CONFIG_SLUG="p_$(numeric_slug "$P_VALUE")_q_$(numeric_slug "$Q_VALUE")"

POINTER_5="$REGISTRY_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY_5/$CONFIG_SLUG.json"
POINTER_17="$REGISTRY_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_17/$MANIFEST_SIZE/selected/primary_seed17/$MANIFEST_IDENTITY_17/$CONFIG_SLUG.json"
POINTER_29="$REGISTRY_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_29/$MANIFEST_SIZE/selected/primary_seed29/$MANIFEST_IDENTITY_29/$CONFIG_SLUG.json"
for pointer in "$POINTER_5" "$POINTER_17" "$POINTER_29"; do require_file "$pointer"; done

MASK_5="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["indices_path"])' "$POINTER_5")"
MASK_17="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["indices_path"])' "$POINTER_17")"
MASK_29="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["indices_path"])' "$POINTER_29")"
for mask in "$MASK_5" "$MASK_17" "$MASK_29"; do require_file "$mask"; done

OUTPUT="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/replications/$MANIFEST_SIZE/$REPLICATION_IDENTITY/$CONFIG_SLUG/mask_overlap.json"
cmd=(
  "$ENV_PYTHON" scripts/compare_pruning_masks.py
  --mask "5=$MASK_5"
  --mask "17=$MASK_17"
  --mask "29=$MASK_29"
  --output "$OUTPUT"
)
printf -v command_text '%q ' "${cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
printf '[overlap] model=%s revision=%s replication_identity=%s p=%s q=%s output=%s\n' \
  "$MODEL_ID" "$MODEL_REVISION" "$REPLICATION_IDENTITY" "$P_VALUE" "$Q_VALUE" "$OUTPUT"
START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
