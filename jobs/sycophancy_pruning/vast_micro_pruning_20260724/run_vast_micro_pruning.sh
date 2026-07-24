#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_NAME="vast_micro_pruning_20260724"
REPO_DIR="${REPO_DIR:-/workspace/LLMsKnow}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
WORK_ROOT="${WORK_ROOT:-/workspace/weight_pruning_micro_20260724}"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
CALIBRATION_SEED=5
MICRO_N=8
EVAL_QUESTIONS="${EVAL_QUESTIONS:-10}"
MAX_QUESTIONS="${MAX_QUESTIONS:-128}"
LAYERS=(3 8 13 18 23 27)
P_VALUE="1e-5"
Q_VALUE="5e-5"

HF_HOME="${HF_HOME:-/workspace/.hf_home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
SHM_ROOT="${SHM_ROOT:-/dev/shm/weight_pruning_micro_20260724}"
SCORE_CACHE="$SHM_ROOT/score_cache"
ARTIFACT_ROOT="$SHM_ROOT/artifacts"
SAMPLING_ROOT="$WORK_ROOT/sampling"
MANIFEST_ROOT="$WORK_ROOT/manifests"
PREDICTION_ROOT="$WORK_ROOT/results/predictions"
EVALUATION_ROOT="$WORK_ROOT/results/evaluation"
REPORT_ROOT="$WORK_ROOT/results/report"
LOG_ROOT="$WORK_ROOT/logs"
STATUS_PATH="$WORK_ROOT/status.json"
ALPACA_DATA="$WORK_ROOT/data/alpaca_data.json"
PRUNE_MANIFEST="$MANIFEST_ROOT/seed_5/micro/pruning.jsonl"
PRESERVE_MANIFEST="$MANIFEST_ROOT/seed_5/micro/preservation.jsonl"
FULL_EVALUATION_MANIFEST="$MANIFEST_ROOT/evaluation/fixed_seed_5_heldout.jsonl"
MICRO_EVALUATION_MANIFEST="$MANIFEST_ROOT/evaluation/micro_strict_val.jsonl"

export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/tools/weight_pruning${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export MPLCONFIGDIR="$WORK_ROOT/matplotlib"
export TMPDIR="$SHM_ROOT/tmp"

mkdir -p \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$SAMPLING_ROOT" "$MANIFEST_ROOT" \
  "$PREDICTION_ROOT" "$EVALUATION_ROOT" "$REPORT_ROOT" "$LOG_ROOT" \
  "$ARTIFACT_ROOT" "$SCORE_CACHE" "$MPLCONFIGDIR" "$TMPDIR" "$(dirname "$ALPACA_DATA")"
cd "$REPO_DIR"

write_status() {
  local stage="$1"
  local state="$2"
  "$PYTHON_BIN" -c '
import json,sys
from datetime import datetime, timezone
from pathlib import Path
path=Path(sys.argv[1])
payload={"bundle":sys.argv[2],"stage":sys.argv[3],"state":sys.argv[4],
         "updated_at":datetime.now(timezone.utc).isoformat()}
path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
' "$STATUS_PATH" "$BUNDLE_NAME" "$stage" "$state"
}

fail() {
  local status="$?"
  write_status "${CURRENT_STAGE:-unknown}" "failed_exit_${status}"
  printf '[weight-pruning-micro] failed stage=%s exit=%s time=%s\n' \
    "${CURRENT_STAGE:-unknown}" "$status" "$(date -Is)" >&2
  exit "$status"
}
trap fail ERR

run_stage() {
  local stage="$1"
  shift
  CURRENT_STAGE="$stage"
  write_status "$stage" running
  printf '[weight-pruning-micro] stage=%s start=%s command=' "$stage" "$(date -Is)"
  printf '%q ' "$@"
  printf '\n'
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
  write_status "$stage" completed
}

CURRENT_STAGE=preflight
write_status "$CURRENT_STAGE" running
"$PYTHON_BIN" -c '
import shutil, torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 2, "two GPUs are required"
free = shutil.disk_usage("/dev/shm").free
assert free >= 20 * 1024**3, f"need 20 GiB free in /dev/shm, found {free / 1024**3:.1f}"
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
       "shm_free_gib": round(free / 1024**3, 1)})
'
write_status "$CURRENT_STAGE" completed

if [[ ! -f "$PRUNE_MANIFEST" || ! -f "$PRESERVE_MANIFEST" || ! -f "$FULL_EVALUATION_MANIFEST" ]]; then
  run_stage prepare_alpaca \
    "$PYTHON_BIN" scripts/prepare_weight_pruning_alpaca.py --output "$ALPACA_DATA"

sample_dataset() {
  local gpu="$1"
  local dataset="$2"
  local run_name="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_sycophancy_bias_probe.py \
    --model "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --device cuda \
    --hf_cache_dir "$HF_HUB_CACHE" \
    --benchmark_source ays_mc_single_turn \
    --input_jsonl are_you_sure.jsonl \
    --dataset_name "$dataset" \
    --ays_mc_datasets "$dataset" \
    --instruction_policy answer_only \
    --bias_types incorrect_suggestion,incorrect_suggestion_strong,suggest_correct,suggest_correct_strong \
    --behavior_generation \
    --temperature 0 \
    --top_p 1 \
    --n_draws 1 \
    --sample_batch_size 8 \
    --max_new_tokens 32 \
    --split_seed 5 \
    --test_frac 0.2 \
    --probe_val_frac 0.2 \
    --seed "$CALIBRATION_SEED" \
    --sampling_only \
    --run_name "$run_name" \
    --out_dir "$SAMPLING_ROOT" \
    --max_questions "$MAX_QUESTIONS"
}

CURRENT_STAGE=sampling
write_status "$CURRENT_STAGE" running
ARC_RUN_NAME="weight_pruning_micro_qwen_arc_seed5_20260724"
CSQA_RUN_NAME="weight_pruning_micro_qwen_csqa_seed5_20260724"
sample_dataset 0 arc_challenge "$ARC_RUN_NAME" >"$LOG_ROOT/sampling_arc.log" 2>&1 &
ARC_PID="$!"
sample_dataset 1 commonsense_qa "$CSQA_RUN_NAME" >"$LOG_ROOT/sampling_csqa.log" 2>&1 &
CSQA_PID="$!"
wait "$ARC_PID"
wait "$CSQA_PID"
write_status "$CURRENT_STAGE" completed

ARC_RUN_DIR="$("$PYTHON_BIN" -c \
  'import sys; from llmssycoph.runtime import build_run_dir_path; print(build_run_dir_path(*sys.argv[1:4], dataset_name=sys.argv[4], ays_mc_datasets=sys.argv[4]))' \
  "$SAMPLING_ROOT" "$MODEL_ID" "$ARC_RUN_NAME" arc_challenge)"
CSQA_RUN_DIR="$("$PYTHON_BIN" -c \
  'import sys; from llmssycoph.runtime import build_run_dir_path; print(build_run_dir_path(*sys.argv[1:4], dataset_name=sys.argv[4], ays_mc_datasets=sys.argv[4]))' \
  "$SAMPLING_ROOT" "$MODEL_ID" "$CSQA_RUN_NAME" commonsense_qa)"

  run_stage build_manifests \
    "$PYTHON_BIN" scripts/build_strict_sycophancy_manifests.py \
    --model-id "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --seed-records "5=$ARC_RUN_DIR" \
    --seed-records "5=$CSQA_RUN_DIR" \
    --alpaca-data "$ALPACA_DATA" \
    --alpaca-utility-size 32 \
    --evaluation-records "$ARC_RUN_DIR" \
    --evaluation-records "$CSQA_RUN_DIR" \
    --output-dir "$MANIFEST_ROOT" \
    --expected-seeds 5 \
    --sizes "micro=$MICRO_N"
else
  printf '[weight-pruning-micro] reusing validated calibration/evaluation manifests\n'
fi
run_stage subset_evaluation \
  "$PYTHON_BIN" scripts/subset_pruning_evaluation_manifest.py \
  --input "$FULL_EVALUATION_MANIFEST" \
  --output "$MICRO_EVALUATION_MANIFEST" \
  --questions "$EVAL_QUESTIONS" \
  --seed "$CALIBRATION_SEED" \
  --require-baseline-strict-flip

COMMON_SCORE_ARGS=(
  tools/weight_pruning/prune.py
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --tokenizer_revision "$MODEL_REVISION"
  --prune_method attribution_score_set_difference_global
  --prune_manifest "$PRUNE_MANIFEST"
  --preserve_manifest "$PRESERVE_MANIFEST"
  --nsamples "$MICRO_N"
  --nsamples_preserve "$MICRO_N"
  --seed "$CALIBRATION_SEED"
  --score_format raw
  --loss_mode completion_nll
  --attribution_variant paper
  --artifact_root "$ARTIFACT_ROOT"
  --score_cache "$SCORE_CACHE"
  --layers "${LAYERS[@]}"
  --control none
  --no_abs
  --neg_prune
  --abs_preserve
)

CURRENT_STAGE=scoring
write_status "$CURRENT_STAGE" running
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "${COMMON_SCORE_ARGS[@]}" \
  --score_role prune --dump_score >"$LOG_ROOT/score_prune.log" 2>&1 &
PRUNE_PID="$!"
for _ in $(seq 1 180); do
  [[ -f "$SCORE_CACHE/identity.json" ]] && break
  kill -0 "$PRUNE_PID" 2>/dev/null || break
  sleep 1
done
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "${COMMON_SCORE_ARGS[@]}" \
  --score_role preserve --dump_score >"$LOG_ROOT/score_preserve.log" 2>&1 &
PRESERVE_PID="$!"
wait "$PRUNE_PID"
wait "$PRESERVE_PID"
write_status "$CURRENT_STAGE" completed

CURRENT_STAGE=mask
write_status "$CURRENT_STAGE" running
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "${COMMON_SCORE_ARGS[@]}" \
  --use_saved_scores \
  --p "$P_VALUE" \
  --q "$Q_VALUE" \
  --dump_mask \
  --dump_indices \
  --mask_only 2>&1 | tee "$LOG_ROOT/mask.log"
write_status "$CURRENT_STAGE" completed

POINTER_PATH="$WORK_ROOT/results/targeted_mask_pointer.json"
run_stage resolve_mask \
  "$PYTHON_BIN" scripts/resolve_paper_pruning_artifact.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --score-cache "$SCORE_CACHE" \
  --model "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --prune-manifest "$PRUNE_MANIFEST" \
  --preserve-manifest "$PRESERVE_MANIFEST" \
  --nsamples "$MICRO_N" \
  --nsamples-preserve "$MICRO_N" \
  --seed "$CALIBRATION_SEED" \
  --score-format raw \
  --loss-mode completion_nll \
  --attribution-variant paper \
  --control none \
  --p "$P_VALUE" \
  --q "$Q_VALUE" \
  --no-abs \
  --neg-prune \
  --abs-preserve \
  --layers "${LAYERS[@]}" \
  --require-existing \
  --mask-artifacts-only \
  --output "$POINTER_PATH"

readarray -t MASK_PATHS < <("$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print(p["indices_path"])
print(p["metadata_path"])
print(json.load(open(p["metadata_path"]))["surviving_count"])
' "$POINTER_PATH")
INDICES_PATH="${MASK_PATHS[0]}"
MASK_METADATA="${MASK_PATHS[1]}"
ACTUAL_MASK_COUNT="${MASK_PATHS[2]}"

base_inference() {
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/run_pruning_heldout_inference.py \
    --evaluation-manifest "$MICRO_EVALUATION_MANIFEST" \
    --output-dir "$PREDICTION_ROOT/base" \
    --p 0 \
    --q 0 \
    --calibration-seed "$CALIBRATION_SEED" \
    --device cuda \
    --hf-cache-dir "$HF_HUB_CACHE" \
    --torch-dtype auto \
    --max-new-tokens 16 \
    --generation-seed 5 \
    --splits val \
    --overwrite
}

targeted_inference() {
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/run_pruning_heldout_inference.py \
    --evaluation-manifest "$MICRO_EVALUATION_MANIFEST" \
    --output-dir "$PREDICTION_ROOT/targeted" \
    --indices-path "$INDICES_PATH" \
    --mask-metadata "$MASK_METADATA" \
    --expected-mask-count "$ACTUAL_MASK_COUNT" \
    --p "$P_VALUE" \
    --q "$Q_VALUE" \
    --calibration-seed "$CALIBRATION_SEED" \
    --device cuda \
    --hf-cache-dir "$HF_HUB_CACHE" \
    --torch-dtype auto \
    --max-new-tokens 16 \
    --generation-seed 5 \
    --splits val \
    --overwrite
}

CURRENT_STAGE=live_inference
write_status "$CURRENT_STAGE" running
base_inference >"$LOG_ROOT/inference_base.log" 2>&1 &
BASE_PID="$!"
targeted_inference >"$LOG_ROOT/inference_targeted.log" 2>&1 &
TARGET_PID="$!"
wait "$BASE_PID"
wait "$TARGET_PID"
write_status "$CURRENT_STAGE" completed

run_stage aggregate \
  "$PYTHON_BIN" scripts/aggregate_pruning_offline_evaluation.py \
  --baseline "$PREDICTION_ROOT/base/candidate_items.jsonl" \
  --candidate "$PREDICTION_ROOT/targeted/candidate_items.jsonl" \
  --output-dir "$EVALUATION_ROOT/targeted" \
  --p "$P_VALUE" \
  --q "$Q_VALUE" \
  --calibration-seed "$CALIBRATION_SEED" \
  --actual-mask-count "$ACTUAL_MASK_COUNT" \
  --n-bootstrap 200 \
  --bootstrap-seed 5

run_stage summarize \
  "$PYTHON_BIN" scripts/summarize_pruning_micro_pilot.py \
  --targeted-evaluation-dir "$EVALUATION_ROOT/targeted" \
  --targeted-mask-metadata "$MASK_METADATA" \
  --output-dir "$REPORT_ROOT"

cp "$INDICES_PATH" "$WORK_ROOT/results/targeted_indices.pt"
cp "$MASK_METADATA" "$WORK_ROOT/results/targeted_mask_metadata.json"
cp "$MICRO_EVALUATION_MANIFEST" "$WORK_ROOT/results/micro_strict_val.jsonl"
cp "$MICRO_EVALUATION_MANIFEST.audit.json" "$WORK_ROOT/results/micro_strict_val.audit.json"
CURRENT_STAGE=complete
write_status "$CURRENT_STAGE" completed
printf '[weight-pruning-micro] complete=%s report=%s\n' "$(date -Is)" "$REPORT_ROOT"
