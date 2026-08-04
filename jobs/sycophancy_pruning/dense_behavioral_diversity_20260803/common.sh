#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
RESULT_ROOT="${RESULT_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/dense_behavioral_diversity_20260803}"
LOG_ROOT="${LOG_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/dense_behavioral_diversity_20260803}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/n/holystore01/LABS/barak_lab/Users/ishapira/hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION="0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_SNAPSHOT="$HF_CACHE_DIR/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/$MODEL_REVISION"
SIZE_ROOT="/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/size_diversity_1x2x5x_20260729"
DIVERSE_ROOT="/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/diverse_templates"
INPUTS_ROOT="$RESULT_ROOT/inputs"
STATES_CONFIG="$RESULT_ROOT/states.json"
CAMPAIGN_PY="$BUNDLE_DIR/campaign.py"
PREVIOUS_BUNDLE="$REPO_DIR/jobs/sycophancy_pruning/size_diversity_1x2x5x_20260729"
DIVERSE_BUNDLE="$REPO_DIR/jobs/sycophancy_pruning/diverse_templates"
ORIGINAL_PRUNE="$SIZE_ROOT/inputs/narrow_1x/prune_manifest.jsonl"
ORIGINAL_PRESERVE="$SIZE_ROOT/inputs/narrow_1x/preserve_mixed.jsonl"
BROAD_PRESERVE="$SIZE_ROOT/inputs/broad_5x/preserve_mixed.jsonl"
SAMPLING_ARC="$SIZE_ROOT/sampling/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/size_div_seed0_arc_llama31_20260729/sampling/raw/sampling_records.jsonl"
SAMPLING_CSQA="$SIZE_ROOT/sampling/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/size_div_seed0_csqa_llama31_20260729/sampling/raw/sampling_records.jsonl"
CAPABILITY_ROOT="/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/postprune_capability_audit_20260726"
CAPABILITY_AUDIT="$REPO_DIR/jobs/sycophancy_pruning/postprune_capability_audit_20260726/capability_audit.py"
CAPABILITY_INPUTS="$CAPABILITY_ROOT/inputs"
SYCOBENCH_SOURCE="/n/holystore01/LABS/barak_lab/Users/ishapira/source_snapshots/postprune_capability_audit_20260726/sycobench-600"
NONFACTUAL_EXPERIMENT_PY="$REPO_DIR/jobs/sycophancy_pruning/July_28_exp1/experiment.py"
NONFACTUAL_INPUTS="/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/July_28_exp1/inputs"
EMAIL_TO="itaishapira@g.harvard.edu"
UNITS=(dense_equal dense_correct2 dense_core2_correct2 dense_core2_correct4)

export BUNDLE_DIR REPO_DIR RESULT_ROOT LOG_ROOT HF_CACHE_DIR PYTHON_BIN MODEL_ID MODEL_REVISION MODEL_SNAPSHOT
export SIZE_ROOT DIVERSE_ROOT INPUTS_ROOT STATES_CONFIG CAMPAIGN_PY PREVIOUS_BUNDLE DIVERSE_BUNDLE
export ORIGINAL_PRUNE ORIGINAL_PRESERVE BROAD_PRESERVE SAMPLING_ARC SAMPLING_CSQA
export CAPABILITY_ROOT CAPABILITY_AUDIT CAPABILITY_INPUTS SYCOBENCH_SOURCE NONFACTUAL_EXPERIMENT_PY NONFACTUAL_INPUTS EMAIL_TO
export PYTHONPATH="$BUNDLE_DIR:$DIVERSE_BUNDLE:$PREVIOUS_BUNDLE:$REPO_DIR:$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR" HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR" HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_DIR/datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline WANDB_SILENT=true
export WANDB_DIR="${WANDB_DIR:-$RESULT_ROOT/wandb}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SAMPLE_BATCH_SIZE=1
export ALLOW_STALE_LOCK_CLEANUP="${ALLOW_STALE_LOCK_CLEANUP:-0}" USE_TF=0 USE_FLAX=0

TASK_LOG_PATH=""; TASK_ERR_PATH=""; TASK_START_EPOCH=""
start_task_log() {
  local stage="$1" label="$2" task_id="${SLURM_ARRAY_TASK_ID:-0}" job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  local task_dir="$LOG_ROOT/by_task/$label/$stage/job_$job_id"
  mkdir -p "$task_dir" "$RESULT_ROOT" "$WANDB_DIR"
  TASK_LOG_PATH="$task_dir/task_$task_id.out"; TASK_ERR_PATH="$task_dir/task_$task_id.err"; TASK_START_EPOCH="$(date +%s)"
  export TASK_LOG_PATH TASK_ERR_PATH TASK_START_EPOCH
  exec > >(tee -a "$TASK_LOG_PATH") 2> >(tee -a "$TASK_ERR_PATH" >&2)
  printf 'experiment=dense_behavioral_diversity_20260803\nstage=%s\ntask_label=%s\n' "$stage" "$label"
  printf 'model=%s\nrevision=%s\nrun_directory=%s\n' "$MODEL_ID" "$MODEL_REVISION" "$RESULT_ROOT"
  printf 'slurm_job_id=%s\nslurm_array_job_id=%s\nslurm_array_task_id=%s\n' "${SLURM_JOB_ID:-unset}" "${SLURM_ARRAY_JOB_ID:-unset}" "${SLURM_ARRAY_TASK_ID:-unset}"
  printf 'hostname=%s\nworking_directory=%s\nstart_time=%s\n' "$(hostname)" "$(pwd)" "$(date -Is)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
}
finish_task_log() {
  local status="$1" end_epoch="$(date +%s)"
  printf 'end_time=%s\nexit_status=%s\nelapsed_seconds=%s\n' "$(date -Is)" "$status" "$((end_epoch-TASK_START_EPOCH))"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  if command -v sstat >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then sstat --jobs "$SLURM_JOB_ID" --format=JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInMax || true; fi
}
print_command() { printf 'command='; printf '%q ' "$@"; printf '\n'; }
