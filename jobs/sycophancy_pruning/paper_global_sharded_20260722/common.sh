#!/usr/bin/env bash
set -euo pipefail

BUNDLE_NAME="paper_global_sharded_20260722"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
HARM_REPO_DIR="${HARM_REPO_DIR:-/n/home12/ishapira/harm_pruning_WIP}"
ENV_PYTHON="${ENV_PYTHON:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"

slugify() {
  "$ENV_PYTHON" -c 'import re,sys; print(re.sub(r"[^A-Za-z0-9_.-]+", "_", sys.argv[1]).strip("_") or "item")' "$1"
}

numeric_slug() {
  "$ENV_PYTHON" -c 'import sys; print(format(float(sys.argv[1]), ".12g"))' "$1"
}

file_sha256_short() {
  local path="$1"
  require_file "$path"
  "$ENV_PYTHON" -c \
    'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1], "rb"); [h.update(chunk) for chunk in iter(lambda: f.read(1024 * 1024), b"")]; print(h.hexdigest()[:12])' \
    "$path"
}

pruning_manifest_identity() {
  local prune_manifest="$1"
  local preserve_manifest="$2"
  local evaluation_manifest="$3"
  local prune_hash preserve_hash evaluation_hash
  prune_hash="$(file_sha256_short "$prune_manifest")"
  preserve_hash="$(file_sha256_short "$preserve_manifest")"
  evaluation_hash="$(file_sha256_short "$evaluation_manifest")"
  printf 'prune_%s_preserve_%s_eval_%s\n' \
    "$prune_hash" "$preserve_hash" "$evaluation_hash"
}

combined_identity() {
  "$ENV_PYTHON" -c \
    'import hashlib,sys; print(hashlib.sha256("\0".join(sys.argv[1:]).encode("utf-8")).hexdigest()[:16])' \
    "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf '%s\n' "Required file is missing: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    printf '%s\n' "Required directory is missing: $1" >&2
    exit 1
  fi
}

require_sha() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9a-fA-F]{7,64}$ ]]; then
    printf '%s\n' "$name must be an immutable hexadecimal commit revision, got: $value" >&2
    exit 1
  fi
}

setup_environment() {
  cd "$REPO_DIR"
  require_dir "$HARM_REPO_DIR/src"
  if [[ ! -x "$ENV_PYTHON" ]]; then
    printf '%s\n' "Missing python interpreter: $ENV_PYTHON" >&2
    exit 1
  fi
  if [[ ! -f .env ]]; then
    printf '%s\n' "Missing .env in $REPO_DIR" >&2
    exit 1
  fi

  module load python/3.10.9-fasrc01
  set -a
  source .env
  set +a

  if [[ -n "${HUGGINGFACE_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
  fi
  HF_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-}}"
  if [[ -z "$HF_CACHE_DIR" || "$HF_CACHE_DIR" == /home/* ]]; then
    printf '%s\n' "HUGGINGFACE_HUB_CACHE/HF_HUB_CACHE must point outside /home" >&2
    exit 1
  fi

  HF_HOME_DIR="${HF_HOME:-$(dirname "$HF_CACHE_DIR")/hf_home}"
  export HF_HUB_CACHE="$HF_CACHE_DIR"
  export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
  export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
  export HF_HOME="$HF_HOME_DIR"
  export TMPDIR="${SYCOPHANCY_TMPDIR:-${HF_HOME_DIR}/tmp}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${HF_HOME_DIR}/matplotlib}"
  export TORCH_HOME="${TORCH_HOME:-${HF_HOME_DIR}/torch}"
  export PYTHONPATH="$REPO_DIR/src:$HARM_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export WANDB_MODE="${WANDB_MODE:-offline}"

  EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(dirname "$HF_CACHE_DIR")/LLMsKnow_results/sycophancy_pruning/$BUNDLE_NAME}"
  SAMPLING_ROOT="${SAMPLING_ROOT:-$EXPERIMENT_ROOT/sampling}"
  MANIFEST_ROOT="${MANIFEST_ROOT:-$EXPERIMENT_ROOT/manifests}"
  PRUNING_ARTIFACT_ROOT="${PRUNING_ARTIFACT_ROOT:-$EXPERIMENT_ROOT/pruning_artifacts}"
  PREDICTION_ROOT="${PREDICTION_ROOT:-$EXPERIMENT_ROOT/predictions}"
  ANALYSIS_ROOT="${ANALYSIS_ROOT:-$EXPERIMENT_ROOT/analysis}"
  REGISTRY_ROOT="${REGISTRY_ROOT:-$EXPERIMENT_ROOT/registry}"
  ALPACA_DATA="${ALPACA_DATA:-}"
  mkdir -p \
    "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" \
    "$SAMPLING_ROOT" "$MANIFEST_ROOT" "$PRUNING_ARTIFACT_ROOT" \
    "$PREDICTION_ROOT" "$ANALYSIS_ROOT" "$REGISTRY_ROOT"
  export EXPERIMENT_ROOT SAMPLING_ROOT MANIFEST_ROOT PRUNING_ARTIFACT_ROOT
  export PREDICTION_ROOT ANALYSIS_ROOT REGISTRY_ROOT ALPACA_DATA HF_CACHE_DIR
}

model_id_for_index() {
  case "$1" in
    0) printf '%s\n' "Qwen/Qwen2.5-7B-Instruct" ;;
    1) printf '%s\n' "meta-llama/Llama-3.1-8B-Instruct" ;;
    *) printf '%s\n' "Unknown model index: $1" >&2; return 2 ;;
  esac
}

model_revision_for_index() {
  case "$1" in
    0) printf '%s\n' "${QWEN_REVISION:-}" ;;
    1) printf '%s\n' "${LLAMA_REVISION:-}" ;;
    *) printf '%s\n' "Unknown model index: $1" >&2; return 2 ;;
  esac
}

dataset_for_index() {
  case "$1" in
    0) printf '%s\n' "arc_challenge" ;;
    1) printf '%s\n' "commonsense_qa" ;;
    *) printf '%s\n' "Unknown dataset index: $1" >&2; return 2 ;;
  esac
}

size_to_n() {
  case "$1" in
    smoke) printf '%s\n' 16 ;;
    pilot) printf '%s\n' 128 ;;
    main) printf '%s\n' 412 ;;
    *) printf '%s\n' "Unknown manifest size: $1" >&2; return 2 ;;
  esac
}

sampling_run_name() {
  local model_slug="$1"
  local dataset="$2"
  local seed="$3"
  printf '%s\n' "syco_prune_${model_slug}_${dataset}_seed${seed}_20260722"
}

sampling_run_dir() {
  local model_id="$1"
  local dataset="$2"
  local seed="$3"
  local run_model_slug run_name
  run_model_slug="$(slugify "$model_id")"
  run_name="$(sampling_run_name "$run_model_slug" "$dataset" "$seed")"
  "$ENV_PYTHON" -c \
    'import sys; from llmssycoph.runtime import build_run_dir_path; print(build_run_dir_path(sys.argv[1], sys.argv[2], sys.argv[3], dataset_name=sys.argv[4], ays_mc_datasets=sys.argv[4]))' \
    "$SAMPLING_ROOT" "$model_id" "$run_name" "$dataset"
}

manifest_bundle_dir() {
  local model_id="$1"
  local revision="$2"
  printf '%s\n' "$MANIFEST_ROOT/$(slugify "$model_id")/revision_$(slugify "$revision")"
}

print_task_start() {
  local label="$1"
  local command_text="$2"
  printf '[%s] started_at=%s hostname=%s pwd=%s\n' "$label" "$(date -Is)" "$(hostname)" "$PWD"
  printf '[%s] slurm_job_id=%s array_job_id=%s array_task_id=%s\n' \
    "$label" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-0}"
  printf '[%s] command=%s\n' "$label" "$command_text"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
}

print_task_end() {
  local label="$1"
  local status="$2"
  local elapsed="$3"
  printf '[%s] finished_at=%s exit_status=%s elapsed_seconds=%s\n' \
    "$label" "$(date -Is)" "$status" "$elapsed"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sstat >/dev/null 2>&1; then
    sstat --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU -j "${SLURM_JOB_ID}.batch" || true
  fi
}
