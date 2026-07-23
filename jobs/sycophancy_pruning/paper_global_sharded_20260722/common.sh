#!/usr/bin/env bash
set -euo pipefail

BUNDLE_NAME="paper_global_sharded_20260722"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
WEIGHT_PRUNING_DIR="$REPO_DIR/tools/weight_pruning"
ENV_PYTHON="${ENV_PYTHON:-/n/holystore01/LABS/barak_lab/Users/ishapira/python_envs/llmsknow_py310_torch220_transformers4423/bin/python}"
EXPECTED_SLURM_CLUSTER="odyssey"
QWEN_REVISION="${QWEN_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
LLAMA_REVISION="${LLAMA_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}"

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

slurm_cluster_name() {
  if command -v scontrol >/dev/null 2>&1; then
    scontrol show config | awk '$1 == "ClusterName" {print $3; exit}'
  elif [[ -n "${SLURM_CLUSTER_NAME:-}" ]]; then
    printf '%s\n' "$SLURM_CLUSTER_NAME"
  fi
}

require_harvard_scheduler() {
  if ! command -v sbatch >/dev/null 2>&1 || ! command -v scontrol >/dev/null 2>&1; then
    printf '%s\n' \
      "The weight_pruning launcher requires the Harvard Slurm commands sbatch and scontrol." >&2
    exit 2
  fi
  local cluster_name
  cluster_name="$(slurm_cluster_name)"
  if [[ "$cluster_name" != "$EXPECTED_SLURM_CLUSTER" ]]; then
    printf '%s\n' \
      "Expected Harvard Slurm cluster $EXPECTED_SLURM_CLUSTER, got ${cluster_name:-unknown}" >&2
    exit 2
  fi
}

require_slurm_job() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' \
      "This script must run inside a Slurm allocation. Submit it with the weight_pruning wrapper; do not execute it with bash." >&2
    exit 2
  fi
  local cluster_name
  cluster_name="$(slurm_cluster_name)"
  if [[ "$cluster_name" != "$EXPECTED_SLURM_CLUSTER" ]]; then
    printf '%s\n' \
      "Expected Harvard Slurm cluster $EXPECTED_SLURM_CLUSTER, got ${cluster_name:-unknown}" >&2
    exit 2
  fi
}

require_slurm_array_task() {
  require_slurm_job
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf '%s\n' "This weight_pruning stage requires a Slurm array task ID." >&2
    exit 2
  fi
}

require_gpu_allocation() {
  require_slurm_array_task
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
    printf '%s\n' \
      "This weight_pruning stage requires a Slurm GPU allocation with CUDA_VISIBLE_DEVICES set." >&2
    exit 2
  fi
}

require_free_space_gb() {
  local path="$1"
  local minimum_gb="$2"
  local available_kb required_kb
  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  if [[ ! "$available_kb" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "Could not determine available storage for $path" >&2
    exit 1
  fi
  required_kb=$((minimum_gb * 1024 * 1024))
  if (( available_kb < required_kb )); then
    printf '%s\n' \
      "Insufficient weight_pruning storage at $path: require ${minimum_gb} GiB free, found $((available_kb / 1024 / 1024)) GiB" >&2
    exit 1
  fi
  printf '[weight_pruning] storage_root=%s free_gib=%s required_gib=%s\n' \
    "$path" "$((available_kb / 1024 / 1024))" "$minimum_gb"
}

require_quota_headroom_gb() {
  local path="$1"
  local minimum_gb="$2"
  if ! command -v quota >/dev/null 2>&1; then
    printf '%s\n' "Harvard quota command is unavailable; cannot verify weight_pruning allocation headroom." >&2
    exit 1
  fi
  local quota_output usage_fields used_value quota_value available_gb
  quota_output="$(quota "$path")"
  printf '%s\n' "$quota_output"
  usage_fields="$(printf '%s\n' "$quota_output" | awk '$1 ~ /^\// {print $2, $3; exit}')"
  read -r used_value quota_value <<< "$usage_fields"
  if [[ -z "${used_value:-}" || -z "${quota_value:-}" ]]; then
    printf '%s\n' "Could not parse Harvard quota output for $path" >&2
    exit 1
  fi
  available_gb="$("$ENV_PYTHON" -c '
import re
import sys

def to_gib(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", value.upper())
    if not match:
        raise SystemExit(f"unsupported quota value: {value}")
    number = float(match.group(1))
    suffix = match.group(2)
    if not suffix:
        return number / (1024 * 1024)
    factors = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3, "E": 1024**4}
    return number * factors[suffix]

print(int(to_gib(sys.argv[2]) - to_gib(sys.argv[1])))
' "$used_value" "$quota_value")"
  if [[ ! "$available_gb" =~ ^-?[0-9]+$ ]] || (( available_gb < minimum_gb )); then
    printf '%s\n' \
      "Insufficient weight_pruning quota at $path: require ${minimum_gb} GiB headroom, found ${available_gb:-unknown} GiB" >&2
    exit 1
  fi
  printf '[weight_pruning] quota_root=%s quota_headroom_gib=%s required_gib=%s\n' \
    "$path" "$available_gb" "$minimum_gb"
}

require_large_storage_path() {
  local name="$1"
  local path="$2"
  if [[ -z "$path" || "$path" == /home/* || "$path" == /n/home* ]]; then
    printf '%s\n' "$name must point to large storage outside home: ${path:-<empty>}" >&2
    exit 1
  fi
}

setup_environment() {
  cd "$REPO_DIR"
  require_dir "$WEIGHT_PRUNING_DIR"
  require_file "$WEIGHT_PRUNING_DIR/prune.py"
  require_file "$WEIGHT_PRUNING_DIR/paper_pruning.py"
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
  require_large_storage_path "HUGGINGFACE_HUB_CACHE/HF_HUB_CACHE" "$HF_CACHE_DIR"

  STORAGE_ROOT="${SYCOPHANCY_STORAGE_ROOT:-$(dirname "$(dirname "$HF_CACHE_DIR")")}"
  require_large_storage_path "SYCOPHANCY_STORAGE_ROOT" "$STORAGE_ROOT"
  require_dir "$STORAGE_ROOT"
  require_free_space_gb "$STORAGE_ROOT" "${WEIGHT_PRUNING_MIN_FREE_GB:-1600}"
  require_quota_headroom_gb "$STORAGE_ROOT" "${WEIGHT_PRUNING_MIN_FREE_GB:-1600}"

  HF_HOME_DIR="${HF_HOME:-$(dirname "$HF_CACHE_DIR")/hf_home}"
  require_large_storage_path "HF_HOME" "$HF_HOME_DIR"
  export HF_HUB_CACHE="$HF_CACHE_DIR"
  export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
  export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
  export HF_HOME="$HF_HOME_DIR"
  export TMPDIR="${SYCOPHANCY_TMPDIR:-${HF_HOME_DIR}/tmp}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${HF_HOME_DIR}/matplotlib}"
  export TORCH_HOME="${TORCH_HOME:-${HF_HOME_DIR}/torch}"
  export PYTHONPATH="$REPO_DIR/src:$WEIGHT_PRUNING_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_DIR="${WANDB_DIR:-$STORAGE_ROOT/LLMsKnow_wandb/$BUNDLE_NAME/runs}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$STORAGE_ROOT/LLMsKnow_wandb/$BUNDLE_NAME/cache}"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$STORAGE_ROOT/LLMsKnow_wandb/$BUNDLE_NAME/config}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$STORAGE_ROOT/LLMsKnow_wandb/$BUNDLE_NAME/data}"

  EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$STORAGE_ROOT/LLMsKnow_results/sycophancy_pruning/$BUNDLE_NAME}"
  require_large_storage_path "EXPERIMENT_ROOT" "$EXPERIMENT_ROOT"
  SAMPLING_ROOT="${SAMPLING_ROOT:-$EXPERIMENT_ROOT/sampling}"
  MANIFEST_ROOT="${MANIFEST_ROOT:-$EXPERIMENT_ROOT/manifests}"
  PRUNING_ARTIFACT_ROOT="${PRUNING_ARTIFACT_ROOT:-$EXPERIMENT_ROOT/pruning_artifacts}"
  PREDICTION_ROOT="${PREDICTION_ROOT:-$EXPERIMENT_ROOT/predictions}"
  ANALYSIS_ROOT="${ANALYSIS_ROOT:-$EXPERIMENT_ROOT/analysis}"
  REGISTRY_ROOT="${REGISTRY_ROOT:-$EXPERIMENT_ROOT/registry}"
  WEIGHT_PRUNING_LOG_ROOT="${WEIGHT_PRUNING_LOG_ROOT:-$STORAGE_ROOT/LLMsKnow_logs/sycophancy_pruning/$BUNDLE_NAME}"
  ALPACA_DATA="${ALPACA_DATA:-$STORAGE_ROOT/datasets/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json}"
  require_large_storage_path "TMPDIR" "$TMPDIR"
  require_large_storage_path "MPLCONFIGDIR" "$MPLCONFIGDIR"
  require_large_storage_path "TORCH_HOME" "$TORCH_HOME"
  require_large_storage_path "HF_DATASETS_CACHE" "$HF_DATASETS_CACHE"
  require_large_storage_path "SAMPLING_ROOT" "$SAMPLING_ROOT"
  require_large_storage_path "MANIFEST_ROOT" "$MANIFEST_ROOT"
  require_large_storage_path "PRUNING_ARTIFACT_ROOT" "$PRUNING_ARTIFACT_ROOT"
  require_large_storage_path "PREDICTION_ROOT" "$PREDICTION_ROOT"
  require_large_storage_path "ANALYSIS_ROOT" "$ANALYSIS_ROOT"
  require_large_storage_path "REGISTRY_ROOT" "$REGISTRY_ROOT"
  require_large_storage_path "WEIGHT_PRUNING_LOG_ROOT" "$WEIGHT_PRUNING_LOG_ROOT"
  require_large_storage_path "ALPACA_DATA" "$ALPACA_DATA"
  require_large_storage_path "WANDB_DIR" "$WANDB_DIR"
  require_large_storage_path "WANDB_CACHE_DIR" "$WANDB_CACHE_DIR"
  require_large_storage_path "WANDB_CONFIG_DIR" "$WANDB_CONFIG_DIR"
  require_large_storage_path "WANDB_DATA_DIR" "$WANDB_DATA_DIR"
  mkdir -p \
    "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" \
    "$SAMPLING_ROOT" "$MANIFEST_ROOT" "$PRUNING_ARTIFACT_ROOT" \
    "$PREDICTION_ROOT" "$ANALYSIS_ROOT" "$REGISTRY_ROOT" "$WEIGHT_PRUNING_LOG_ROOT/by_task" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
  export EXPERIMENT_ROOT SAMPLING_ROOT MANIFEST_ROOT PRUNING_ARTIFACT_ROOT
  export PREDICTION_ROOT ANALYSIS_ROOT REGISTRY_ROOT ALPACA_DATA HF_CACHE_DIR STORAGE_ROOT
  export WEIGHT_PRUNING_LOG_ROOT
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
  printf '%s\n' "weight_pruning_${model_slug}_${dataset}_seed${seed}_20260723"
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
    "$label" "$SLURM_JOB_ID" "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID:-not_array}"
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
