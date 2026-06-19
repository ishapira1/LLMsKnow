#!/usr/bin/env bash

# Shared storage policy for Harvard cluster sycophancy-bias Slurm jobs.
# Source this after `.env`, then call:
#   configure_sycophancy_bias_storage "$BUNDLE_NAME"

sycophancy_bias_is_home_path() {
  local path="$1"
  case "$path" in
    /home|/home/*|/n/home*) return 0 ;;
    *) return 1 ;;
  esac
}

sycophancy_bias_abs_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "${ROOT_DIR:-${REPO_DIR:-$PWD}}" "$path"
  fi
}

sycophancy_bias_reject_home_path() {
  local label="$1"
  local path="$2"
  local abs_path
  abs_path="$(sycophancy_bias_abs_path "$path")"
  if sycophancy_bias_is_home_path "$abs_path"; then
    printf '%s\n' "Refusing to run: $label points to home quota storage ($abs_path)" >&2
    exit 1
  fi
}

sycophancy_bias_root_from_storage_path() {
  local path="${1%/}"
  local base parent parent_base
  if [[ -z "$path" ]]; then
    return 1
  fi

  base="$(basename "$path")"
  parent="$(dirname "$path")"
  parent_base="$(basename "$parent")"

  case "$base" in
    hub|datasets|home|tmp|torch|matplotlib)
      if [[ "$parent_base" == "hf_cache" ]]; then
        dirname "$parent"
      else
        printf '%s\n' "$parent"
      fi
      ;;
    cache|config|data)
      if [[ "$parent_base" == "wandb" ]]; then
        dirname "$parent"
      else
        printf '%s\n' "$parent"
      fi
      ;;
    hf_cache|triton_cache|wandb|xdg_cache)
      printf '%s\n' "$parent"
      ;;
    *)
      printf '%s\n' "$path"
      ;;
  esac
}

sycophancy_bias_infer_storage_root() {
  local default_root="${SYCOPHANCY_DEFAULT_STORAGE_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira}"
  local candidate

  if [[ -n "${SYCOPHANCY_STORAGE_ROOT:-}" ]]; then
    printf '%s\n' "${SYCOPHANCY_STORAGE_ROOT%/}"
    return 0
  fi

  if [[ -d "$default_root" || "${REPO_DIR:-}" == /n/home* || "${ROOT_DIR:-}" == /n/home* || "${HOME:-}" == /n/home* ]]; then
    printf '%s\n' "$default_root"
    return 0
  fi

  for candidate in \
    "${HUGGINGFACE_HUB_CACHE:-}" \
    "${HF_HUB_CACHE:-}" \
    "${HF_DATASETS_CACHE:-}" \
    "${TRITON_CACHE_DIR:-}" \
    "${WANDB_DIR:-}" \
    "${WANDB_CACHE_DIR:-}" \
    "${XDG_CACHE_HOME:-}"
  do
    if [[ -n "$candidate" ]]; then
      sycophancy_bias_root_from_storage_path "$candidate"
      return 0
    fi
  done

  printf '%s\n' "$default_root"
}

configure_sycophancy_bias_storage() {
  local bundle_name="${1:?missing bundle name}"
  local hf_cache_parent

  if [[ -n "${HUGGINGFACE_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
  fi

  SYCOPHANCY_STORAGE_ROOT="$(sycophancy_bias_infer_storage_root)"
  SYCOPHANCY_STORAGE_ROOT="${SYCOPHANCY_STORAGE_ROOT%/}"
  if [[ -z "$SYCOPHANCY_STORAGE_ROOT" ]]; then
    printf '%s\n' "Unable to resolve SYCOPHANCY_STORAGE_ROOT" >&2
    exit 1
  fi
  sycophancy_bias_reject_home_path "SYCOPHANCY_STORAGE_ROOT" "$SYCOPHANCY_STORAGE_ROOT"

  hf_cache_parent="${SYCOPHANCY_HF_CACHE_PARENT:-${SYCOPHANCY_STORAGE_ROOT}/hf_cache}"
  HF_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-${hf_cache_parent}/hub}}"
  HF_DATASETS_DIR="${HF_DATASETS_CACHE:-${hf_cache_parent}/datasets}"
  HF_HOME_DIR="${HF_HOME:-${hf_cache_parent}/home}"

  HF_HUB_CACHE="$HF_CACHE_DIR"
  HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
  TRANSFORMERS_CACHE="$HF_CACHE_DIR"
  HF_DATASETS_CACHE="$HF_DATASETS_DIR"
  HF_HOME="$HF_HOME_DIR"

  TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SYCOPHANCY_STORAGE_ROOT}/triton_cache}"
  WANDB_DIR="${WANDB_DIR:-${SYCOPHANCY_STORAGE_ROOT}/wandb}"
  WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}"
  WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/data}"
  TMPDIR="${SYCOPHANCY_TMPDIR:-${HF_HOME_DIR}/tmp}"
  MPLCONFIGDIR="${MPLCONFIGDIR:-${HF_HOME_DIR}/matplotlib}"
  TORCH_HOME="${TORCH_HOME:-${HF_HOME_DIR}/torch}"
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SYCOPHANCY_STORAGE_ROOT}/xdg_cache}"

  SYCOPHANCY_BIAS_RESULTS_DIR="${SYCOPHANCY_BIAS_RESULTS_DIR:-${SYCOPHANCY_STORAGE_ROOT}/LLMsKnow_results/sycophancy_bias_probe}"
  OUT_DIR="${OUT_DIR:-$SYCOPHANCY_BIAS_RESULTS_DIR}"

  SYCOPHANCY_BIAS_LOG_ROOT="${SYCOPHANCY_BIAS_LOG_ROOT:-${SYCOPHANCY_STORAGE_ROOT}/LLMsKnow_logs/sycophancy_bias_probe/${bundle_name}}"
  LOG_ROOT="${LOG_ROOT:-$SYCOPHANCY_BIAS_LOG_ROOT}"

  sycophancy_bias_reject_home_path "HF_HUB_CACHE" "$HF_HUB_CACHE"
  sycophancy_bias_reject_home_path "HF_DATASETS_CACHE" "$HF_DATASETS_CACHE"
  sycophancy_bias_reject_home_path "HF_HOME" "$HF_HOME"
  sycophancy_bias_reject_home_path "TRITON_CACHE_DIR" "$TRITON_CACHE_DIR"
  sycophancy_bias_reject_home_path "WANDB_DIR" "$WANDB_DIR"
  sycophancy_bias_reject_home_path "WANDB_CACHE_DIR" "$WANDB_CACHE_DIR"
  sycophancy_bias_reject_home_path "WANDB_CONFIG_DIR" "$WANDB_CONFIG_DIR"
  sycophancy_bias_reject_home_path "WANDB_DATA_DIR" "$WANDB_DATA_DIR"
  sycophancy_bias_reject_home_path "TMPDIR" "$TMPDIR"
  sycophancy_bias_reject_home_path "MPLCONFIGDIR" "$MPLCONFIGDIR"
  sycophancy_bias_reject_home_path "TORCH_HOME" "$TORCH_HOME"
  sycophancy_bias_reject_home_path "XDG_CACHE_HOME" "$XDG_CACHE_HOME"
  sycophancy_bias_reject_home_path "OUT_DIR" "$OUT_DIR"
  sycophancy_bias_reject_home_path "LOG_ROOT" "$LOG_ROOT"

  export SYCOPHANCY_STORAGE_ROOT
  export SYCOPHANCY_BIAS_RESULTS_DIR
  export SYCOPHANCY_BIAS_LOG_ROOT
  export HF_HUB_CACHE
  export HUGGINGFACE_HUB_CACHE
  export TRANSFORMERS_CACHE
  export HF_DATASETS_CACHE
  export HF_HOME
  export TRITON_CACHE_DIR
  export WANDB_DIR
  export WANDB_CACHE_DIR
  export WANDB_CONFIG_DIR
  export WANDB_DATA_DIR
  export TMPDIR
  export MPLCONFIGDIR
  export TORCH_HOME
  export XDG_CACHE_HOME
  export OUT_DIR
  export LOG_ROOT
}

sycophancy_bias_nearest_existing_path() {
  local path="$1"
  while [[ ! -e "$path" && "$path" != "/" ]]; do
    path="$(dirname "$path")"
  done
  printf '%s\n' "$path"
}

sycophancy_bias_print_storage_env() {
  local tag="${1:-[storage]}"
  local path existing
  printf '%s storage_root=%s\n' "$tag" "${SYCOPHANCY_STORAGE_ROOT:-}"
  printf '%s out_dir=%s\n' "$tag" "${OUT_DIR:-}"
  printf '%s log_root=%s\n' "$tag" "${LOG_ROOT:-}"
  printf '%s hf_hub_cache=%s\n' "$tag" "${HF_HUB_CACHE:-}"
  printf '%s hf_datasets_cache=%s\n' "$tag" "${HF_DATASETS_CACHE:-}"
  printf '%s hf_home=%s\n' "$tag" "${HF_HOME:-}"
  printf '%s triton_cache_dir=%s\n' "$tag" "${TRITON_CACHE_DIR:-}"
  printf '%s wandb_dir=%s\n' "$tag" "${WANDB_DIR:-}"
  printf '%s tmpdir=%s\n' "$tag" "${TMPDIR:-}"
  printf '%s torch_home=%s\n' "$tag" "${TORCH_HOME:-}"
  for path in "${OUT_DIR:-}" "${LOG_ROOT:-}" "${HF_HUB_CACHE:-}" "${HF_DATASETS_CACHE:-}" "${TMPDIR:-}"; do
    if [[ -n "$path" ]]; then
      existing="$(sycophancy_bias_nearest_existing_path "$path")"
      printf '%s df_h_for=%s nearest_existing=%s\n' "$tag" "$path" "$existing"
      df -h "$existing" || true
    fi
  done
}
