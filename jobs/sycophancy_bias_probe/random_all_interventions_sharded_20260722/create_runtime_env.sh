#!/usr/bin/env bash

set -euo pipefail

BUNDLE_NAME="random_all_interventions_sharded_20260722"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  printf '%s\n' "Missing .env in $ROOT_DIR" >&2
  exit 1
fi
set -a
source .env
set +a

if [[ -n "${SYCOPHANCY_STORAGE_ROOT_OVERRIDE:-}" ]]; then
  export SYCOPHANCY_STORAGE_ROOT="$SYCOPHANCY_STORAGE_ROOT_OVERRIDE"
  unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE HF_HOME
  unset TRITON_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
  unset TMPDIR MPLCONFIGDIR TORCH_HOME XDG_CACHE_HOME OUT_DIR LOG_ROOT
fi
source jobs/sycophancy_bias_probe/storage_common.sh
configure_sycophancy_bias_storage "$BUNDLE_NAME"

module load python/3.10.9-fasrc01
BASE_PYTHON="${BASE_PYTHON:-python3}"
RUNTIME_ENV_DIR="${RANDOM_ALL_INTERVENTION_ENV_DIR:-$SYCOPHANCY_STORAGE_ROOT/python_envs/llmsknow_py310_torch220_transformers4432}"
RUNTIME_PYTHON="$RUNTIME_ENV_DIR/bin/python"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SYCOPHANCY_STORAGE_ROOT/pip_cache}"

"$BASE_PYTHON" -c \
  'import sys; sys.exit("BASE_PYTHON must be >=3.10, found " + sys.version.split()[0]) if sys.version_info < (3,10) else None'

if [[ -d "$RUNTIME_ENV_DIR" && "${ALLOW_ENV_UPDATE:-0}" != "1" ]]; then
  printf '%s\n' \
    "Runtime environment already exists: $RUNTIME_ENV_DIR (set ALLOW_ENV_UPDATE=1 to update it)" >&2
  exit 1
fi

mkdir -p "$(dirname "$RUNTIME_ENV_DIR")" "$PIP_CACHE_DIR"
if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  "$BASE_PYTHON" -m venv "$RUNTIME_ENV_DIR"
fi

"$RUNTIME_PYTHON" -m pip install --upgrade pip setuptools wheel
"$RUNTIME_PYTHON" -m pip install --requirement "$ROOT_DIR/requirements.txt"
"$RUNTIME_PYTHON" -m pip install --upgrade "transformers==4.43.2"
"$RUNTIME_PYTHON" -m pip check
"$RUNTIME_PYTHON" \
  "$ROOT_DIR/jobs/sycophancy_bias_probe/$BUNDLE_NAME/runtime_contract.py"

printf '[runtime-env] ready=%s\n' "$RUNTIME_PYTHON"
printf '[runtime-env] next: export ENV_PYTHON=%q\n' "$RUNTIME_PYTHON"
