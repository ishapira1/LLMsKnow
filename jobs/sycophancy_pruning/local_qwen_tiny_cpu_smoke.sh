#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
export RUN_NAME="${RUN_NAME:-local_qwen25_05b_cpu_smoke}"

exec bash "$SCRIPT_DIR/local_tiny_cpu_smoke.sh" "$@"
