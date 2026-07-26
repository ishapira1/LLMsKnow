#!/usr/bin/env bash

set -euo pipefail

SIGNAL_BUNDLE_NAME="activation_steering_signal_sharded_20260726"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/n/home12/ishapira/LLMsKnow}}"
export REPO_DIR
export ACTIVATION_STEERING_CONFIG="${ACTIVATION_STEERING_CONFIG:-$REPO_DIR/configs/experiments/activation_steering_signal_20260726.json}"
export QUESTION_MANIFEST="${QUESTION_MANIFEST:-$REPO_DIR/configs/experiments/activation_steering_signal_300_20260726.jsonl}"
export ACTIVATION_STEERING_LOG_ROOT="${ACTIVATION_STEERING_LOG_ROOT:-$REPO_DIR/jobs/sycophancy_bias_probe/logs/$SIGNAL_BUNDLE_NAME}"
if [[ -n "${SYCOPHANCY_STORAGE_ROOT_OVERRIDE:-}" ]]; then
  export INTERVENTION_BASE_ROOT="${INTERVENTION_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT_OVERRIDE/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_signal_20260726}"
fi

source "$REPO_DIR/jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh"

# ARC has complete sampling artifacts in its sampling-only run but no frozen
# random_all probe. Fit/screen/test use that canonical prompt source; the sparse
# fixed-probe stage remains CSQA-only.
SOURCE_RUN_NAMES_LLAMA[1]="arc_challenge_llama31_8b_allfamilies_sampling_20260618"
SOURCE_RUN_NAMES_QWEN[1]="arc_challenge_qwen25_7b_allfamilies_sampling_20260618"

"$ENV_PYTHON" -c \
  'import json,sys; c=json.load(open(sys.argv[1])); assert c.get("study_scope")=="exploratory_benchmark_label_signal_v1_20260726"; assert c["splits"]["semantic_wrong_option_requires_human_approval"] is False' \
  "$CONFIG_PATH"
