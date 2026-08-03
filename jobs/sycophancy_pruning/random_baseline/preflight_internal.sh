#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qwen_root="/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/July_28_exp3"
qwen_slug="Qwen_Qwen2_5_7B_Instruct"
qwen_paraphrase="$RESULT_ROOT/inputs/qwen_final_paraphrase_manifest.jsonl"
"$PYTHON_BIN" "$BUNDLE_DIR/prepare_qwen_paraphrases.py" \
  --final-manifest "$qwen_root/inputs/final_manifest.jsonl" \
  --sampling-record "$qwen_root/sampling/$qwen_slug/arc_challenge/july_28_exp3_arc_qwen25_seed0/sampling/raw/sampling_records.jsonl" \
  --sampling-record "$qwen_root/sampling/$qwen_slug/commonsense_qa/july_28_exp3_csqa_qwen25_seed0/sampling/raw/sampling_records.jsonl" \
  --paraphrase-artifact "$REPO_DIR/data/ad_hoc/paraphrase_robustness_test_stems_v1" \
  --output "$qwen_paraphrase"
"$PYTHON_BIN" "$BUNDLE_DIR/random_baseline.py" preflight \
  --result-root "$RESULT_ROOT" --hf-cache "$HF_CACHE_DIR"
