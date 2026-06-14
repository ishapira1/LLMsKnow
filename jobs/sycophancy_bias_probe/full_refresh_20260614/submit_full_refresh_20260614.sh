#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p \
  jobs/sycophancy_bias_probe/logs/full_refresh_20260614/commonsense_qa \
  jobs/sycophancy_bias_probe/logs/full_refresh_20260614/arc_challenge

bash jobs/sycophancy_bias_probe/full_refresh_20260614/submit_commonsense_qa_full_refresh_20260614.sh
bash jobs/sycophancy_bias_probe/full_refresh_20260614/submit_arc_challenge_full_refresh_20260614.sh
