#!/usr/bin/env bash
set -euo pipefail

target="${1:-smoke}"

case "$target" in
  smoke)
    shift || true
    exec sbatch jobs/sycophancy_pruning/smoke_qwen25_two_dataset.sbatch "$@"
    ;;
  pilot)
    shift || true
    exec sbatch jobs/sycophancy_pruning/pilot_qwen25_two_dataset.sbatch "$@"
    ;;
  full)
    shift || true
    exec sbatch jobs/sycophancy_pruning/qwen25_two_dataset.sbatch "$@"
    ;;
  arc)
    shift || true
    exec sbatch jobs/sycophancy_pruning/full_arc_challenge_qwen25.sbatch "$@"
    ;;
  commonsense_qa|csqa)
    shift || true
    exec sbatch jobs/sycophancy_pruning/full_commonsense_qa_qwen25.sbatch "$@"
    ;;
  all-full)
    shift || true
    sbatch jobs/sycophancy_pruning/full_arc_challenge_qwen25.sbatch "$@"
    sbatch jobs/sycophancy_pruning/full_commonsense_qa_qwen25.sbatch "$@"
    sbatch jobs/sycophancy_pruning/qwen25_two_dataset.sbatch "$@"
    ;;
  *)
    printf '%s\n' "Usage: $0 {smoke|pilot|full|arc|commonsense_qa|csqa|all-full} [extra run_sycophancy_pruning.py args]" >&2
    exit 2
    ;;
esac
