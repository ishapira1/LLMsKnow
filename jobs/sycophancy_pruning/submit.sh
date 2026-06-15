#!/usr/bin/env bash
set -euo pipefail

target="${1:-smoke}"

case "$target" in
  preflight)
    shift || true
    exec sbatch jobs/sycophancy_pruning/preflight_qwen25_two_dataset.sbatch "$@"
    ;;
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
  chain)
    shift || true
    preflight_id="$(sbatch --parsable jobs/sycophancy_pruning/preflight_qwen25_two_dataset.sbatch "$@")"
    printf '%s\n' "Submitted preflight: $preflight_id"
    smoke_id="$(sbatch --parsable --dependency="afterok:$preflight_id" jobs/sycophancy_pruning/smoke_qwen25_two_dataset.sbatch "$@")"
    printf '%s\n' "Submitted smoke after preflight: $smoke_id"
    pilot_id="$(sbatch --parsable --dependency="afterok:$smoke_id" jobs/sycophancy_pruning/pilot_qwen25_two_dataset.sbatch "$@")"
    printf '%s\n' "Submitted pilot after smoke: $pilot_id"
    full_id="$(sbatch --parsable --dependency="afterok:$pilot_id" jobs/sycophancy_pruning/qwen25_two_dataset.sbatch "$@")"
    printf '%s\n' "Submitted full after pilot: $full_id"
    ;;
  *)
    printf '%s\n' "Usage: $0 {preflight|smoke|pilot|full|arc|commonsense_qa|csqa|all-full|chain} [extra run args]" >&2
    exit 2
    ;;
esac
