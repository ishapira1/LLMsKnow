from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_bootstrap_src_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a completed endorsed-option-grid backfill and compute claim-3 metrics "
            "from its candidate-level probe scores."
        ),
    )
    parser.add_argument("--run_dir", required=True, help="Absolute or repo-relative completed run directory.")
    parser.add_argument(
        "--probe_name",
        default="probe_no_bias",
        help="Chosen probe family to analyze. Defaults to probe_no_bias.",
    )
    parser.add_argument(
        "--sampling_subdir",
        default=None,
        help="Optional sampling backfill subdirectory. Defaults to the endorsed-option-grid convention.",
    )
    parser.add_argument(
        "--probe_subdir",
        default=None,
        help="Optional probe backfill subdirectory. Defaults to the endorsed-option-grid convention.",
    )
    parser.add_argument(
        "--output_subdir",
        default=None,
        help="Optional output subdirectory under the run dir. Defaults to analysis/claim3/<probe>__<template>.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Optional split filter. Repeat to include multiple splits.",
    )
    return parser


def main() -> None:
    from llmssycoph.analysis.claim3 import run_claim3_analysis

    args = build_parser().parse_args()
    result = run_claim3_analysis(
        args.run_dir,
        probe_name=args.probe_name,
        sampling_subdir=args.sampling_subdir,
        probe_subdir=args.probe_subdir,
        output_subdir=args.output_subdir,
        requested_splits=args.split,
    )
    summary = result["summary_payload"]

    print(f"output_dir={result['output_dir']}")
    if summary.get("overall_truth"):
        print(
            "overall_truth "
            f"mean_truth_gap={summary['overall_truth'].get('mean_truth_gap')} "
            f"mean_pairwise_k={summary['overall_truth'].get('mean_pairwise_k')} "
            f"probe_top1_correct_rate={summary['overall_truth'].get('probe_top1_correct_rate')}"
        )
    if summary.get("overall_leakage"):
        print(
            "overall_leakage "
            f"mean_endorsement_leakage={summary['overall_leakage'].get('mean_endorsement_leakage')} "
            f"mean_abs_endorsement_leakage={summary['overall_leakage'].get('mean_abs_endorsement_leakage')}"
        )
    if summary.get("wrong_candidate_leakage"):
        print(
            "wrong_candidate_leakage "
            f"mean_endorsement_leakage={summary['wrong_candidate_leakage'].get('mean_endorsement_leakage')} "
            f"mean_abs_endorsement_leakage={summary['wrong_candidate_leakage'].get('mean_abs_endorsement_leakage')}"
        )


if __name__ == "__main__":
    main()
