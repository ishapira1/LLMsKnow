#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llmssycoph.pruning.global_selection import FeasibilityThresholds, select_global_configuration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a feasible paper-faithful global sycophancy-pruning configuration."
    )
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--calibration_seed", type=int, default=5)
    parser.add_argument("--minimum_wrong_uplift_reduction", type=float, default=0.30)
    parser.add_argument("--minimum_biased_correct_probability_gain", type=float, default=0.02)
    parser.add_argument("--maximum_neutral_accuracy_drop", type=float, default=0.02)
    parser.add_argument("--maximum_neutral_correct_probability_drop", type=float, default=0.02)
    parser.add_argument("--maximum_correction_accuracy_drop", type=float, default=0.02)
    parser.add_argument("--maximum_agreement_accuracy_drop", type=float, default=0.02)
    parser.add_argument("--maximum_preservation_loss_increase", type=float, default=0.05)
    parser.add_argument("--maximum_wikitext_perplexity_increase", type=float, default=0.05)
    parser.add_argument("--maximum_other_wrong_invalid_increase", type=float, default=0.02)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary_path = Path(args.summary_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = FeasibilityThresholds(
        minimum_wrong_uplift_reduction=args.minimum_wrong_uplift_reduction,
        minimum_biased_correct_probability_gain=args.minimum_biased_correct_probability_gain,
        maximum_neutral_accuracy_drop=args.maximum_neutral_accuracy_drop,
        maximum_neutral_correct_probability_drop=args.maximum_neutral_correct_probability_drop,
        maximum_correction_accuracy_drop=args.maximum_correction_accuracy_drop,
        maximum_agreement_accuracy_drop=args.maximum_agreement_accuracy_drop,
        maximum_preservation_loss_increase=args.maximum_preservation_loss_increase,
        maximum_wikitext_perplexity_increase=args.maximum_wikitext_perplexity_increase,
        maximum_other_wrong_invalid_increase=args.maximum_other_wrong_invalid_increase,
    )
    result, audit = select_global_configuration(
        pd.read_csv(summary_path),
        thresholds=thresholds,
        split=args.split,
        calibration_seed=args.calibration_seed,
    )
    audit.to_csv(output_dir / "selection_audit.csv", index=False)
    payload = {
        **result.to_dict(),
        "summary_csv": str(summary_path),
        "split": args.split,
        "calibration_seed": args.calibration_seed,
        "thresholds": thresholds.__dict__,
    }
    (output_dir / "selected_configuration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
