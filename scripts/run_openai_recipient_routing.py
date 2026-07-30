#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmssycoph.recipient_routing import (
    ExperimentPaths,
    MAX_COST_USD,
    analyze_experiment,
    audit_completion,
    prepare_experiment,
    run_live,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_MANIFEST = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "openai_sycophancy_development_cohort_gpt54nano_v1.jsonl"
)
COHORT_SPEC = COHORT_MANIFEST.with_suffix(".json")
DEFAULT_ROOT = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_api"
    / "experiment_4_recipient_routing_gpt56terra_20260730"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("prepare", "estimate", "run-live", "analyze", "audit", "all"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=MAX_COST_USD)
    args = parser.parse_args()
    paths = ExperimentPaths(args.output_root.resolve())
    result = None
    if args.mode in {"prepare", "all"}:
        result = prepare_experiment(
            paths=paths,
            cohort_manifest=COHORT_MANIFEST,
            cohort_spec=COHORT_SPEC,
        )
    if args.mode == "estimate":
        result = json.loads(paths.estimate.read_text(encoding="utf-8"))
    if args.mode in {"run-live", "all"}:
        result = run_live(
            paths=paths,
            repo_root=REPO_ROOT,
            confirm_spend=args.confirm_spend,
            max_cost_usd=args.max_cost_usd,
        )
    if args.mode in {"analyze", "all"}:
        result = analyze_experiment(paths=paths)
    if args.mode in {"audit", "all"}:
        result = audit_completion(paths=paths)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
