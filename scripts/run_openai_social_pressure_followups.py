#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmssycoph.social_pressure_followups import (
    DEFAULT_MAX_COST_USD,
    EXPERIMENTS,
    ExperimentPaths,
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
DEFAULT_BASE = (
    REPO_ROOT / "results" / "sycophancy_bias_probe" / "openai_api"
)


def default_root(experiment: str) -> Path:
    return DEFAULT_BASE / f"{experiment}_gpt54nano_20260730"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "count", "estimate", "run-live", "analyze", "audit", "all"))
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    paths = ExperimentPaths((args.output_root or default_root(args.experiment)).resolve())

    result = None
    if args.mode in {"prepare", "all"}:
        result = prepare_experiment(
            experiment=args.experiment,
            paths=paths,
            cohort_manifest=COHORT_MANIFEST,
            cohort_spec=COHORT_SPEC,
        )
    if args.mode == "count":
        result = json.loads(paths.counts.read_text(encoding="utf-8"))
    elif args.mode == "estimate":
        result = json.loads(paths.estimate.read_text(encoding="utf-8"))
    if args.mode in {"run-live", "all"}:
        result = run_live(
            paths=paths,
            repo_root=REPO_ROOT,
            confirm_spend=args.confirm_spend,
            max_cost_usd=args.max_cost_usd,
            concurrency=args.concurrency,
        )
    if args.mode in {"analyze", "all"}:
        result = analyze_experiment(experiment=args.experiment, paths=paths)
    if args.mode in {"audit", "all"}:
        result = audit_completion(experiment=args.experiment, paths=paths)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
