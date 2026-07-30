#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmssycoph import recipient_routing as experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_MANIFEST = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "openai_sycophancy_development_cohort_gpt54nano_v1.jsonl"
)
COHORT_SPEC = COHORT_MANIFEST.with_suffix(".json")
TERRA_ROOT = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_api"
    / "experiment_4_recipient_routing_gpt56terra_20260730"
)
NANO_ROOT = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_api"
    / "experiment_4_recipient_routing_gpt54nano_replication_20260730"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("prepare", "estimate", "run-live", "analyze", "audit", "all"),
    )
    parser.add_argument("--profile", choices=("terra", "nano"), default="terra")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args()
    profile = experiment.configure_profile(args.profile)
    output_root = args.output_root or (
        NANO_ROOT if args.profile == "nano" else TERRA_ROOT
    )
    max_cost_usd = (
        float(args.max_cost_usd)
        if args.max_cost_usd is not None
        else float(experiment.MAX_COST_USD)
    )
    paths = experiment.ExperimentPaths(output_root.resolve())
    result = None
    if args.mode in {"prepare", "all"}:
        result = experiment.prepare_experiment(
            paths=paths,
            cohort_manifest=COHORT_MANIFEST,
            cohort_spec=COHORT_SPEC,
        )
    if args.mode == "estimate":
        result = json.loads(paths.estimate.read_text(encoding="utf-8"))
    if args.mode in {"run-live", "all"}:
        result = experiment.run_live(
            paths=paths,
            repo_root=REPO_ROOT,
            confirm_spend=args.confirm_spend,
            max_cost_usd=max_cost_usd,
        )
    if args.mode in {"analyze", "all"}:
        result = experiment.analyze_experiment(paths=paths)
    if args.mode in {"audit", "all"}:
        result = experiment.audit_completion(paths=paths)
    print(json.dumps({"profile": profile, "result": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
