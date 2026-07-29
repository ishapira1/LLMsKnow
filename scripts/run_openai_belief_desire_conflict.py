#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llmssycoph.addressee_indexing import DEFAULT_USER_SPEND_LIMIT_USD, read_json  # noqa: E402
from llmssycoph.belief_desire_conflict import (  # noqa: E402
    TARGET_QUESTIONS_PER_DATASET,
    ExperimentPaths,
    analyze_experiment,
    audit_completion,
    prepare_experiment,
    quick_rates,
    run_live,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_api"
    / "belief_desire_conflict_gpt54nano_20260729"
)
DEFAULT_PRIOR_ROOT = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_api"
    / "reliability_and_motive_gpt54nano_20260729"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _paths(args: argparse.Namespace) -> ExperimentPaths:
    return ExperimentPaths(resolve_path(args.output_root))


def command_prepare(args: argparse.Namespace) -> None:
    result = prepare_experiment(
        paths=_paths(args),
        prior_root=resolve_path(args.prior_root),
        target=int(args.target),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_count(args: argparse.Namespace) -> None:
    print(json.dumps(read_json(_paths(args).request_counts), indent=2, sort_keys=True))


def command_estimate(args: argparse.Namespace) -> None:
    print(json.dumps(read_json(_paths(args).cost_estimate), indent=2, sort_keys=True))


def command_run_live(args: argparse.Namespace) -> None:
    result = run_live(
        paths=_paths(args),
        repo_root=REPO_ROOT,
        confirm_spend=bool(args.confirm_spend),
        max_cost_usd=float(args.max_cost_usd),
        concurrency=int(args.concurrency),
        timeout_seconds=float(args.timeout_seconds),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_quick_rates(args: argparse.Namespace) -> None:
    print(json.dumps(quick_rates(_paths(args)), indent=2, sort_keys=True))


def command_analyze(args: argparse.Namespace) -> None:
    result = analyze_experiment(
        paths=_paths(args),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_audit(args: argparse.Namespace) -> None:
    print(json.dumps(audit_completion(_paths(args)), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the GPT-5.4-nano belief-X versus desired-Y conflict experiment."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=5)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--prior-root", default=str(DEFAULT_PRIOR_ROOT))
    prepare.add_argument("--target", type=int, default=TARGET_QUESTIONS_PER_DATASET)
    prepare.set_defaults(func=command_prepare)
    for name, func in (
        ("count", command_count),
        ("estimate", command_estimate),
        ("quick-rates", command_quick_rates),
        ("audit", command_audit),
    ):
        command = subparsers.add_parser(name)
        command.set_defaults(func=func)
    live = subparsers.add_parser("run-live")
    live.add_argument("--confirm-spend", action="store_true")
    live.add_argument("--max-cost-usd", type=float, default=DEFAULT_USER_SPEND_LIMIT_USD)
    live.add_argument("--concurrency", type=int, default=96)
    live.add_argument("--timeout-seconds", type=float, default=90.0)
    live.set_defaults(func=command_run_live)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--bootstrap-iterations", type=int, default=10_000)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
