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

from llmssycoph.addressee_indexing import (  # noqa: E402
    DEFAULT_USER_SPEND_LIMIT_USD,
    TARGET_QUESTIONS_PER_DATASET,
    ExperimentPaths,
    analyze_experiment,
    audit_completion,
    prepare_experiment,
    read_json,
    run_live,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_api"
    / "addressee_indexed_endorsement_gpt54nano_20260727"
)
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "sycophancy-eval"
DEFAULT_HISTORICAL_RESULTS = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "openai_batch"
    / "anti_sycophancy_request_gpt54nano_incorrect_live_20260624"
    / "sampling_records.jsonl"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _paths(args: argparse.Namespace) -> ExperimentPaths:
    return ExperimentPaths(resolve_path(args.output_root))


def command_prepare(args: argparse.Namespace) -> None:
    result = prepare_experiment(
        paths=_paths(args),
        data_dir=resolve_path(args.data_dir),
        historical_path=resolve_path(args.historical_results),
        target=int(args.target),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_count(args: argparse.Namespace) -> None:
    path = _paths(args).request_counts
    if not path.exists():
        raise FileNotFoundError("Run prepare before count")
    print(json.dumps(read_json(path), indent=2, sort_keys=True))


def command_estimate(args: argparse.Namespace) -> None:
    path = _paths(args).cost_estimate
    if not path.exists():
        raise FileNotFoundError("Run prepare before estimate")
    print(json.dumps(read_json(path), indent=2, sort_keys=True))


def command_run_live(args: argparse.Namespace) -> None:
    result = run_live(
        paths=_paths(args),
        repo_root=REPO_ROOT,
        confirm_spend=bool(args.confirm_spend),
        max_cost_usd=float(args.max_cost_usd),
        concurrency=int(args.concurrency),
        timeout_seconds=float(args.timeout_seconds),
        screen_batch_size=int(args.screen_batch_size),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_analyze(args: argparse.Namespace) -> None:
    result = analyze_experiment(
        paths=_paths(args),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_audit(args: argparse.Namespace) -> None:
    print(json.dumps(audit_completion(_paths(args)), indent=2, sort_keys=True))


def command_all(args: argparse.Namespace) -> None:
    command_prepare(args)
    command_run_live(args)
    command_analyze(args)
    command_audit(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the GPT-5.4-nano addressee-indexed endorsement experiment. "
            "Preparing, counting, estimating, and analyzing are offline. "
            "Only run-live/all can submit paid requests."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=5)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Dry-run: validate reuse and write manifests")
    prepare.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    prepare.add_argument("--historical-results", default=str(DEFAULT_HISTORICAL_RESULTS))
    prepare.add_argument("--target", type=int, default=TARGET_QUESTIONS_PER_DATASET)
    prepare.set_defaults(func=command_prepare)

    count = subparsers.add_parser("count", help="Print the prepared request counts")
    count.set_defaults(func=command_count)

    estimate = subparsers.add_parser("estimate", help="Print the conservative cost estimate")
    estimate.set_defaults(func=command_estimate)

    live = subparsers.add_parser("run-live", help="Submit the staged paid API requests")
    live.add_argument("--confirm-spend", action="store_true")
    live.add_argument("--max-cost-usd", type=float, default=DEFAULT_USER_SPEND_LIMIT_USD)
    live.add_argument("--concurrency", type=int, default=64)
    live.add_argument("--timeout-seconds", type=float, default=90.0)
    live.add_argument("--screen-batch-size", type=int, default=64)
    live.set_defaults(func=command_run_live)

    analyze = subparsers.add_parser("analyze", help="Analyze completed six-condition pairs")
    analyze.add_argument("--bootstrap-iterations", type=int, default=10_000)
    analyze.set_defaults(func=command_analyze)

    audit = subparsers.add_parser("audit", help="Verify all expected outputs and safety invariants")
    audit.set_defaults(func=command_audit)

    all_command = subparsers.add_parser(
        "all",
        help="Prepare, run live, analyze, and audit in one resumable command",
    )
    all_command.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    all_command.add_argument("--historical-results", default=str(DEFAULT_HISTORICAL_RESULTS))
    all_command.add_argument("--target", type=int, default=TARGET_QUESTIONS_PER_DATASET)
    all_command.add_argument("--confirm-spend", action="store_true")
    all_command.add_argument("--max-cost-usd", type=float, default=DEFAULT_USER_SPEND_LIMIT_USD)
    all_command.add_argument("--concurrency", type=int, default=64)
    all_command.add_argument("--timeout-seconds", type=float, default=90.0)
    all_command.add_argument("--screen-batch-size", type=int, default=64)
    all_command.add_argument("--bootstrap-iterations", type=int, default=10_000)
    all_command.set_defaults(func=command_all)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
