#!/usr/bin/env python3
"""CLI for the frozen cross-model confidence--resistance experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmssycoph import confidence_resistance as experiment
from llmssycoph.analysis.confidence_resistance import run_analysis


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "sycophancy_bias_probe" / experiment.EXPERIMENT_NAME
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "command",
        choices=(
            "prepare",
            "score-hf-neutral",
            "score-hf-endorsed",
            "run-gpt-batch",
            "audit-gpt-pilot",
            "freeze-targets",
            "select-reserve",
            "audit-coverage",
            "analyze",
        ),
    )
    root.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    root.add_argument("--model-key", choices=tuple(experiment.MODEL_PROFILES))
    root.add_argument("--dataset", choices=experiment.DATASETS)
    root.add_argument("--hf-cache-dir")
    root.add_argument("--device", default="cuda")
    root.add_argument("--smoke-limit", type=int)
    root.add_argument("--gpt-stage", choices=("pilot-biased", "pilot-unbiased", "neutral", "endorsed"))
    root.add_argument(
        "--reserve",
        action="store_true",
        help="Operate on the pre-frozen reserve cohort rather than the initial 800 questions.",
    )
    root.add_argument(
        "--gpt-unbiased",
        action="store_true",
        help="Use unbiased top-20 scoring after a failed equal-bias pilot.",
    )
    root.add_argument("--confirm-spend", action="store_true")
    root.add_argument("--poll-seconds", type=int, default=20)
    root.add_argument("--scope", choices=("discovery", "confirmation"), default="discovery")
    root.add_argument("--unlock-confirmation", action="store_true")
    root.add_argument("--bootstrap-replicates", type=int, default=2000)
    root.add_argument("--permutation-replicates", type=int, default=2000)
    root.add_argument("--seed", type=int, default=experiment.SEED)
    root.add_argument("--analysis-output", type=Path)
    return root


def _require(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise experiment.ConfidenceResistanceError(
            "Missing required arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def _gpt_stage_paths(
    paths: experiment.ExperimentPaths, args: argparse.Namespace
) -> tuple[str, Path, Path, bool]:
    stage = str(args.gpt_stage)
    if stage == "pilot-biased":
        return stage, paths.pilot_manifest(True), paths.pilot_records(True), True
    if stage == "pilot-unbiased":
        return stage, paths.pilot_manifest(False), paths.pilot_records(False), False
    _require(args, "dataset")
    if stage == "neutral":
        pilot_passed = paths.pilot_audit.exists() and bool(
            experiment.read_json(paths.pilot_audit).get("passed")
        )
        if not pilot_passed and not args.gpt_unbiased:
            raise experiment.ConfidenceResistanceError(
                "A failed GPT pilot requires --gpt-unbiased for top-20 fallback scoring"
            )
        return (
            f"neutral-{args.dataset}"
            + ("-reserve" if args.reserve else "")
            + ("-unbiased" if args.gpt_unbiased else ""),
            (
                paths.reserve_neutral_manifest("gpt", args.dataset)
                if args.reserve
                else paths.neutral_manifest("gpt", args.dataset)
            ),
            (
                paths.reserve_neutral_records("gpt", args.dataset)
                if args.reserve
                else paths.neutral_records("gpt", args.dataset)
            ),
            not args.gpt_unbiased,
        )
    if stage == "endorsed":
        return (
            f"endorsed-{args.dataset}"
            + ("-reserve" if args.reserve else "")
            + ("-unbiased" if args.gpt_unbiased else ""),
            (
                paths.reserve_endorsed_manifest("gpt", args.dataset)
                if args.reserve
                else paths.endorsed_manifest("gpt", args.dataset)
            ),
            (
                paths.reserve_endorsed_records("gpt", args.dataset)
                if args.reserve
                else paths.endorsed_records("gpt", args.dataset)
            ),
            not args.gpt_unbiased,
        )
    raise AssertionError(stage)


def main() -> None:
    args = parser().parse_args()
    paths = experiment.ExperimentPaths(args.output_root.resolve())
    if args.command == "prepare":
        result = experiment.prepare_experiment(paths, REPO_ROOT)
    elif args.command == "score-hf-neutral":
        _require(args, "model_key", "dataset")
        result = experiment.score_hf_manifest(
            (
                paths.reserve_neutral_manifest(args.model_key, args.dataset)
                if args.reserve
                else paths.neutral_manifest(args.model_key, args.dataset)
            ),
            (
                paths.reserve_neutral_records(args.model_key, args.dataset)
                if args.reserve
                else paths.neutral_records(args.model_key, args.dataset)
            ),
            model_key=args.model_key,
            hf_cache_dir=args.hf_cache_dir,
            device=args.device,
            smoke_limit=args.smoke_limit,
        )
    elif args.command == "score-hf-endorsed":
        _require(args, "model_key", "dataset")
        result = experiment.score_hf_manifest(
            (
                paths.reserve_endorsed_manifest(args.model_key, args.dataset)
                if args.reserve
                else paths.endorsed_manifest(args.model_key, args.dataset)
            ),
            (
                paths.reserve_endorsed_records(args.model_key, args.dataset)
                if args.reserve
                else paths.endorsed_records(args.model_key, args.dataset)
            ),
            model_key=args.model_key,
            hf_cache_dir=args.hf_cache_dir,
            device=args.device,
            smoke_limit=args.smoke_limit,
        )
    elif args.command == "run-gpt-batch":
        _require(args, "gpt_stage")
        stage, manifest_path, output_path, biased = _gpt_stage_paths(paths, args)
        result = experiment.run_openai_batch(
            paths,
            stage=stage,
            manifest_path=manifest_path,
            output_path=output_path,
            biased=biased,
            confirm_spend=args.confirm_spend,
            poll_seconds=args.poll_seconds,
        )
    elif args.command == "audit-gpt-pilot":
        result = experiment.audit_gpt_pilot(paths)
    elif args.command == "freeze-targets":
        _require(args, "model_key")
        result = experiment.build_endorsement_manifests(
            paths, model_key=args.model_key, reserve=args.reserve
        )
    elif args.command == "select-reserve":
        result = experiment.select_reserve_extension(paths)
    elif args.command == "audit-coverage":
        result = experiment.audit_measurement_coverage(paths)
    elif args.command == "analyze":
        analysis_output = args.analysis_output or paths.root / "analysis" / args.scope
        result = run_analysis(
            paths,
            analysis_output,
            scope=args.scope,
            bootstrap_replicates=args.bootstrap_replicates,
            permutation_replicates=args.permutation_replicates,
            seed=args.seed,
            unlock_confirmation=args.unlock_confirmation,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
