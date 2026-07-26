from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .controlled import PRIMARY_ALPHA_GRID, REQUIRED_CONDITIONS
from .controlled_runtime import (
    aggregate_controlled_results,
    fit_controlled_directions,
    inspect_controlled_examples,
    run_alpaca_guardrail,
    run_controlled_geometry,
    run_controlled_interventions,
    validate_controlled_sources,
)


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map-auto", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--torch-dtype", default="auto")


def _source_arguments(parser: argparse.ArgumentParser, *, repeat: bool) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        action="append" if repeat else "store",
    )
    parser.add_argument("--question-manifest", type=Path, required=True)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    _source_arguments(parser, repeat=False)
    _runtime_arguments(parser)
    parser.add_argument("--directions-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument(
        "--alphas",
        default=",".join(str(value) for value in PRIMARY_ALPHA_GRID),
    )
    parser.add_argument("--control-seeds", default=",".join(str(value) for value in range(10)))
    parser.add_argument("--learned-directions", default="wn,cn,wc,sw")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--score-fixed-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--generation-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--progress-every", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled prompt-only activation steering protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-source")
    _source_arguments(validate, repeat=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument(
        "--require-human-approval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    inspect = subparsers.add_parser("inspect-examples")
    _source_arguments(inspect, repeat=False)
    _runtime_arguments(inspect)
    inspect.add_argument("--output-dir", type=Path, required=True)
    inspect.add_argument("--layers", required=True)
    inspect.add_argument("--directions-path", type=Path, default=None)

    fit = subparsers.add_parser("fit-directions")
    _source_arguments(fit, repeat=True)
    _runtime_arguments(fit)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument(
        "--layers",
        default="",
        help="Comma-separated nonterminal residual layers; empty scans every nonterminal layer.",
    )
    fit.add_argument("--control-seeds", default=",".join(str(value) for value in range(10)))
    fit.add_argument("--progress-every", type=int, default=25)

    for command in ("screen-layers", "tiny-dry-run", "run-selected", "score-fixed-probe"):
        run = subparsers.add_parser(command)
        _run_arguments(run)
        if command == "score-fixed-probe":
            run.set_defaults(score_fixed_probe=True)

    geometry = subparsers.add_parser("run-geometry")
    _source_arguments(geometry, repeat=False)
    _runtime_arguments(geometry)
    geometry.add_argument("--directions-path", type=Path, required=True)
    geometry.add_argument("--output-dir", type=Path, required=True)
    geometry.add_argument("--split", choices=("val", "test"), default="test")
    geometry.add_argument("--layers", required=True)
    geometry.add_argument(
        "--permutation-seeds",
        default=",".join(str(value) for value in range(100)),
    )

    alpaca = subparsers.add_parser("run-alpaca-guardrail")
    alpaca.add_argument("--config", type=Path, required=True)
    alpaca.add_argument("--source-run-dir", type=Path, required=True)
    alpaca.add_argument("--alpaca-manifest", type=Path, required=True)
    alpaca.add_argument("--directions-path", type=Path, required=True)
    alpaca.add_argument("--output-dir", type=Path, required=True)
    alpaca.add_argument("--layer", type=int, required=True)
    alpaca.add_argument("--alphas", default="-128,-4,0,4,128")
    _runtime_arguments(alpaca)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input", type=Path, action="append", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--n-bootstrap", type=int, default=2000)
    aggregate.add_argument("--seed", type=int, default=5)
    aggregate.add_argument(
        "--enforce-cross-shard-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail when independently launched alpha-zero shards exceed the replay "
            "thresholds. Disable only for exploratory within-shard paired analyses; "
            "the discrepancy is still reported."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-source":
        output = validate_controlled_sources(
            config_path=args.config,
            source_run_dirs=args.source_run_dir,
            question_manifest_path=args.question_manifest,
            output_dir=args.output_dir,
            require_human_approval=args.require_human_approval,
        )
    elif args.command == "inspect-examples":
        output = inspect_controlled_examples(
            config_path=args.config,
            source_run_dir=args.source_run_dir,
            question_manifest_path=args.question_manifest,
            output_dir=args.output_dir,
            layers=_csv_ints(args.layers),
            directions_path=args.directions_path,
            device=args.device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=args.hf_cache_dir,
            torch_dtype=args.torch_dtype,
        )
    elif args.command == "fit-directions":
        output = fit_controlled_directions(
            config_path=args.config,
            source_run_dirs=args.source_run_dir,
            question_manifest_path=args.question_manifest,
            output_dir=args.output_dir,
            layers=_csv_ints(args.layers),
            control_seeds=_csv_ints(args.control_seeds),
            device=args.device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=args.hf_cache_dir,
            torch_dtype=args.torch_dtype,
            progress_every=args.progress_every,
        )
    elif args.command in {
        "screen-layers",
        "tiny-dry-run",
        "run-selected",
        "score-fixed-probe",
    }:
        output = run_controlled_interventions(
            stage=args.command.replace("-", "_"),
            config_path=args.config,
            source_run_dir=args.source_run_dir,
            question_manifest_path=args.question_manifest,
            directions_path=args.directions_path,
            output_dir=args.output_dir,
            split=args.split,
            layers=_csv_ints(args.layers),
            alphas=_csv_floats(args.alphas),
            control_seeds=_csv_ints(args.control_seeds),
            learned_directions=_csv_strings(args.learned_directions),
            max_batch_size=args.max_batch_size,
            score_fixed_probe=args.score_fixed_probe,
            generation_diagnostics=args.generation_diagnostics,
            device=args.device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=args.hf_cache_dir,
            torch_dtype=args.torch_dtype,
            progress_every=args.progress_every,
        )
    elif args.command == "run-geometry":
        output = run_controlled_geometry(
            config_path=args.config,
            source_run_dir=args.source_run_dir,
            question_manifest_path=args.question_manifest,
            directions_path=args.directions_path,
            output_dir=args.output_dir,
            split=args.split,
            layers=_csv_ints(args.layers),
            permutation_seeds=_csv_ints(args.permutation_seeds),
            device=args.device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=args.hf_cache_dir,
            torch_dtype=args.torch_dtype,
        )
    elif args.command == "run-alpaca-guardrail":
        output = run_alpaca_guardrail(
            config_path=args.config,
            source_run_dir=args.source_run_dir,
            alpaca_manifest_path=args.alpaca_manifest,
            directions_path=args.directions_path,
            output_dir=args.output_dir,
            layer=args.layer,
            alphas=_csv_floats(args.alphas),
            device=args.device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=args.hf_cache_dir,
            torch_dtype=args.torch_dtype,
        )
    elif args.command == "aggregate":
        output = aggregate_controlled_results(
            input_paths=args.input,
            output_dir=args.output_dir,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            enforce_cross_shard_replay=args.enforce_cross_shard_replay,
        )
    else:  # pragma: no cover - argparse enforces commands
        raise AssertionError(args.command)
    print(json.dumps({"output": str(output), "required_conditions": REQUIRED_CONDITIONS}, indent=2))
    return 0


__all__ = ["build_parser", "main"]
