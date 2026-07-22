from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .data import build_intervention_pairs, load_source_bundle
from .experiment import (
    DEFAULT_ALPHAS,
    EXPERIMENT_CONDITIONS,
    aggregate_intervention_results,
    fit_restoration_directions,
    run_intervention_layer,
    select_validation_dose,
    select_validation_layers,
)


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def _optional_positive(value: int) -> Optional[int]:
    return int(value) if int(value) > 0 else None


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--device-map-auto", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--torch-dtype", default=None, help="auto, float16, bfloat16, or float32")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--probe-name", default="probe_bias_random_all")
    parser.add_argument(
        "--conditions",
        default=",".join(EXPERIMENT_CONDITIONS),
        help="Comma-separated saved prompt conditions to pair exactly.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only MeanDiff steering and paired pre-answer activation patching for the "
            "random_all sycophancy probe experiments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-source", help="Validate artifacts and pairing without loading a model.")
    _add_source_arguments(validate)
    validate.add_argument("--output-dir", type=Path, required=True)

    fit = subparsers.add_parser("fit-directions", help="Fit train-only pre-answer restoration MeanDiff directions.")
    _add_source_arguments(fit)
    _add_runtime_arguments(fit)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--fit-split", default="train")
    fit.add_argument("--layers", default="", help="Comma-separated residual layers; empty means all.")
    fit.add_argument("--max-questions", type=int, default=0, help="0 means all paired questions.")
    fit.add_argument("--seed", type=int, default=5)
    fit.add_argument("--progress-every", type=int, default=50)
    fit.add_argument("--n-control-directions", type=int, default=20)

    run = subparsers.add_parser("run-layer", help="Run one validation/test layer shard.")
    _add_source_arguments(run)
    _add_runtime_arguments(run)
    run.add_argument("--directions-path", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--layer", type=int, required=True)
    run.add_argument("--split", default="val")
    run.add_argument("--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS))
    run.add_argument("--max-questions", type=int, default=0)
    run.add_argument("--random-control-seeds", default="0,1,2,3,4")
    run.add_argument(
        "--include-transported-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exploratory only: transport the post-answer random_all vector to the pre-answer site.",
    )
    run.add_argument("--min-baseline-replay-agreement", type=float, default=0.98)
    run.add_argument("--max-baseline-probability-p99-error", type=float, default=0.01)
    run.add_argument("--max-batch-size", type=int, default=4)
    run.add_argument(
        "--protocol",
        choices=("patch-localize", "dose-tune"),
        default="patch-localize",
    )
    run.add_argument("--progress-every", type=int, default=25)

    dose = subparsers.add_parser(
        "run-dose-candidate",
        help="Run the full validation dose grid at one frozen patch-layer candidate.",
    )
    _add_source_arguments(dose)
    _add_runtime_arguments(dose)
    dose.add_argument("--directions-path", type=Path, required=True)
    dose.add_argument("--layer-selection-json", type=Path, required=True)
    dose.add_argument("--candidate-index", type=int, required=True)
    dose.add_argument("--output-root", type=Path, required=True)
    dose.add_argument("--split", default="val")
    dose.add_argument("--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS))
    dose.add_argument("--max-questions", type=int, default=0)
    dose.add_argument("--random-control-seeds", default="0,1,2,3,4")
    dose.add_argument("--include-transported-probe", action=argparse.BooleanOptionalAction, default=True)
    dose.add_argument("--min-baseline-replay-agreement", type=float, default=0.98)
    dose.add_argument("--max-baseline-probability-p99-error", type=float, default=0.01)
    dose.add_argument("--max-batch-size", type=int, default=4)
    dose.add_argument("--progress-every", type=int, default=25)

    selected = subparsers.add_parser(
        "run-selected",
        help="Run the frozen validation-selected layer/dose on a held-out split.",
    )
    _add_source_arguments(selected)
    _add_runtime_arguments(selected)
    selected.add_argument("--directions-path", type=Path, required=True)
    selected.add_argument("--selection-json", type=Path, required=True)
    selected.add_argument("--output-root", type=Path, required=True)
    selected.add_argument("--split", default="test")
    selected.add_argument(
        "--random-control-seeds",
        default=",".join(str(value) for value in range(20)),
    )
    selected.add_argument("--include-transported-probe", action=argparse.BooleanOptionalAction, default=True)
    selected.add_argument("--min-baseline-replay-agreement", type=float, default=0.98)
    selected.add_argument("--max-baseline-probability-p99-error", type=float, default=0.01)
    selected.add_argument("--max-batch-size", type=int, default=4)
    selected.add_argument("--progress-every", type=int, default=25)

    choose_layers = subparsers.add_parser(
        "select-layers", help="Freeze a small patch-localized validation layer window."
    )
    choose_layers.add_argument("--output-root", type=Path, required=True)
    choose_layers.add_argument("--split", default="val")
    choose_layers.add_argument("--top-k", type=int, default=3)

    choose_dose = subparsers.add_parser(
        "select-dose", help="Freeze layer and dose from candidate-layer validation DiDs."
    )
    choose_dose.add_argument("--output-root", type=Path, required=True)
    choose_dose.add_argument("--split", default="val")
    choose_dose.add_argument("--max-neutral-accuracy-cost", type=float, default=0.02)
    choose_dose.add_argument("--max-correct-suggestion-accuracy-cost", type=float, default=0.02)
    choose_dose.add_argument("--max-correct-suggestion-probability-cost", type=float, default=0.02)
    choose_dose.add_argument("--max-genuine-agreement-relative-margin-cost", type=float, default=0.10)
    choose_dose.add_argument("--min-dose-response-spearman", type=float, default=0.70)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate item shards with paired bootstrap intervals.")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--split", default="", help="Optional split filter.")
    aggregate.add_argument("--n-bootstrap", type=int, default=2000)
    aggregate.add_argument("--seed", type=int, default=5)
    return parser


def _conditions(args: argparse.Namespace) -> list[str]:
    values = _csv_strings(args.conditions)
    if "neutral" not in values or "incorrect_suggestion_strong" not in values:
        raise ValueError("Conditions must include neutral and incorrect_suggestion_strong.")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-source":
        conditions = _conditions(args)
        source = load_source_bundle(
            args.source_run_dir,
            probe_name=args.probe_name,
            record_conditions=conditions,
        )
        pairs, coverage = build_intervention_pairs(
            source.records,
            probe_scores=source.probe_scores,
            required_conditions=conditions,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.output_dir / "pair_coverage.csv", index=False)
        summary = {
            "source_run_dir": str(source.run_dir),
            "model_name": source.model_name,
            "dataset_name": source.dataset_name,
            "chosen_probe_layer": source.chosen_layer,
            "conditions": conditions,
            "n_pairs": len(pairs),
            "pairs_by_split": {
                str(split): int(count)
                for split, count in coverage[coverage["included"]].groupby("split").size().items()
            },
        }
        (args.output_dir / "source_validation.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "fit-directions":
        artifact = fit_restoration_directions(
            source_run_dir=args.source_run_dir,
            output_dir=args.output_dir,
            fit_split=args.fit_split,
            layers=_csv_ints(args.layers) or None,
            conditions=_conditions(args),
            max_questions=_optional_positive(args.max_questions),
            seed=args.seed,
            probe_name=args.probe_name,
            device=args.device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=args.hf_cache_dir,
            torch_dtype=args.torch_dtype,
            progress_every=args.progress_every,
            n_control_directions=args.n_control_directions,
        )
        print(f"directions={artifact.path}")
        print(f"manifest={artifact.metadata_path}")
        return 0

    if args.command in {"run-layer", "run-dose-candidate", "run-selected"}:
        if args.command == "run-selected":
            selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
            layer = int(selection["selected_layer"])
            alphas = [float(value) for value in selection["test_alphas"]]
            protocol = "confirm"
            max_questions = None
            selection_path = args.selection_json
        elif args.command == "run-dose-candidate":
            layer_selection = json.loads(args.layer_selection_json.read_text(encoding="utf-8"))
            candidate_layers = [int(value) for value in layer_selection["candidate_layers"]]
            if args.candidate_index < 0 or args.candidate_index >= len(candidate_layers):
                raise IndexError(
                    f"candidate-index={args.candidate_index} outside 0..{len(candidate_layers) - 1}."
                )
            layer = candidate_layers[args.candidate_index]
            alphas = _csv_floats(args.alphas)
            protocol = "dose-tune"
            max_questions = _optional_positive(args.max_questions)
            selection_path = None
        else:
            layer = int(args.layer)
            alphas = _csv_floats(args.alphas)
            protocol = args.protocol
            max_questions = _optional_positive(args.max_questions)
            selection_path = None
        path = run_intervention_layer(
            source_run_dir=args.source_run_dir,
            directions_path=args.directions_path,
            output_root=args.output_root,
            layer=layer,
            split=args.split,
            conditions=_conditions(args),
            alphas=alphas,
            max_questions=max_questions,
            random_control_seeds=_csv_ints(args.random_control_seeds),
            probe_name=args.probe_name,
            include_transported_probe=args.include_transported_probe,
            min_baseline_replay_agreement=args.min_baseline_replay_agreement,
            max_baseline_probability_p99_error=args.max_baseline_probability_p99_error,
            max_batch_size=args.max_batch_size,
            device=args.device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=args.hf_cache_dir,
            torch_dtype=args.torch_dtype,
            protocol=protocol,
            selection_path=selection_path,
            progress_every=args.progress_every,
        )
        print(f"item_results={path}")
        return 0

    if args.command == "select-layers":
        selection = select_validation_layers(
            output_root=args.output_root,
            split=args.split,
            top_k=args.top_k,
        )
        print(json.dumps(selection, indent=2, default=lambda value: value.item()))
        return 0

    if args.command == "select-dose":
        selection = select_validation_dose(
            output_root=args.output_root,
            split=args.split,
            max_neutral_accuracy_cost=args.max_neutral_accuracy_cost,
            max_correct_suggestion_accuracy_cost=args.max_correct_suggestion_accuracy_cost,
            max_correct_suggestion_probability_cost=(
                args.max_correct_suggestion_probability_cost
            ),
            max_genuine_agreement_relative_margin_cost=(
                args.max_genuine_agreement_relative_margin_cost
            ),
            min_dose_response_spearman=args.min_dose_response_spearman,
        )
        print(json.dumps(selection, indent=2, default=lambda value: value.item()))
        return 0

    if args.command == "aggregate":
        paths = aggregate_intervention_results(
            output_root=args.output_root,
            split=args.split or None,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        for name, path in paths.items():
            print(f"{name}={path}")
        return 0

    raise AssertionError(f"Unhandled command {args.command!r}")


__all__ = ["build_parser", "main"]
