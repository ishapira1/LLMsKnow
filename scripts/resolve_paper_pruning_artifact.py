#!/usr/bin/env python3
from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve the identity-keyed in-repo weight-pruning score and mask paths."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prune-manifest", type=Path, required=True)
    parser.add_argument("--preserve-manifest", type=Path, required=True)
    parser.add_argument("--nsamples", type=int, required=True)
    parser.add_argument("--nsamples-preserve", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--score-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--loss-mode", choices=("completion_nll", "choice_token"), default="completion_nll")
    parser.add_argument("--attribution-variant", choices=("paper", "released_abs"), default="paper")
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--neg-prune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-first-top-q", action="store_true")
    parser.add_argument(
        "--control",
        choices=("none", "structure_matched", "alpaca_only", "random_magnitude"),
        default="none",
    )
    parser.add_argument("--match-bins", type=int, default=20)
    parser.add_argument("--no-abs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abs-prune", action="store_true")
    parser.add_argument("--abs-preserve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-score-length", type=int, default=4096)
    parser.add_argument("--require-existing", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--print-field",
        choices=("score_dir", "mask_dir", "indices_path", "metadata_path", "evaluation_path"),
    )
    return parser


def resolve(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    weight_pruning_dir = repo_root / "tools" / "weight_pruning"
    if not weight_pruning_dir.is_dir():
        raise FileNotFoundError(
            f"in-repo weight pruning source directory not found: {weight_pruning_dir}"
        )
    sys.path.insert(0, str(weight_pruning_dir))
    from paper_pruning import expected_score_dir, mask_output_dir, score_identity  # noqa: E402

    values = Namespace(
        model=args.model,
        revision=args.revision,
        tokenizer=None,
        tokenizer_revision=args.revision,
        prune_manifest=str(args.prune_manifest.expanduser().resolve()),
        preserve_manifest=str(args.preserve_manifest.expanduser().resolve()),
        nsamples=args.nsamples,
        nsamples_preserve=args.nsamples_preserve,
        seed=args.seed,
        score_format=args.score_format,
        loss_mode=args.loss_mode,
        attribution_variant=args.attribution_variant,
        no_abs=args.no_abs,
        abs_prune=args.abs_prune,
        abs_preserve=args.abs_preserve,
        layers=None,
        max_score_length=args.max_score_length,
        artifact_root=str(args.artifact_root.expanduser().resolve()),
        score_cache=None,
        p=args.p,
        q=args.q,
        neg_prune=args.neg_prune,
        freeze_first_top_q=args.freeze_first_top_q,
        control=args.control,
        match_bins=args.match_bins,
    )
    score_dir = expected_score_dir(values)
    mask_dir = mask_output_dir(values, score_dir)
    payload: dict[str, object] = {
        "score_dir": str(score_dir),
        "mask_dir": str(mask_dir),
        "indices_path": str(mask_dir / "indices.pt") if args.q != 0 else None,
        "metadata_path": str(mask_dir / "metadata.json"),
        "evaluation_path": str(mask_dir / "evaluation.json"),
        "score_identity": score_identity(values),
        "p": args.p,
        "q": args.q,
        "neg_prune": args.neg_prune,
        "freeze_first_top_q": args.freeze_first_top_q,
        "control": args.control,
        "match_bins": args.match_bins if args.control == "random_magnitude" else None,
    }
    if args.require_existing:
        required = [Path(str(payload["metadata_path"])), Path(str(payload["evaluation_path"]))]
        if args.q != 0:
            required.append(Path(str(payload["indices_path"])))
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"resolved pruning artifacts do not exist: {missing}")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    payload = resolve(args)
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_field:
        value = payload[args.print_field]
        if value is not None:
            print(value)
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
