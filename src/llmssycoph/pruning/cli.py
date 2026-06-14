from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from ..cli import load_env_file, resolve_device
from ..constants import MC_MODE_STRICT
from ..runtime import build_fresh_run_name


DEFAULT_SPARSITIES = "0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3"
DEFAULT_EVAL_FAMILIES = (
    "incorrect_suggestion,"
    "incorrect_suggestion_strong,"
    "suggest_random,"
    "doubt_correct,"
    "model_congruent_suggestion,"
    "incorrect_suggestion_rephrase_1,"
    "incorrect_suggestion_rephrase_2"
)


def _csv_values(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _csv_floats(value: str) -> List[float]:
    out = []
    for part in _csv_values(value):
        out.append(float(part))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run signed SNIP-style sycophancy weight-pruning experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--datasets", default="arc_challenge,commonsense_qa")
    parser.add_argument("--prune_family", default="incorrect_suggestion")
    parser.add_argument("--target_loss", default="choice_token", choices=["choice_token"])
    parser.add_argument("--benchmark_source", default="ays_mc_single_turn")
    parser.add_argument("--input_jsonl", default="are_you_sure.jsonl")
    parser.add_argument("--data_dir", default="data/sycophancy-eval")
    parser.add_argument("--sycophancy_repo", default="meg-tong/sycophancy-eval")
    parser.add_argument("--instruction_policy", default="answer_only")
    parser.add_argument("--mc_mode", default=MC_MODE_STRICT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--env_file", default=".env")
    parser.add_argument("--out_dir", default="results/sycophancy_pruning")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--fresh_run", action="store_true")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=5)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--max_questions_per_dataset", type=int, default=None)
    parser.add_argument("--max_calibration_records", type=int, default=None)
    parser.add_argument("--max_preservation_records", type=int, default=None)
    parser.add_argument("--max_eval_records", type=int, default=None)
    parser.add_argument("--wrong_control_min_examples", type=int, default=50)
    parser.add_argument("--sparsities", default=DEFAULT_SPARSITIES)
    parser.add_argument("--preserve_exclude_fraction", type=float, default=0.01)
    parser.add_argument("--syc_reduction_target", type=float, default=0.30)
    parser.add_argument("--preservation_loss_budget", type=float, default=0.10)
    parser.add_argument("--neutral_accuracy_drop_budget", type=float, default=0.05)
    parser.add_argument("--eval_families", default=DEFAULT_EVAL_FAMILIES)
    parser.add_argument("--score_batch_size", type=int, default=1)
    parser.add_argument("--save_all_sweep_masks", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.datasets = _csv_values(args.datasets)
    args.sparsities = _csv_floats(args.sparsities)
    args.eval_families = _csv_values(args.eval_families)
    args.bias_types = [
        "incorrect_suggestion",
        "incorrect_suggestion_strong",
        "suggest_correct",
        "suggest_correct_strong",
        "suggest_random",
        "doubt_correct",
    ]
    args.ays_mc_datasets = list(args.datasets)
    if args.fresh_run:
        args.run_name = build_fresh_run_name(args.run_name)
    load_env_file(args.env_file)
    args.resolved_device = resolve_device(args.device)
    args.model_backend = "huggingface"
    return args


__all__ = ["DEFAULT_EVAL_FAMILIES", "DEFAULT_SPARSITIES", "build_parser", "parse_args"]
