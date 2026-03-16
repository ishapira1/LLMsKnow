from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from .constants import (
    ALL_BIAS_TYPES,
    ALL_MC_MODES,
    DEFAULT_AYS_MC_DATASETS,
    GRADING_SPEC_VERSION,
    MC_MODE_STRICT,
    MC_MODE_WITH_RATIONALE,
    PROMPT_SPEC_VERSION,
    SUPPORTED_BENCHMARK_SOURCES,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run sycophancy x/x' sampling + per-type probe training/evaluation."
    )
    ap.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--device_map_auto", action="store_true")

    ap.add_argument(
        "--benchmark_source",
        type=str,
        default="answer_json",
        choices=list(SUPPORTED_BENCHMARK_SOURCES),
        help="Which benchmark construction to run. 'answer_json' uses answer.jsonl as-is; 'ays_mc_single_turn' derives single-turn bias prompts from the multiple-choice AYS source.",
    )
    ap.add_argument("--data_dir", type=str, default="data/sycophancy-eval")
    ap.add_argument("--sycophancy_repo", type=str, default="meg-tong/sycophancy-eval")
    ap.add_argument("--force_download_sycophancy", action="store_true")
    ap.add_argument(
        "--input_jsonl",
        type=str,
        default="answer.jsonl",
        choices=["answer.jsonl", "are_you_sure.jsonl"],
    )
    ap.add_argument(
        "--dataset_name",
        "--dataset_type",
        dest="dataset_name",
        type=str,
        default="all",
        help="Filter to a specific source dataset from base.dataset (for example truthful_qa). Use 'all' to keep every dataset.",
    )
    ap.add_argument(
        "--ays_mc_datasets",
        type=str,
        default=",".join(DEFAULT_AYS_MC_DATASETS),
        help="Comma-separated AYS source datasets to derive when --benchmark_source=ays_mc_single_turn.",
    )
    ap.add_argument(
        "--mc_mode",
        type=str,
        default=MC_MODE_STRICT,
        choices=list(ALL_MC_MODES),
        help="Multiple-choice prompting/grading mode. strict_mc is the canonical benchmark path; mc_with_rationale preserves an explicit answer line but allows rationale after it.",
    )

    ap.add_argument(
        "--bias_types",
        type=str,
        default="incorrect_suggestion,doubt_correct,suggest_correct",
        help=f"Comma-separated subset from: {','.join(ALL_BIAS_TYPES)}",
    )
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--max_questions", type=int, default=None)
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--smoke_questions", type=int, default=24)

    ap.add_argument("--n_draws", type=int, default=4)
    ap.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size for sampling draws per prompt. Falls back to sequential on OOM/runtime generation errors.",
    )
    ap.add_argument(
        "--sampling_checkpoint_every",
        type=int,
        default=200,
        help="Persist sampling checkpoint every N newly generated responses. Set 0 to disable periodic checkpoints.",
    )
    ap.add_argument(
        "--no_reuse_sampling_cache",
        action="store_true",
        help="Disable reuse of matching sampling checkpoints from earlier runs.",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum generated tokens. Defaults are mode-aware: answer_json keeps the legacy budget, strict_mc uses a short budget, and mc_with_rationale uses a longer budget.",
    )
    ap.add_argument(
        "--probe_feature_mode",
        type=str,
        default="response_raw_final_token",
        choices=["response_raw_final_token"],
        help="Probe on the final token of the full raw sampled completion.",
    )

    ap.add_argument("--probe_layer_min", type=int, default=1)
    ap.add_argument("--probe_layer_max", type=int, default=32)
    ap.add_argument(
        "--probe_val_frac",
        "--val_frac",
        dest="probe_val_frac",
        type=float,
        default=0.2,
        help="Fraction of non-test questions reserved for the question-level validation split used in probe layer selection.",
    )
    ap.add_argument("--probe_seed", type=int, default=0)
    ap.add_argument("--probe_selection_max_samples", type=int, default=2000)
    ap.add_argument("--probe_train_max_samples", type=int, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Optional .env file to load before resolving cache env vars. Use empty string to disable.",
    )
    ap.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="HF model/tokenizer cache dir. If unset, uses HF_HUB_CACHE or HUGGINGFACE_HUB_CACHE env vars.",
    )
    ap.add_argument("--out_dir", type=str, default="output/sycophancy_bias_probe")
    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()
    if args.max_new_tokens is None:
        if args.benchmark_source != "ays_mc_single_turn":
            args.max_new_tokens = 96
        elif args.mc_mode == MC_MODE_STRICT:
            args.max_new_tokens = 20
        elif args.mc_mode == MC_MODE_WITH_RATIONALE:
            args.max_new_tokens = 192
        else:
            args.max_new_tokens = 96
    args.prompt_spec_version = int(PROMPT_SPEC_VERSION)
    args.grading_spec_version = int(GRADING_SPEC_VERSION)
    return args


def resolve_bias_types(arg: str) -> List[str]:
    choices = [x.strip() for x in arg.split(",") if x.strip()]
    invalid = [x for x in choices if x not in ALL_BIAS_TYPES]
    if invalid:
        raise ValueError(f"Unknown bias types: {invalid}. Valid: {list(ALL_BIAS_TYPES)}")
    if not choices:
        raise ValueError("At least one bias type is required.")
    return choices


def resolve_csv_choices(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_hf_cache_dir(cli_cache_dir: Optional[str]) -> Optional[str]:
    if cli_cache_dir:
        return cli_cache_dir
    env_cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        return env_cache
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return str(Path(hf_home) / "hub")
    return None


def load_env_file(env_file: Optional[str]) -> None:
    if not env_file:
        return
    path = Path(env_file)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
