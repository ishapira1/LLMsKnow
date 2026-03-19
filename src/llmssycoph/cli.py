from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

from .constants import (
    ALL_BIAS_TYPES,
    DEFAULT_AYS_MC_DATASETS,
    DEFAULT_INSTRUCTION_POLICY_NAME,
    GENERATION_SPEC_VERSION,
    GRADING_SPEC_VERSION,
    PROMPT_SPEC_VERSION,
    SUPPORTED_BENCHMARK_SOURCES,
    VISIBLE_INSTRUCTION_POLICY_NAMES,
)
from .data import canonical_instruction_policy_name, legacy_mc_mode_for_instruction_policy


def build_parser() -> argparse.ArgumentParser:
    class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    ap = argparse.ArgumentParser(
        description=(
            "Run the sycophancy pipeline: load a benchmark, build or reuse prompt variants, "
            "sample model responses, save artifacts, and train/score per-bias probes."
        ),
        epilog=(
            "Benchmark-source dependencies:\n"
            "  answer_json:\n"
            "    - requires --input_jsonl=answer.jsonl\n"
            "    - uses prompt variants already present in the dataset\n"
            "    - ignores --ays_mc_datasets and --instruction_policy for prompt construction\n"
            "  ays_mc_single_turn:\n"
            "    - requires --input_jsonl=are_you_sure.jsonl\n"
            "    - derives local neutral/bias prompt variants from canonical question data\n"
            "    - uses --ays_mc_datasets to choose source MC datasets\n"
            "    - uses --instruction_policy to choose the rendered answer-format instruction\n"
            "\n"
            "Bias-type behavior:\n"
            "  - --bias_types selects the non-neutral agreement-bias variants to keep or generate\n"
            "  - neutral is always included automatically\n"
            "  - question groups are kept only if every selected bias type is available\n"
            "\n"
            "Common examples:\n"
            "  python run_sycophancy_bias_probe.py --benchmark_source answer_json --input_jsonl answer.jsonl\n"
            "  python run_sycophancy_bias_probe.py --benchmark_source ays_mc_single_turn --input_jsonl are_you_sure.jsonl --ays_mc_datasets aqua_mc --instruction_policy answer_only"
        ),
        formatter_class=_HelpFormatter,
    )
    model_group = ap.add_argument_group("Model And Runtime")
    benchmark_group = ap.add_argument_group("Benchmark Construction")
    split_group = ap.add_argument_group("Question Splitting")
    sampling_group = ap.add_argument_group("Sampling")
    probe_group = ap.add_argument_group("Probe Training")
    io_group = ap.add_argument_group("Run I/O")

    model_group.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help=(
            "Model identifier passed to Hugging Face Transformers "
            "from_pretrained(...), usually a Hugging Face repo name like "
            "'mistralai/Mistral-7B-Instruct-v0.2'. A local model path also works."
        ),
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' prefers cuda, then mps, then cpu.",
    )
    model_group.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Let Transformers shard the model automatically across available devices.",
    )

    benchmark_group.add_argument(
        "--benchmark_source",
        type=str,
        default="answer_json",
        choices=list(SUPPORTED_BENCHMARK_SOURCES),
        help=(
            "How to construct prompt variants.\n"
            "'answer_json' reads dataset-provided prompt variants as-is.\n"
            "'ays_mc_single_turn' derives prompt variants locally from MC question rows."
        ),
    )
    benchmark_group.add_argument(
        "--data_dir",
        type=str,
        default="data/sycophancy-eval",
        help="Local cache directory for sycophancy benchmark files.",
    )
    benchmark_group.add_argument(
        "--sycophancy_repo",
        type=str,
        default="meg-tong/sycophancy-eval",
        help="Hugging Face dataset repo used when benchmark files are missing locally.",
    )
    benchmark_group.add_argument(
        "--force_download_sycophancy",
        action="store_true",
        help="Force a fresh download of the benchmark files instead of reusing local copies.",
    )
    benchmark_group.add_argument(
        "--input_jsonl",
        type=str,
        default="answer.jsonl",
        choices=["answer.jsonl", "are_you_sure.jsonl"],
        help=(
            "Which raw file to read.\n"
            "Use answer.jsonl with --benchmark_source=answer_json.\n"
            "Use are_you_sure.jsonl with --benchmark_source=ays_mc_single_turn."
        ),
    )
    benchmark_group.add_argument(
        "--dataset_name",
        "--dataset_type",
        dest="dataset_name",
        type=str,
        default="all",
        help=(
            "Filter to a specific source dataset from base.dataset, or use 'all'.\n"
            "Examples: truthful_qa, trivia_qa, aqua_mc.\n"
            "For ays_mc_single_turn this filter is applied after local prompt-variant materialization."
        ),
    )
    benchmark_group.add_argument(
        "--ays_mc_datasets",
        type=str,
        default=",".join(DEFAULT_AYS_MC_DATASETS),
        help=(
            "Comma-separated subset of AYS MC source datasets to derive when "
            "--benchmark_source=ays_mc_single_turn. Ignored for answer_json."
        ),
    )
    benchmark_group.add_argument(
        "--instruction_policy",
        "--mc_mode",
        dest="instruction_policy",
        type=str,
        default=DEFAULT_INSTRUCTION_POLICY_NAME,
        metavar="{" + ",".join(VISIBLE_INSTRUCTION_POLICY_NAMES) + "}",
        help=(
            "Instruction policy for locally generated prompts.\n"
            "answer_only requires the answer line and disallows reasoning.\n"
            "answer_with_reasoning requires the answer line and then allows brief reasoning.\n"
            "The legacy --mc_mode flag still works and accepts strict_mc / mc_with_rationale aliases.\n"
            "This mainly matters for --benchmark_source=ays_mc_single_turn."
        ),
    )

    benchmark_group.add_argument(
        "--bias_types",
        type=str,
        default="incorrect_suggestion,doubt_correct,suggest_correct",
        help=(
            "Comma-separated subset of non-neutral agreement-bias variants to keep or generate.\n"
            f"Valid values: {','.join(ALL_BIAS_TYPES)}.\n"
            "Neutral is always included automatically."
        ),
    )
    split_group.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Fraction of complete question groups assigned to the held-out test split.",
    )
    split_group.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed used for question-level train/val/test splitting and max-question subsampling.",
    )
    split_group.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional cap on the number of complete question groups kept before sampling.",
    )
    split_group.add_argument(
        "--smoke_test",
        action="store_true",
        help="Enable smoke-run behavior; if --max_questions is unset, it is replaced by --smoke_questions.",
    )
    split_group.add_argument(
        "--smoke_questions",
        type=int,
        default=24,
        help="Question cap used when --smoke_test is enabled and --max_questions is not set.",
    )

    sampling_group.add_argument(
        "--n_draws",
        type=int,
        default=4,
        help=(
            "Number of independent sampled completions per prompt variant.\n"
            "For strict_mc / answer_only, this is forced to 1 because the pipeline uses first-token choice scoring."
        ),
    )
    sampling_group.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size for sampling draws per prompt. Falls back to sequential on OOM/runtime generation errors.",
    )
    sampling_group.add_argument(
        "--sampling_checkpoint_every",
        type=int,
        default=200,
        help="Persist sampling checkpoint every N newly generated responses. Set 0 to disable periodic checkpoints.",
    )
    sampling_group.add_argument(
        "--no_reuse_sampling_cache",
        action="store_true",
        help="Disable reuse of matching sampling checkpoints from earlier runs.",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature.\n"
            "If omitted, answer_only under ays_mc_single_turn defaults to 0.1; all other cases default to 0.7.\n"
            "For strict_mc / answer_only, this is forced to 0.0 because no stochastic sampling is used."
        ),
    )
    ap.set_defaults(temperature=None)
    sampling_group.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling parameter used during generation.",
    )
    sampling_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum generated tokens. If omitted, the pipeline uses a high default budget of 256 to avoid truncating answer-bearing completions.",
    )
    probe_group.add_argument(
        "--probe_feature_mode",
        type=str,
        default="response_raw_final_token",
        choices=["response_raw_final_token"],
        help="Probe on the final token of the full raw sampled completion.",
    )
    probe_group.add_argument(
        "--probe_construction",
        type=str,
        default="auto",
        choices=["auto", "sampled_completions", "choice_candidates"],
        help=(
            "How to construct probe-training examples.\n"
            "'auto' uses strict-MC choice candidates when available and falls back to sampled completions otherwise.\n"
            "'sampled_completions' always trains probes on sampled responses.\n"
            "'choice_candidates' teacher-forces every allowed answer choice for strict-MC prompts."
        ),
    )
    probe_group.add_argument(
        "--probe_example_weighting",
        type=str,
        default="model_probability",
        choices=["model_probability", "uniform"],
        help=(
            "Per-example weighting strategy for probe training.\n"
            "For strict-MC choice-candidate probes, 'model_probability' weights each candidate by the model's "
            "first-token choice probability, while 'uniform' gives each candidate equal weight.\n"
            "Sampled-completion probes currently use uniform weighting regardless."
        ),
    )

    probe_group.add_argument(
        "--probe_layer_min",
        type=int,
        default=1,
        help="First transformer layer index considered during probe layer selection.",
    )
    probe_group.add_argument(
        "--probe_layer_max",
        type=int,
        default=32,
        help="Last transformer layer index considered during probe layer selection.",
    )
    probe_group.add_argument(
        "--probe_val_frac",
        "--val_frac",
        dest="probe_val_frac",
        type=float,
        default=0.2,
        help="Fraction of non-test questions reserved for the question-level validation split used in probe layer selection.",
    )
    probe_group.add_argument(
        "--probe_seed",
        type=int,
        default=0,
        help="Random seed used in probe layer selection and retraining.",
    )
    probe_group.add_argument(
        "--probe_selection_max_samples",
        type=int,
        default=2000,
        help="Optional cap on records used per layer during probe selection.",
    )
    probe_group.add_argument(
        "--probe_train_max_samples",
        type=int,
        default=None,
        help="Optional cap on records used when retraining the final selected probe.",
    )

    io_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global run seed for sampling, grouping, and model-side randomness where applicable.",
    )
    io_group.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Optional .env file to load before resolving cache env vars. Use empty string to disable.",
    )
    io_group.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="HF model/tokenizer cache dir. If unset, uses HF_HUB_CACHE or HUGGINGFACE_HUB_CACHE env vars.",
    )
    io_group.add_argument(
        "--out_dir",
        type=str,
        default="results/sycophancy_bias_probe",
        help="Root directory where run artifacts are written.",
    )
    io_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional explicit run directory name. If omitted, the runtime builds one automatically.",
    )
    return ap


def _validate_cli_dependencies(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.benchmark_source == "answer_json" and args.input_jsonl != "answer.jsonl":
        ap.error("--benchmark_source=answer_json requires --input_jsonl=answer.jsonl.")
    if args.benchmark_source == "ays_mc_single_turn" and args.input_jsonl != "are_you_sure.jsonl":
        ap.error("--benchmark_source=ays_mc_single_turn requires --input_jsonl=are_you_sure.jsonl.")
    try:
        resolve_bias_types(args.bias_types)
    except ValueError as exc:
        ap.error(str(exc))


def _apply_effective_sampling_overrides(args: argparse.Namespace) -> None:
    if args.mc_mode == "strict_mc":
        args.n_draws = 1
        args.temperature = 0.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        args.instruction_policy = canonical_instruction_policy_name(args.instruction_policy)
    except ValueError as exc:
        ap.error(str(exc))
    args.mc_mode = legacy_mc_mode_for_instruction_policy(args.instruction_policy)
    _validate_cli_dependencies(ap, args)
    if args.max_new_tokens is None:
        args.max_new_tokens = 256
    if args.temperature is None:
        if args.benchmark_source == "ays_mc_single_turn" and args.instruction_policy == "answer_only":
            args.temperature = 0.1
        else:
            args.temperature = 0.7
    _apply_effective_sampling_overrides(args)
    args.prompt_spec_version = int(PROMPT_SPEC_VERSION)
    args.grading_spec_version = int(GRADING_SPEC_VERSION)
    args.generation_spec_version = int(GENERATION_SPEC_VERSION)
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
