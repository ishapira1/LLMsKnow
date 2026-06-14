from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_bootstrap_src_path()

from llmssycoph.cli import load_env_file
from llmssycoph.data import (
    build_question_groups,
    deduplicate_rows,
    ensure_sycophancy_eval_cached,
    load_external_ays_mc_rows,
    prepare_benchmark_rows,
    read_jsonl,
    split_groups_by_source_split,
    split_groups_train_val_test,
)
from llmssycoph.data.agreement_biases.neutral_bias import NeutralBias
from llmssycoph.data.datasets import render_multiple_choice_question
from llmssycoph.data.instruction_policies.answer_only_policy import AnswerOnlyPolicy
from llmssycoph.data.question import Question
from llmssycoph.logging_utils import tqdm_desc
from llmssycoph.runtime import utc_now_iso, write_json_atomic, write_jsonl_atomic


SCRIPT_NAME = Path(__file__).name
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENV_FILE = ".env"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_DATASETS = ("commonsense_qa", "arc_challenge")
DEFAULT_TRACKED_OUTPUT_DIR = "data/ad_hoc/paraphrase_robustness_test_stems_v1"
DEFAULT_SCRATCH_OUTPUT_DIR = "output/paraphrase_robustness_test_stems_v1"
DEFAULT_MAX_ESTIMATED_COST_USD = 5.0
DEFAULT_BATCH_COMPLETION_WINDOW = "24h"
DEFAULT_MAX_COMPLETION_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_OUTPUT_TOKEN_MULTIPLIER = 1.2
DEFAULT_SPLIT_SEED = 5
DEFAULT_TEST_FRAC = 0.2
DEFAULT_PROBE_VAL_FRAC = 0.2

TRACKED_DATASET_FILES = {
    "commonsense_qa": "commonsense_qa_test_paraphrases.jsonl",
    "arc_challenge": "arc_challenge_test_paraphrases.jsonl",
}
TRACKED_MANIFEST_NAME = "paraphrase_manifest.json"
TRACKED_SUMMARY_NAME = "paraphrase_summary.md"

SCRATCH_RUN_STATE_NAME = "run_state.json"
SCRATCH_STAGE_LOG_NAME = "stage.log"
SCRATCH_PREPARED_ITEMS_NAME = "prepared_items.jsonl"
SCRATCH_BATCH_REQUESTS_NAME = "batch_requests.jsonl"
SCRATCH_BATCH_OUTPUT_NAME = "batch_output_raw.jsonl"
SCRATCH_BATCH_ERROR_NAME = "batch_error_raw.jsonl"

EXPECTED_TEST_COUNTS = {
    "commonsense_qa": 2192,
    "arc_challenge": 1172,
}

BATCH_PRICING_USD_PER_MILLION = {
    "gpt-5.4": {
        "input": 1.25,
        "output": 7.50,
    },
    "gpt-5.4-mini": {
        "input": 0.375,
        "output": 2.25,
    },
}

GENERATION_SYSTEM_PROMPT = (
    "You rewrite multiple-choice question stems conservatively.\n"
    "Preserve the exact meaning, difficulty, scenario details, and correct answer.\n"
    "Return only the rewritten stem with no explanation."
)
GENERATION_USER_PROMPT_TEMPLATE = (
    "Rewrite only the question stem below.\n\n"
    "Rules:\n"
    "- Keep the exact same meaning, difficulty, and correct answer.\n"
    "- Preserve all concrete details from the original stem, including names, quantities, locations, time references, and causal structure.\n"
    "- Do not simplify by removing context.\n"
    "- Do not change tense, modality, scope, or question type.\n"
    "- Do not mention, alter, or copy answer labels or option markers.\n"
    "- Return exactly one rewritten stem on one line.\n"
    "- If a fully conservative paraphrase is not possible, return the original stem unchanged.\n\n"
    "Question stem:\n"
    "{original_stem}"
)

API_MODE = "openai_batch_chat_completions"
SCHEMA_VERSION = 1
SELECTED_BIAS_TYPES = ("incorrect_suggestion", "doubt_correct", "suggest_correct")
ANSWER_OPTIONS_MARKER_RE = re.compile(r"\([A-Za-z0-9]\)")
ANSWER_LABEL_RE = re.compile(r"(^|\s)(Answer\s*:|[A-E][\)\.\:\-]\s)", re.IGNORECASE)


def _repo_rel(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _append_stage_log(scratch_dir: Path, message: str) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / SCRATCH_STAGE_LOG_NAME
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now_iso()}] {message}\n")


def _deep_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except Exception:
                return None
            continue
        current = getattr(current, part, None)
    return current


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _require_openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY_FOR_PROJECT") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY_FOR_PROJECT (or OPENAI_API_KEY) in environment or .env."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the 'openai' package. Install project dependencies first."
        ) from exc
    kwargs: Dict[str, Any] = {"api_key": api_key}
    org_id = str(os.getenv("OPENAI_ORG_ID", "") or "").strip()
    if org_id:
        kwargs["organization"] = org_id
    try:
        return OpenAI(**kwargs)
    except TypeError:
        if org_id:
            kwargs.pop("organization", None)
            kwargs["default_headers"] = {"OpenAI-Organization": org_id}
        return OpenAI(**kwargs)


def _parse_datasets(value: str) -> List[str]:
    datasets = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not datasets:
        raise ValueError("At least one dataset is required.")
    invalid = [dataset for dataset in datasets if dataset not in DEFAULT_DATASETS]
    if invalid:
        raise ValueError(
            f"Unsupported datasets {invalid}. Valid values: {list(DEFAULT_DATASETS)}"
        )
    return datasets


def _estimate_tokens_from_text(text: str) -> int:
    cleaned = str(text or "")
    if not cleaned:
        return 0
    return int(math.ceil(len(cleaned) / 4.0))


def _pricing_for_model(model_name: str) -> Dict[str, float]:
    pricing = BATCH_PRICING_USD_PER_MILLION.get(str(model_name or "").strip())
    if pricing is None:
        raise ValueError(
            f"No batch pricing entry is configured for model {model_name!r}. "
            f"Valid values: {sorted(BATCH_PRICING_USD_PER_MILLION)}"
        )
    return dict(pricing)


def _build_generation_messages(original_stem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERATION_USER_PROMPT_TEMPLATE.format(original_stem=original_stem),
        },
    ]


def _raw_response_text_from_batch_line(payload: Dict[str, Any]) -> str:
    response_body = _deep_get(payload, "response.body")
    if response_body is None:
        return ""
    content = _deep_get(response_body, "choices.0.message.content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    texts.append(str(text))
        return "".join(texts).strip()
    return str(content or "").strip()


def _normalized_whitespace(value: str) -> str:
    return " ".join(str(value or "").split())


def _render_neutral_prompt_text(base_metadata: Dict[str, Any], stem: str) -> str:
    working_base = dict(base_metadata)
    working_base["question"] = str(stem or "").strip()
    working_base["question_text"] = str(stem or "").strip()
    question_text = render_multiple_choice_question(working_base)
    question = Question(
        dataset=str(working_base.get("dataset", "") or "").strip(),
        question_text=question_text,
        correct_answer=str(working_base.get("correct_answer", "") or "").strip(),
        incorrect_answer=str(working_base.get("incorrect_answer", "") or "").strip(),
        base_metadata=working_base,
    )
    return NeutralBias().render_prompt_text(question, instruction_policy=AnswerOnlyPolicy())


def _source_row_map(rows: Sequence[Dict[str, Any]], dataset_name: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        base = row.get("base", {}) or {}
        if str(base.get("dataset", "") or "").strip() != dataset_name:
            continue
        source_example_id = str(base.get("source_example_id", "") or "").strip()
        if source_example_id:
            mapping[source_example_id] = row
    return mapping


def _canonical_test_groups_for_dataset(dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    data_files = ensure_sycophancy_eval_cached(data_dir=str(REPO_ROOT / "data" / "sycophancy-eval"))
    rows_raw = read_jsonl(data_files["are_you_sure.jsonl"])
    external_rows = load_external_ays_mc_rows(
        data_dir=str(REPO_ROOT / "data" / "sycophancy-eval"),
        selected_ays_mc_datasets=[dataset_name],
        force_download=False,
    )
    rows_raw = [*rows_raw, *external_rows]
    prepared_rows = prepare_benchmark_rows(
        benchmark_source="ays_mc_single_turn",
        rows=rows_raw,
        input_jsonl="are_you_sure.jsonl",
        selected_bias_types=SELECTED_BIAS_TYPES,
        selected_ays_mc_datasets=[dataset_name],
        instruction_policy="answer_only",
        mc_mode="strict_mc",
        seed=DEFAULT_SPLIT_SEED,
    )
    rows = deduplicate_rows(prepared_rows)
    groups = build_question_groups(
        rows,
        selected_bias_types=SELECTED_BIAS_TYPES,
        selected_dataset_name=dataset_name,
    )
    if dataset_name == "arc_challenge":
        _, _, test_groups = split_groups_by_source_split(groups)
    else:
        _, _, test_groups = split_groups_train_val_test(
            groups,
            test_frac=DEFAULT_TEST_FRAC,
            val_frac=DEFAULT_PROBE_VAL_FRAC,
            seed=DEFAULT_SPLIT_SEED,
        )

    expected = EXPECTED_TEST_COUNTS[dataset_name]
    if len(test_groups) != expected:
        raise ValueError(
            f"Canonical test split mismatch for {dataset_name}: got {len(test_groups)} expected {expected}"
        )
    return test_groups, _source_row_map(external_rows, dataset_name)


def _prepared_item_from_group(
    dataset_name: str,
    group: Dict[str, Any],
    source_rows: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    neutral_row = group["rows_by_type"]["neutral"]
    base = dict(neutral_row.get("base", {}) or {})
    source_example_id = str(base.get("source_example_id", "") or "").strip()
    source_row = source_rows.get(source_example_id)
    if source_row is None:
        raise KeyError(f"Missing source row for source_example_id={source_example_id!r} dataset={dataset_name}")
    source_base = source_row.get("base", {}) or {}
    original_stem = str(source_base.get("question", "") or "").strip()
    if not original_stem:
        raise ValueError(f"Missing original stem for source_example_id={source_example_id!r}")

    original_prompt_text = _render_neutral_prompt_text(base, original_stem)
    return {
        "dataset": dataset_name,
        "split": "test",
        "question_id": str(group.get("question_id", "") or ""),
        "source_example_id": source_example_id,
        "source_split": str(source_base.get("source_split", "") or "").strip(),
        "original_stem": original_stem,
        "correct_letter": str(base.get("correct_letter", "") or "").strip(),
        "letters": str(base.get("letters", "") or "").strip(),
        "answers_list": list(base.get("answers_list", []) or []),
        "answer_options": str(base.get("answers", "") or ""),
        "original_prompt_text": original_prompt_text,
        "neutral_base": base,
        "custom_id": f"{dataset_name}::{source_example_id}",
    }


def build_prepared_items(
    dataset_names: Sequence[str],
    *,
    max_items: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    dataset_counts: Dict[str, int] = {}

    for dataset_name in tqdm(
        dataset_names,
        desc=tqdm_desc(SCRIPT_NAME, "rebuilding canonical test splits"),
        unit="dataset",
    ):
        test_groups, source_rows = _canonical_test_groups_for_dataset(dataset_name)
        dataset_counts[dataset_name] = len(test_groups)
        dataset_items = [
            _prepared_item_from_group(dataset_name, group, source_rows)
            for group in tqdm(
                test_groups,
                desc=tqdm_desc(SCRIPT_NAME, f"preparing {dataset_name} test stems"),
                unit="question",
            )
        ]
        dataset_items.sort(key=lambda row: (row["question_id"], row["source_example_id"]))
        all_items.extend(dataset_items)

    all_items.sort(key=lambda row: (row["dataset"], row["question_id"], row["source_example_id"]))
    if max_items is not None:
        all_items = all_items[: max(0, int(max_items))]

    stats = {
        "expected_test_counts": {dataset: EXPECTED_TEST_COUNTS[dataset] for dataset in dataset_names},
        "canonical_test_counts": dataset_counts,
        "prepared_item_count": len(all_items),
        "prepared_counts_by_dataset": {
            dataset: sum(1 for row in all_items if row["dataset"] == dataset)
            for dataset in dataset_names
        },
    }
    return all_items, stats


def _build_batch_request_line(item: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    return {
        "custom_id": str(item["custom_id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": _build_generation_messages(str(item["original_stem"])),
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "reasoning_effort": "none",
        },
    }


def build_batch_requests(items: Sequence[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    return [_build_batch_request_line(item, model_name) for item in items]


def estimate_request_cost(
    request_lines: Sequence[Dict[str, Any]],
    items: Sequence[Dict[str, Any]],
    *,
    model_name: str,
) -> Dict[str, Any]:
    pricing = _pricing_for_model(model_name)
    serialized_lines = [json.dumps(line, ensure_ascii=False, sort_keys=True) for line in request_lines]
    serialized_payload = "\n".join(serialized_lines)
    estimated_input_tokens = _estimate_tokens_from_text(serialized_payload)
    original_stem_tokens = sum(_estimate_tokens_from_text(str(item["original_stem"])) for item in items)
    estimated_output_tokens = int(math.ceil(original_stem_tokens * DEFAULT_OUTPUT_TOKEN_MULTIPLIER))
    estimated_total_cost_usd = (
        estimated_input_tokens / 1_000_000.0 * pricing["input"]
        + estimated_output_tokens / 1_000_000.0 * pricing["output"]
    )
    return {
        "pricing": pricing,
        "input_estimation_method": "serialized_batch_json_chars_div_4",
        "output_estimation_method": "original_stem_chars_div_4_times_multiplier",
        "output_token_multiplier": DEFAULT_OUTPUT_TOKEN_MULTIPLIER,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_cost_usd": round(float(estimated_total_cost_usd), 6),
        "original_stem_token_estimate": original_stem_tokens,
    }


def validate_paraphrased_stem(original_stem: str, raw_model_text: str) -> Tuple[str, List[str]]:
    stripped = str(raw_model_text or "").strip()
    normalized = _normalized_whitespace(stripped)
    original_normalized = _normalized_whitespace(original_stem)
    flags: List[str] = []
    if not normalized:
        flags.append("empty_output")
    if "\n" in stripped:
        flags.append("multi_line_output")
    if ANSWER_OPTIONS_MARKER_RE.search(stripped):
        flags.append("contains_option_marker")
    if ANSWER_LABEL_RE.search(stripped):
        flags.append("contains_answer_label")
    if normalized and normalized == original_normalized:
        flags.append("unchanged_stem")
    normalized_len = len(normalized)
    original_len = len(original_normalized)
    too_short_threshold = max(15, int(round(original_len * 0.5)))
    too_long_threshold = max(original_len + 20, int(round(original_len * 1.8)))
    if normalized and normalized_len < too_short_threshold:
        flags.append("too_short")
    if normalized_len > too_long_threshold:
        flags.append("too_long")
    return normalized, sorted(set(flags))


def _print_cost_block(estimate: Dict[str, Any], max_estimated_cost_usd: float) -> None:
    print("Estimated Batch API cost:")
    print(f"  input tokens:   {estimate['estimated_input_tokens']}")
    print(f"  output tokens:  {estimate['estimated_output_tokens']}")
    print(f"  total USD:      ${estimate['estimated_total_cost_usd']:.4f}")
    print(f"  configured cap: ${float(max_estimated_cost_usd):.4f}")


def _manifest_paths(tracked_output_dir: Path) -> Dict[str, Path]:
    return {
        "manifest": tracked_output_dir / TRACKED_MANIFEST_NAME,
        "summary": tracked_output_dir / TRACKED_SUMMARY_NAME,
        **{
            dataset: tracked_output_dir / filename
            for dataset, filename in TRACKED_DATASET_FILES.items()
        },
    }


def _scratch_paths(scratch_output_dir: Path) -> Dict[str, Path]:
    return {
        "run_state": scratch_output_dir / SCRATCH_RUN_STATE_NAME,
        "stage_log": scratch_output_dir / SCRATCH_STAGE_LOG_NAME,
        "prepared_items": scratch_output_dir / SCRATCH_PREPARED_ITEMS_NAME,
        "batch_requests": scratch_output_dir / SCRATCH_BATCH_REQUESTS_NAME,
        "batch_output_raw": scratch_output_dir / SCRATCH_BATCH_OUTPUT_NAME,
        "batch_error_raw": scratch_output_dir / SCRATCH_BATCH_ERROR_NAME,
    }


def _load_run_state(scratch_output_dir: Path) -> Dict[str, Any]:
    return _read_json(_scratch_paths(scratch_output_dir)["run_state"])


def _prepared_items_for_state(scratch_output_dir: Path) -> List[Dict[str, Any]]:
    return read_jsonl(str(_scratch_paths(scratch_output_dir)["prepared_items"]))


def _write_run_state(state: Dict[str, Any], scratch_output_dir: Path, tracked_output_dir: Path) -> None:
    scratch_paths = _scratch_paths(scratch_output_dir)
    tracked_paths = _manifest_paths(tracked_output_dir)
    write_json_atomic(scratch_paths["run_state"], state)
    write_json_atomic(tracked_paths["manifest"], state)
    _write_text(tracked_paths["summary"], build_summary_markdown(state))


def build_summary_markdown(state: Dict[str, Any]) -> str:
    dataset_names = list(state.get("datasets", []))
    final_rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in list(state.get("final_rows", []) or []):
        final_rows_by_dataset.setdefault(str(row.get("dataset", "") or ""), []).append(row)

    lines = [
        "# Test-Stem Paraphrase Package",
        "",
        f"- script: `{state.get('script_name', SCRIPT_NAME)}`",
        f"- model: `{state.get('model', '')}`",
        f"- api mode: `{state.get('api_mode', API_MODE)}`",
        f"- stage: `{state.get('stage', 'unknown')}`",
        f"- estimated total API cost: `${float(state.get('cost_estimate', {}).get('estimated_total_cost_usd', 0.0)):.4f}`",
        f"- configured budget cap: `${float(state.get('budget_cap_usd', DEFAULT_MAX_ESTIMATED_COST_USD)):.4f}`",
        f"- submission blocked by cost guard: `{bool(state.get('submission_blocked_by_cost_guard', False))}`",
        "",
    ]

    for dataset_name in dataset_names:
        prepared_count = int(state.get("prepared_counts_by_dataset", {}).get(dataset_name, 0) or 0)
        final_rows = final_rows_by_dataset.get(dataset_name, [])
        invalid_count = sum(1 for row in final_rows if str(row.get("status", "")) != "valid")
        unchanged_count = sum(
            1
            for row in final_rows
            if "unchanged_stem" in list(row.get("validation_flags", []) or [])
        )
        before_lengths = [int(row.get("original_stem_char_len", 0) or 0) for row in final_rows]
        after_lengths = [int(row.get("paraphrased_stem_char_len", 0) or 0) for row in final_rows]
        avg_before = sum(before_lengths) / len(before_lengths) if before_lengths else 0.0
        avg_after = sum(after_lengths) / len(after_lengths) if after_lengths else 0.0
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                f"- prepared rows: `{prepared_count}`",
                f"- final rows: `{len(final_rows)}`",
                f"- invalid rows: `{invalid_count}`",
                f"- unchanged rows: `{unchanged_count}`",
                f"- average original stem chars: `{avg_before:.2f}`",
                f"- average paraphrased stem chars: `{avg_after:.2f}`",
                "",
            ]
        )
        if final_rows:
            sample_count = min(5, len(final_rows))
            rng = random.Random(20260614 + len(dataset_name))
            sampled_rows = rng.sample(final_rows, sample_count)
            lines.append("### Sample before/after pairs")
            lines.append("")
            for idx, row in enumerate(sampled_rows, start=1):
                lines.append(f"{idx}. original: {row.get('original_stem', '')}")
                lines.append(f"   paraphrased: {row.get('paraphrased_stem', '')}")
                lines.append(
                    f"   status: {row.get('status', '')} flags: {', '.join(list(row.get('validation_flags', []) or [])) or 'none'}"
                )
            lines.append("")
        else:
            lines.append("_Paraphrases not collected yet._")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _new_state(args: argparse.Namespace, dataset_names: Sequence[str], stats: Dict[str, Any]) -> Dict[str, Any]:
    tracked_output_dir = Path(args.tracked_output_dir)
    scratch_output_dir = Path(args.scratch_output_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "script_name": SCRIPT_NAME,
        "api_mode": API_MODE,
        "run_mode": "batch",
        "model": str(args.model),
        "datasets": list(dataset_names),
        "stage": "prepared",
        "generated_at": utc_now_iso(),
        "prepared_at": utc_now_iso(),
        "budget_cap_usd": float(args.max_estimated_cost_usd),
        "batch_completion_window": DEFAULT_BATCH_COMPLETION_WINDOW,
        "batch_description": str(args.batch_description or "").strip(),
        "pricing_table_used": {
            str(args.model): _pricing_for_model(str(args.model)),
        },
        "dataset_split_policy": {
            "commonsense_qa": {
                "type": "local_split",
                "test_frac": DEFAULT_TEST_FRAC,
                "probe_val_frac": DEFAULT_PROBE_VAL_FRAC,
                "split_seed": DEFAULT_SPLIT_SEED,
            },
            "arc_challenge": {
                "type": "source_test_split",
            },
        },
        "expected_test_counts": stats["expected_test_counts"],
        "canonical_test_counts": stats["canonical_test_counts"],
        "prepared_counts_by_dataset": stats["prepared_counts_by_dataset"],
        "prepared_item_count": int(stats["prepared_item_count"]),
        "realized_counts_by_dataset": {
            dataset: 0
            for dataset in dataset_names
        },
        "realized_row_count_total": 0,
        "submission_blocked_by_cost_guard": False,
        "cost_estimate": {},
        "openai_submission": {
            "input_file_id": "",
            "batch_id": "",
            "batch_status": "",
            "output_file_id": "",
            "error_file_id": "",
            "submitted_at": "",
            "completed_at": "",
            "request_counts": {},
        },
        "files": {
            "tracked_output_dir": _repo_rel(tracked_output_dir),
            "scratch_output_dir": _repo_rel(scratch_output_dir),
            "prepared_items_jsonl": _repo_rel(_scratch_paths(scratch_output_dir)["prepared_items"]),
            "batch_requests_jsonl": _repo_rel(_scratch_paths(scratch_output_dir)["batch_requests"]),
            "batch_output_raw_jsonl": _repo_rel(_scratch_paths(scratch_output_dir)["batch_output_raw"]),
            "batch_error_raw_jsonl": _repo_rel(_scratch_paths(scratch_output_dir)["batch_error_raw"]),
            "stage_log": _repo_rel(_scratch_paths(scratch_output_dir)["stage_log"]),
            "manifest_json": _repo_rel(_manifest_paths(tracked_output_dir)["manifest"]),
            "summary_md": _repo_rel(_manifest_paths(tracked_output_dir)["summary"]),
            **{
                f"{dataset}_tracked_jsonl": _repo_rel(_manifest_paths(tracked_output_dir)[dataset])
                for dataset in dataset_names
            },
        },
        "generation_prompt_template": {
            "system": GENERATION_SYSTEM_PROMPT,
            "user_template": GENERATION_USER_PROMPT_TEMPLATE,
        },
        "final_rows": [],
    }


def build_prepared_payload(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_names = _parse_datasets(args.datasets)
    items, stats = build_prepared_items(dataset_names, max_items=args.max_items)
    request_lines = build_batch_requests(items, str(args.model))
    state = _new_state(args, dataset_names, stats)
    state["cost_estimate"] = estimate_request_cost(request_lines, items, model_name=str(args.model))
    return state, items, request_lines


def _ensure_prepare_can_write(scratch_output_dir: Path, force: bool) -> None:
    scratch_paths = _scratch_paths(scratch_output_dir)
    existing = [path for path in scratch_paths.values() if path.exists()]
    if existing and not force:
        preview = ", ".join(_repo_rel(path) for path in existing[:5])
        raise FileExistsError(
            f"Prepare would overwrite existing scratch artifacts: {preview}. Use --force to overwrite."
        )


def command_prepare(args: argparse.Namespace) -> int:
    scratch_output_dir = REPO_ROOT / args.scratch_output_dir
    tracked_output_dir = REPO_ROOT / args.tracked_output_dir
    _ensure_prepare_can_write(scratch_output_dir, force=bool(args.force))
    scratch_output_dir.mkdir(parents=True, exist_ok=True)
    tracked_output_dir.mkdir(parents=True, exist_ok=True)
    _append_stage_log(scratch_output_dir, "prepare: starting")
    state, items, request_lines = build_prepared_payload(args)
    write_jsonl_atomic(_scratch_paths(scratch_output_dir)["prepared_items"], items)
    write_jsonl_atomic(_scratch_paths(scratch_output_dir)["batch_requests"], request_lines)
    _write_run_state(state, scratch_output_dir, tracked_output_dir)
    _append_stage_log(
        scratch_output_dir,
        f"prepare: wrote {len(items)} items and {len(request_lines)} batch requests",
    )
    _print_cost_block(state["cost_estimate"], float(args.max_estimated_cost_usd))
    print(
        f"Prepared {len(items)} test-stem paraphrase tasks "
        f"-> {_repo_rel(_scratch_paths(scratch_output_dir)['batch_requests'])}"
    )
    return 0


def _budget_guard_or_raise(state: Dict[str, Any], max_estimated_cost_usd: float, *, force: bool) -> None:
    estimate = dict(state.get("cost_estimate", {}) or {})
    estimated_total_cost_usd = float(estimate.get("estimated_total_cost_usd", 0.0) or 0.0)
    previously_blocked = bool(state.get("submission_blocked_by_cost_guard", False))
    if previously_blocked:
        if not force:
            raise RuntimeError(
                "A previous submit attempt was blocked by the cost guard. "
                "Re-run with --force and a higher --max-estimated-cost-usd."
            )
        if float(max_estimated_cost_usd) <= estimated_total_cost_usd:
            raise RuntimeError(
                "Re-running after a budget failure requires a higher cap than the stored estimated total cost."
            )
    if estimated_total_cost_usd > float(max_estimated_cost_usd):
        state["submission_blocked_by_cost_guard"] = True
        state["budget_cap_usd"] = float(max_estimated_cost_usd)
        raise RuntimeError(
            "Estimated cost exceeds the configured cap.\n"
            f"estimated_input_tokens={estimate.get('estimated_input_tokens')}\n"
            f"estimated_output_tokens={estimate.get('estimated_output_tokens')}\n"
            f"estimated_total_usd={estimated_total_cost_usd:.4f}\n"
            f"configured_cap_usd={float(max_estimated_cost_usd):.4f}"
        )


def command_submit(args: argparse.Namespace) -> int:
    scratch_output_dir = REPO_ROOT / args.scratch_output_dir
    tracked_output_dir = REPO_ROOT / args.tracked_output_dir
    scratch_paths = _scratch_paths(scratch_output_dir)
    state = _load_run_state(scratch_output_dir)
    state["budget_cap_usd"] = float(args.max_estimated_cost_usd)
    if str(state.get("model", "")) != str(args.model):
        raise ValueError(
            f"Prepared state model={state.get('model')!r} does not match requested model={args.model!r}. "
            "Run prepare again with --force if you want to change models."
        )
    if str(state.get("openai_submission", {}).get("batch_id", "") or "").strip() and not args.force:
        raise RuntimeError("A batch has already been submitted for this scratch directory. Use --force to resubmit.")

    _print_cost_block(state["cost_estimate"], float(args.max_estimated_cost_usd))
    try:
        _budget_guard_or_raise(state, float(args.max_estimated_cost_usd), force=bool(args.force))
    except Exception:
        _write_run_state(state, scratch_output_dir, tracked_output_dir)
        _append_stage_log(scratch_output_dir, "submit: blocked by cost guard")
        raise

    load_env_file(args.env_file)
    client = _require_openai_client()
    request_path = scratch_paths["batch_requests"]
    if not request_path.exists():
        raise FileNotFoundError(f"Missing batch request file: {request_path}")

    _append_stage_log(scratch_output_dir, "submit: uploading batch request file")
    with request_path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")
    file_obj = client.files.wait_for_processing(file_obj.id)
    _append_stage_log(scratch_output_dir, f"submit: uploaded input file {file_obj.id}")

    batch_metadata = {"script": SCRIPT_NAME}
    if str(args.batch_description or "").strip():
        batch_metadata["description"] = str(args.batch_description).strip()[:500]
    batch = client.batches.create(
        completion_window=DEFAULT_BATCH_COMPLETION_WINDOW,
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        metadata=batch_metadata,
    )
    openai_submission = state.setdefault("openai_submission", {})
    openai_submission.update(
        {
            "input_file_id": str(file_obj.id),
            "batch_id": str(batch.id),
            "batch_status": str(getattr(batch, "status", "") or ""),
            "output_file_id": str(getattr(batch, "output_file_id", "") or ""),
            "error_file_id": str(getattr(batch, "error_file_id", "") or ""),
            "submitted_at": utc_now_iso(),
            "completed_at": "",
            "request_counts": {},
        }
    )
    state["submission_blocked_by_cost_guard"] = False
    state["stage"] = "submitted"
    _write_run_state(state, scratch_output_dir, tracked_output_dir)
    _append_stage_log(scratch_output_dir, f"submit: created batch {batch.id}")
    print(f"Submitted Batch API job: batch_id={batch.id} input_file_id={file_obj.id}")
    return 0


def _update_state_from_batch(state: Dict[str, Any], batch: Any) -> None:
    request_counts = getattr(batch, "request_counts", None)
    request_counts_payload = {}
    if request_counts is not None:
        for key in ("total", "completed", "failed"):
            value = getattr(request_counts, key, None)
            if value is not None:
                request_counts_payload[key] = int(value)

    openai_submission = state.setdefault("openai_submission", {})
    openai_submission.update(
        {
            "batch_id": str(getattr(batch, "id", "") or ""),
            "batch_status": str(getattr(batch, "status", "") or ""),
            "input_file_id": str(getattr(batch, "input_file_id", "") or openai_submission.get("input_file_id", "")),
            "output_file_id": str(getattr(batch, "output_file_id", "") or ""),
            "error_file_id": str(getattr(batch, "error_file_id", "") or ""),
            "request_counts": request_counts_payload,
        }
    )
    completed_at = getattr(batch, "completed_at", None)
    if completed_at:
        openai_submission["completed_at"] = str(completed_at)


def command_status(args: argparse.Namespace) -> int:
    scratch_output_dir = REPO_ROOT / args.scratch_output_dir
    tracked_output_dir = REPO_ROOT / args.tracked_output_dir
    state = _load_run_state(scratch_output_dir)
    batch_id = str(state.get("openai_submission", {}).get("batch_id", "") or "").strip()
    if not batch_id:
        print("No batch has been submitted yet.")
        print(f"Current stage: {state.get('stage', 'prepared')}")
        return 0

    load_env_file(args.env_file)
    try:
        client = _require_openai_client()
    except Exception:
        print(f"Local batch metadata only: batch_id={batch_id}")
        print(f"Known status: {state.get('openai_submission', {}).get('batch_status', '')}")
        return 0

    batch = client.batches.retrieve(batch_id)
    _update_state_from_batch(state, batch)
    if str(state.get("stage", "") or "") != "collected":
        state["stage"] = "submitted"
    _write_run_state(state, scratch_output_dir, tracked_output_dir)
    _append_stage_log(scratch_output_dir, f"status: refreshed batch {batch_id} -> {getattr(batch, 'status', '')}")
    print(f"batch_id={batch_id}")
    print(f"status={state.get('openai_submission', {}).get('batch_status', '')}")
    print(f"output_file_id={state.get('openai_submission', {}).get('output_file_id', '')}")
    print(f"error_file_id={state.get('openai_submission', {}).get('error_file_id', '')}")
    return 0


def _download_file_text(client: Any, file_id: str) -> str:
    content = client.files.content(file_id)
    raw_bytes = getattr(content, "content", None)
    if isinstance(raw_bytes, bytes):
        return raw_bytes.decode("utf-8")
    return str(raw_bytes if raw_bytes is not None else content)


def _final_status_and_flags(raw_model_text: str, original_stem: str, error_payload: Any = None) -> Tuple[str, str, List[str]]:
    normalized_text, flags = validate_paraphrased_stem(original_stem, raw_model_text)
    if error_payload:
        flags = sorted(set([*flags, "api_error"]))
    status = "valid" if not flags else "invalid"
    return status, normalized_text, flags


def _build_final_rows(
    prepared_items: Sequence[Dict[str, Any]],
    raw_output_rows: Sequence[Dict[str, Any]],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    by_custom_id = {
        str(row.get("custom_id", "") or ""): row
        for row in raw_output_rows
        if str(row.get("custom_id", "") or "")
    }
    batch_id = str(state.get("openai_submission", {}).get("batch_id", "") or "")
    model_name = str(state.get("model", "") or "")
    final_rows: List[Dict[str, Any]] = []

    for item in tqdm(
        prepared_items,
        desc=tqdm_desc(SCRIPT_NAME, "assembling collected paraphrase rows"),
        unit="question",
    ):
        payload = by_custom_id.get(str(item["custom_id"]))
        raw_model_text = _raw_response_text_from_batch_line(payload or {})
        error_payload = None if payload is None else payload.get("error")
        validation_flags: List[str]
        if payload is None:
            status = "invalid"
            paraphrased_stem = ""
            validation_flags = ["missing_response", "empty_output"]
            raw_model_text = ""
        else:
            status, paraphrased_stem, validation_flags = _final_status_and_flags(
                raw_model_text,
                str(item["original_stem"]),
                error_payload=error_payload,
            )
        paraphrased_prompt_text = _render_neutral_prompt_text(
            dict(item["neutral_base"]),
            paraphrased_stem or str(item["original_stem"]),
        )
        final_rows.append(
            {
                "dataset": item["dataset"],
                "split": item["split"],
                "question_id": item["question_id"],
                "source_example_id": item["source_example_id"],
                "source_split": item["source_split"],
                "original_stem": item["original_stem"],
                "paraphrased_stem": paraphrased_stem,
                "correct_letter": item["correct_letter"],
                "letters": item["letters"],
                "answers_list": item["answers_list"],
                "answer_options": item["answer_options"],
                "original_prompt_text": item["original_prompt_text"],
                "paraphrased_prompt_text": paraphrased_prompt_text,
                "model_name": model_name,
                "api_mode": API_MODE,
                "batch_id": batch_id,
                "custom_id": item["custom_id"],
                "raw_model_text": raw_model_text,
                "status": status,
                "validation_flags": validation_flags,
                "original_stem_char_len": len(str(item["original_stem"])),
                "paraphrased_stem_char_len": len(paraphrased_stem),
            }
        )
    return final_rows


def command_collect(args: argparse.Namespace) -> int:
    scratch_output_dir = REPO_ROOT / args.scratch_output_dir
    tracked_output_dir = REPO_ROOT / args.tracked_output_dir
    scratch_paths = _scratch_paths(scratch_output_dir)
    tracked_paths = _manifest_paths(tracked_output_dir)
    state = _load_run_state(scratch_output_dir)
    batch_id = str(state.get("openai_submission", {}).get("batch_id", "") or "").strip()
    if not batch_id:
        raise RuntimeError("Cannot collect before submit: missing batch_id in local state.")

    load_env_file(args.env_file)
    client = _require_openai_client()
    batch = client.batches.retrieve(batch_id)
    _update_state_from_batch(state, batch)

    batch_status = str(state.get("openai_submission", {}).get("batch_status", "") or "")
    output_file_id = str(state.get("openai_submission", {}).get("output_file_id", "") or "")
    error_file_id = str(state.get("openai_submission", {}).get("error_file_id", "") or "")
    if batch_status != "completed" or not output_file_id:
        if error_file_id and (args.force or not scratch_paths["batch_error_raw"].exists()):
            error_text = _download_file_text(client, error_file_id)
            _write_text(scratch_paths["batch_error_raw"], error_text)
        _write_run_state(state, scratch_output_dir, tracked_output_dir)
        raise RuntimeError(
            f"Batch is not ready for collection: batch_id={batch_id} status={batch_status} output_file_id={output_file_id!r}"
        )

    if args.force or not scratch_paths["batch_output_raw"].exists():
        output_text = _download_file_text(client, output_file_id)
        _write_text(scratch_paths["batch_output_raw"], output_text)
        _append_stage_log(scratch_output_dir, f"collect: downloaded output file {output_file_id}")

    raw_output_rows = read_jsonl(str(scratch_paths["batch_output_raw"]))
    prepared_items = _prepared_items_for_state(scratch_output_dir)
    final_rows = _build_final_rows(prepared_items, raw_output_rows, state)

    for dataset_name in state.get("datasets", []):
        dataset_rows = [row for row in final_rows if row["dataset"] == dataset_name]
        if dataset_rows:
            write_jsonl_atomic(tracked_paths[dataset_name], dataset_rows)

    state["final_rows"] = final_rows
    state["realized_counts_by_dataset"] = {
        dataset_name: sum(1 for row in final_rows if row["dataset"] == dataset_name)
        for dataset_name in state.get("datasets", [])
    }
    state["realized_row_count_total"] = len(final_rows)
    state["stage"] = "collected"
    _write_run_state(state, scratch_output_dir, tracked_output_dir)
    _append_stage_log(
        scratch_output_dir,
        f"collect: wrote {len(final_rows)} final rows across datasets={state.get('datasets', [])}",
    )
    print(f"Collected {len(final_rows)} paraphrase rows.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and collect offline Batch API paraphrases for test-set question stems."
    )
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--tracked-output-dir", default=DEFAULT_TRACKED_OUTPUT_DIR)
    parser.add_argument("--scratch-output-dir", default=DEFAULT_SCRATCH_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--batch-description", default="")
    parser.add_argument("--max-estimated-cost-usd", type=float, default=DEFAULT_MAX_ESTIMATED_COST_USD)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("submit")
    subparsers.add_parser("status")
    subparsers.add_parser("collect")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "submit":
        return command_submit(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "collect":
        return command_collect(args)
    raise ValueError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
