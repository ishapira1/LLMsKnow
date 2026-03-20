from __future__ import annotations

import math
import os
import random
import textwrap
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from tqdm.auto import tqdm

from .cli import build_parser, load_env_file, resolve_bias_types, resolve_device, resolve_hf_cache_dir
from .constants import (
    MC_MODE_STRICT,
    STRICT_MC_MAX_CAP_HIT_RATE,
    STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_RATE_WARN,
    STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_SELECTED_PROB,
    STRICT_MC_COLLAPSE_MEDIAN_EFFECTIVE_OPTIONS_WARN,
    STRICT_MC_DOMINANT_SELECTED_LABEL_EXCESS_WARN,
    STRICT_MC_DOMINANT_SELECTED_LABEL_RATE_WARN,
    STRICT_MC_MAX_EXPLICIT_PARSE_FAILURES,
    STRICT_MC_MAX_MULTIPLE_ANSWER_MARKER_ROWS,
    STRICT_MC_MIN_EXACT_FORMAT_RATE,
    STRICT_MC_MIN_COMMITMENT_RATE,
    STRICT_MC_MIN_STARTS_WITH_ANSWER_RATE,
    STRICT_MC_SELECTED_LABEL_TV_DISTANCE_WARN,
)
from .data import (
    build_question_groups,
    deduplicate_rows,
    ensure_sycophancy_eval_cached,
    load_external_ays_mc_rows,
    prepare_benchmark_rows,
    read_jsonl,
    resolve_ays_mc_datasets,
    split_groups_train_val_test,
    unique_dataset_names,
)
from .grading import add_empirical_t, build_probe_record_sets, refresh_sample_records_for_groups
from .logging_utils import (
    build_warning_summary_payload,
    clear_run_logging,
    configure_run_logging,
    log_status,
    ok_status,
    tqdm_desc,
    warn_status,
)
from .llm import (
    build_sampling_spec,
    load_llm,
    load_sampling_cache_candidate,
    normalize_sample_records,
    resolve_llm_capabilities,
    sample_records_for_groups,
    sampling_spec_hash,
    sort_sample_records,
    enumerate_expected_sample_keys,
)
from .probes import (
    evaluate_probe_from_cache,
    filter_usable_probe_records,
    maybe_subsample,
    prepare_probe_eval_cache,
    save_probe_family_artifacts,
    score_records_with_probe,
    select_best_layer_by_auc,
    train_probe_for_layer,
    write_probe_group_manifest,
)
from .runtime import (
    acquire_run_lock,
    assert_resume_compatible,
    make_run_dir,
    preferred_run_artifact_path,
    release_run_lock,
    run_lock_path,
    utc_now_iso,
    write_json_atomic,
    write_run_status,
)
from .sampling_integrity import build_sampling_integrity_summary, log_sampling_integrity_summary
from .saving_manager import (
    build_terminal_final_stats_lines,
    persist_sampling_state,
    refresh_runtime_summary_artifacts,
    save_run_results,
    save_sampling_integrity_summary,
)


def _next_record_id(*groups_of_records: Sequence[Dict[str, Any]]) -> int:
    max_id = -1
    for rows in groups_of_records:
        for record in rows:
            try:
                max_id = max(max_id, int(record.get("record_id", -1)))
            except Exception:
                continue
    return max_id + 1


def _preview_text(value: Any, limit: int = 160) -> str:
    text = str(value).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _preview_lines(value: Any, *, limit: int = 320, width: int = 88) -> List[str]:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ["<empty>"]
    if len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."

    lines: List[str] = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [stripped])
    return lines or ["<empty>"]


def _append_preview_block(
    lines: List[str],
    label: str,
    value: Any,
    *,
    indent: int = 2,
    value_indent: int = 4,
    limit: int = 320,
    width: int = 88,
) -> None:
    lines.append(f"{' ' * indent}{label}:")
    lines.extend(
        f"{' ' * value_indent}{line}"
        for line in _preview_lines(value, limit=limit, width=width)
    )


def _format_group_example_lines(example: Dict[str, Any], bias_types: Sequence[str]) -> List[str]:
    lines = ["dataset example"]
    lines.append(f"  question_id: {example.get('question_id', '<unknown>')}")

    dataset_name = str(example.get("dataset", "") or "").strip()
    if dataset_name:
        lines.append(f"  dataset: {dataset_name}")

    _append_preview_block(lines, "question", example.get("question", ""), limit=360, width=88)
    _append_preview_block(lines, "correct_answer", example.get("correct_answer", ""), limit=220, width=88)
    _append_preview_block(lines, "incorrect_answer", example.get("incorrect_answer", ""), limit=220, width=88)

    rows_by_type = example.get("rows_by_type", {}) or {}
    for template_type in ["neutral", *bias_types]:
        row = rows_by_type.get(template_type)
        if not isinstance(row, dict):
            continue

        lines.append(f"  template={template_type}")
        _append_preview_block(
            lines,
            "prompt_template",
            (row.get("metadata", {}) or {}).get("prompt_template", ""),
            indent=4,
            value_indent=6,
            limit=360,
            width=84,
        )

        prompt_messages = row.get("prompt", [])
        lines.append("    prompt_messages:")
        if not isinstance(prompt_messages, list) or not prompt_messages:
            lines.append("      <empty>")
            continue

        for message_index, message in enumerate(prompt_messages, start=1):
            if not isinstance(message, dict):
                lines.append(f"      {message_index}. [message]")
                lines.extend(
                    f"        {line}"
                    for line in _preview_lines(message, limit=700, width=80)
                )
                continue

            role = str(message.get("type") or message.get("role") or "message")
            lines.append(f"      {message_index}. [{role}]")
            lines.extend(
                f"        {line}"
                for line in _preview_lines(message.get("content", ""), limit=700, width=80)
            )
    return lines


def _format_arg_value(value: Any) -> str:
    if value is None:
        return "<unset>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        rendered = ", ".join(str(item) for item in value)
        return rendered or "<empty>"
    rendered = str(value)
    return rendered if rendered else "<empty>"


def _format_parsed_argument_lines(args: Any) -> List[str]:
    parsed_args = vars(args)
    if not parsed_args:
        return ["parsed arguments: <none>"]

    cli_keys: List[str] = []
    seen: Set[str] = set()
    for action in build_parser()._actions:
        dest = getattr(action, "dest", None)
        if not isinstance(dest, str) or dest == "help" or dest in seen or dest not in parsed_args:
            continue
        cli_keys.append(dest)
        seen.add(dest)

    derived_keys = [key for key in parsed_args if key not in seen]
    ordered_keys = [*cli_keys, *derived_keys]
    width = max(len(key) for key in ordered_keys)

    lines = ["parsed arguments:"]
    for key in cli_keys:
        lines.append(f"  {key.ljust(width)} = {_format_arg_value(parsed_args[key])}")
    if derived_keys:
        lines.append("  derived settings:")
        for key in derived_keys:
            lines.append(f"    {key.ljust(width)} = {_format_arg_value(parsed_args[key])}")
    return lines


def _warn_strict_mc_temperature_bookkeeping(args: Any) -> None:
    if str(getattr(args, "mc_mode", "") or "") != MC_MODE_STRICT:
        return
    if str(getattr(args, "model_backend", "") or "") == "openai":
        return

    effective_temperature = float(getattr(args, "temperature", 1.0) or 1.0)
    if effective_temperature != 1.0:
        return

    requested_temperature = float(getattr(args, "requested_temperature", effective_temperature) or effective_temperature)
    if requested_temperature != effective_temperature:
        warn_status(
            "pipeline.py",
            "strict_mc_temperature_bookkeeping",
            f"strict MC mode overrides temperature from {requested_temperature} to 1.0 for bookkeeping. "
            "First-token choice scoring ignores temperature, but if any prompt later falls back to text "
            "generation this value will apply there.",
        )
        return

    warn_status(
        "pipeline.py",
        "strict_mc_temperature_bookkeeping",
        "strict MC mode records temperature=1.0 for bookkeeping. First-token choice scoring ignores "
        "temperature, but if any prompt later falls back to text generation this value will apply there.",
    )


def _warn_sampling_only_split_expectations(args: Any) -> None:
    if not bool(getattr(args, "sampling_only", False)):
        return

    test_frac = float(getattr(args, "test_frac", 0.0) or 0.0)
    probe_val_frac = float(getattr(args, "probe_val_frac", 0.0) or 0.0)
    if test_frac <= 0.0 and probe_val_frac <= 0.0:
        return

    warn_status(
        "pipeline.py",
        "sampling_only_split_active",
        "sampling-only mode skips probe training, but the pipeline still applies question splitting before "
        f"sampling (test_frac={test_frac}, probe_val_frac={probe_val_frac}). "
        "Some questions will still be assigned to val/test and sampled there. "
        "Use --test_frac 0 --probe_val_frac 0 if you want one unsplit sampling pool.",
    )


def _apply_model_backend_overrides(args: Any, model_capabilities: Any) -> None:
    args.model_backend = str(getattr(model_capabilities, "backend_name", "huggingface") or "huggingface")

    if (
        str(getattr(args, "mc_mode", "") or "") == MC_MODE_STRICT
        and not bool(getattr(model_capabilities, "supports_choice_scoring", False))
        and getattr(args, "requested_temperature", None) is not None
    ):
        args.temperature = float(args.requested_temperature)

    if not bool(getattr(args, "sampling_only", False)) and not bool(
        getattr(model_capabilities, "supports_hidden_state_probes", False)
    ):
        raise ValueError(
            f"model backend '{args.model_backend}' currently supports sampling-only runs only. "
            "Re-run with --sampling_only."
        )


def _count_expected_by_template(
    expected_keys: Set[tuple[str, str, str, int]],
    bias_types: Sequence[str],
) -> Dict[str, int]:
    counts = {template_type: 0 for template_type in ["neutral", *bias_types]}
    for _, _, template_type, _ in expected_keys:
        counts[template_type] = counts.get(template_type, 0) + 1
    return counts


def _probe_fit_subset(
    records: Sequence[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    usable_records = filter_usable_probe_records(records)
    return maybe_subsample(usable_records, max_samples, seed)


def _log_group_example(groups: Sequence[Dict[str, Any]], bias_types: Sequence[str]) -> None:
    if not groups:
        log_status("pipeline.py", "dataset example: no valid grouped questions found")
        return

    for line in _format_group_example_lines(groups[0], bias_types):
        log_status("pipeline.py", line)


def _row_uses_choice_scoring(row: Dict[str, Any]) -> bool:
    base = row.get("base", {}) or {}
    return (
        str(base.get("task_format", "") or "") == "multiple_choice"
        and str(base.get("mc_mode", "") or "") == MC_MODE_STRICT
        and bool(str(base.get("letters", "") or "").strip())
    )


def _choice_scoring_coverage(groups: Sequence[Dict[str, Any]], bias_types: Sequence[str]) -> tuple[bool, bool]:
    wanted_types = ["neutral", *bias_types]
    present = 0
    covered = 0
    for group in groups:
        for template_type in wanted_types:
            row = group.get("rows_by_type", {}).get(template_type)
            if not isinstance(row, dict):
                continue
            prompt_messages = row.get("prompt", [])
            if not isinstance(prompt_messages, list) or not prompt_messages:
                continue
            present += 1
            if _row_uses_choice_scoring(row):
                covered += 1
    return covered > 0, present > 0 and covered == present


def _multiple_choice_mode_summary(
    groups: Sequence[Dict[str, Any]],
    bias_types: Sequence[str],
) -> Dict[str, int]:
    summary = {
        "total_prompt_variants": 0,
        "multiple_choice": 0,
        "strict_mc": 0,
        "strict_mc_choice_scoring": 0,
        "strict_mc_without_letters": 0,
        "non_strict_multiple_choice": 0,
        "non_multiple_choice": 0,
    }
    for group in groups:
        for template_type in ["neutral", *bias_types]:
            row = group.get("rows_by_type", {}).get(template_type)
            if not isinstance(row, dict):
                continue
            prompt_messages = row.get("prompt", [])
            if not isinstance(prompt_messages, list) or not prompt_messages:
                continue
            summary["total_prompt_variants"] += 1

            base = row.get("base", {}) or {}
            task_format = str(base.get("task_format", "") or "")
            mc_mode = str(base.get("mc_mode", "") or "")
            if task_format != "multiple_choice":
                summary["non_multiple_choice"] += 1
                continue

            summary["multiple_choice"] += 1
            if mc_mode == MC_MODE_STRICT:
                summary["strict_mc"] += 1
                if bool(str(base.get("letters", "") or "").strip()):
                    summary["strict_mc_choice_scoring"] += 1
                else:
                    summary["strict_mc_without_letters"] += 1
            else:
                summary["non_strict_multiple_choice"] += 1
    return summary


def _log_multiple_choice_mode_summary(summary: Dict[str, int], requested_n_draws: int) -> None:
    total_prompt_variants = int(summary.get("total_prompt_variants", 0) or 0)
    strict_mc_choice_scoring = int(summary.get("strict_mc_choice_scoring", 0) or 0)
    strict_mc_without_letters = int(summary.get("strict_mc_without_letters", 0) or 0)
    non_strict_multiple_choice = int(summary.get("non_strict_multiple_choice", 0) or 0)

    log_status(
        "pipeline.py",
        f"prompt mode summary: total_prompt_variants={total_prompt_variants} "
        f"multiple_choice={int(summary.get('multiple_choice', 0) or 0)} "
        f"strict_mc={int(summary.get('strict_mc', 0) or 0)} "
        f"strict_mc_choice_scoring={strict_mc_choice_scoring} "
        f"strict_mc_without_letters={strict_mc_without_letters} "
        f"non_strict_multiple_choice={non_strict_multiple_choice} "
        f"non_multiple_choice={int(summary.get('non_multiple_choice', 0) or 0)}",
    )

    if strict_mc_choice_scoring > 0 and strict_mc_choice_scoring < total_prompt_variants:
        warn_status(
            "pipeline.py",
            "mixed_sampling_semantics",
            "strict-MC choice scoring covers "
            f"{strict_mc_choice_scoring}/{total_prompt_variants} prompt variants. "
            "This run mixes strict-MC first-token choice scoring with text generation, so "
            "`n_draws`, `temperature`, `top_p`, and `max_new_tokens` do not apply uniformly.",
        )
    if strict_mc_without_letters > 0:
        warn_status(
            "pipeline.py",
            "strict_mc_missing_letters",
            f"{strict_mc_without_letters} strict-MC prompt variants are missing usable choice labels, "
            "so they will fall back to text generation instead of first-token choice scoring.",
        )
    if strict_mc_choice_scoring > 0 and non_strict_multiple_choice > 0:
        warn_status(
            "pipeline.py",
            "mixed_multiple_choice_modes",
            f"{non_strict_multiple_choice} multiple-choice prompt variants use mc_mode != strict_mc while "
            f"{strict_mc_choice_scoring} strict-MC prompt variants use choice scoring. "
            "This can create mixed answer-format and sampling behavior within the same run.",
        )
    if strict_mc_choice_scoring > 0 and strict_mc_choice_scoring < total_prompt_variants and requested_n_draws != 1:
        warn_status(
            "pipeline.py",
            "mixed_draw_semantics",
            f"requested n_draws={requested_n_draws}, but strict-MC choice-scored prompt variants will only "
            "produce 1 draw each while other prompt variants may still use the requested draw count.",
        )


def _log_sampling_plan(
    bias_types: Sequence[str],
    split_expected_keys: Dict[str, Set[tuple[str, str, str, int]]],
    checkpoint_every: int,
    sample_batch_size: int,
    sampling_hash: str,
) -> None:
    log_status(
        "pipeline.py",
        f"sampling plan: sample_batch_size={sample_batch_size} checkpoint_every={checkpoint_every} "
        f"sampling_hash={sampling_hash}",
    )
    for split_name in ("train", "val", "test"):
        counts = _count_expected_by_template(split_expected_keys[split_name], bias_types)
        summary = " ".join(
            f"{template_type}={counts.get(template_type, 0)}"
            for template_type in ["neutral", *bias_types]
        )
        log_status(
            "pipeline.py",
            f"sampling plan split={split_name}: total_expected={len(split_expected_keys[split_name])} {summary}",
        )


def _log_reuse_summary(
    split_expected_keys: Dict[str, Set[tuple[str, str, str, int]]],
    split_records: Dict[str, List[Dict[str, Any]]],
    reuse_enabled: bool,
    cached_source_run: Optional[Path],
) -> None:
    if not reuse_enabled:
        log_status(
            "pipeline.py",
            "reuse strategy: disabled by --no_reuse_sampling_cache/--override_sampling_cache",
        )
    elif cached_source_run is None:
        log_status("pipeline.py", "reuse strategy: no compatible cached sampling run found")
    else:
        log_status("pipeline.py", f"reuse strategy: using compatible cache from {cached_source_run}")

    for split_name in ("train", "val", "test"):
        reused = len(split_records[split_name])
        expected = len(split_expected_keys[split_name])
        remaining = max(0, expected - reused)
        log_status(
            "pipeline.py",
            f"reuse strategy split={split_name}: reused={reused} expected={expected} remaining={remaining}",
        )


def _log_sample_preview(split_name: str, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        log_status("pipeline.py", f"sampling example split={split_name}: no records available")
        return

    example = next((record for record in records if record.get("usable_for_metrics")), records[0])
    log_status(
        "pipeline.py",
        f"sampling example split={split_name} template={example.get('template_type')} "
        f"question='{_preview_text(example.get('question', ''))}' "
        f"response='{_preview_text(example.get('response_raw', ''))}' "
        f"parsed='{_preview_text(example.get('response', ''))}' "
        f"correctness={example.get('correctness')} grading_status={example.get('grading_status')}",
    )


def _log_post_sampling_metrics(records: Sequence[Dict[str, Any]]) -> None:
    overall_usable = sum(1 for record in records if record.get("usable_for_metrics"))
    overall_ambiguous = len(records) - overall_usable
    cap_hits = sum(1 for record in records if record.get("hit_max_new_tokens"))
    no_commitment = sum(
        1
        for record in records
        if str(record.get("task_format", "") or "") == "multiple_choice"
        and str(record.get("commitment_kind", "") or "") in {"", "none"}
    )
    log_status(
        "pipeline.py",
        f"post-sampling metrics: total_records={len(records)} usable={overall_usable} ambiguous={overall_ambiguous} "
        f"cap_hits={cap_hits} no_commitment={no_commitment}",
    )
    if records:
        cap_hit_rate = cap_hits / len(records)
        no_commitment_rate = no_commitment / len(records)
        if cap_hit_rate >= 0.10:
            warn_status(
                "pipeline.py",
                "high_cap_hit_rate",
                f"high cap-hit rate detected ({cap_hit_rate:.1%}); this run may be generation-budget constrained.",
            )
        if no_commitment_rate >= 0.10:
            warn_status(
                "pipeline.py",
                "high_no_commitment_rate",
                f"high no-commitment rate detected ({no_commitment_rate:.1%}); strict MC protocol compliance is poor.",
            )

    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("split", "")), str(record.get("template_type", "")))
        stats = grouped.setdefault(
            key,
            {
                "total": 0,
                "usable": 0,
                "ambiguous": 0,
                "correctness_sum": 0,
                "prompt_t_by_question": {},
                "cap_hits": 0,
                "no_commitment": 0,
            },
        )
        stats["total"] += 1
        if record.get("hit_max_new_tokens"):
            stats["cap_hits"] += 1
        if (
            str(record.get("task_format", "") or "") == "multiple_choice"
            and str(record.get("commitment_kind", "") or "") in {"", "none"}
        ):
            stats["no_commitment"] += 1
        if record.get("usable_for_metrics"):
            stats["usable"] += 1
            stats["correctness_sum"] += int(record["correctness"])
        else:
            stats["ambiguous"] += 1
        t_prompt = record.get("T_prompt")
        if isinstance(t_prompt, (int, float)) and not np.isnan(float(t_prompt)):
            stats["prompt_t_by_question"][str(record.get("question_id", ""))] = float(t_prompt)

    for split_name, template_type in sorted(grouped):
        stats = grouped[(split_name, template_type)]
        mean_correctness = (
            stats["correctness_sum"] / stats["usable"] if stats["usable"] > 0 else float("nan")
        )
        t_values = list(stats["prompt_t_by_question"].values())
        mean_t_prompt = float(np.mean(t_values)) if t_values else float("nan")
        mean_correctness_text = "nan" if np.isnan(mean_correctness) else f"{mean_correctness:.4f}"
        mean_t_prompt_text = "nan" if np.isnan(mean_t_prompt) else f"{mean_t_prompt:.4f}"
        log_status(
            "pipeline.py",
            f"post-sampling metrics split={split_name} template={template_type}: "
            f"total={stats['total']} usable={stats['usable']} ambiguous={stats['ambiguous']} "
            f"cap_hits={stats['cap_hits']} no_commitment={stats['no_commitment']} "
            f"mean_correctness={mean_correctness_text} mean_T_prompt={mean_t_prompt_text}",
        )


def _log_sampling_mode_warnings(records: Sequence[Dict[str, Any]]) -> None:
    strict_mc_choice_rows = 0
    strict_mc_generation_rows = 0
    non_strict_choice_rows = 0
    for record in records:
        if str(record.get("task_format", "") or "") != "multiple_choice":
            continue
        mc_mode = str(record.get("mc_mode", "") or "")
        sampling_mode = str(record.get("sampling_mode", "generation") or "generation")
        if mc_mode == MC_MODE_STRICT:
            if sampling_mode == "choice_probabilities":
                strict_mc_choice_rows += 1
            else:
                strict_mc_generation_rows += 1
        elif sampling_mode == "choice_probabilities":
            non_strict_choice_rows += 1

    if strict_mc_choice_rows or strict_mc_generation_rows or non_strict_choice_rows:
        log_status(
            "pipeline.py",
            f"sampling mode summary: strict_mc_choice_probabilities={strict_mc_choice_rows} "
            f"strict_mc_generation={strict_mc_generation_rows} "
            f"non_strict_choice_probabilities={non_strict_choice_rows}",
        )
    if strict_mc_generation_rows > 0:
        warn_status(
            "pipeline.py",
            "strict_mc_generation_fallback",
            f"{strict_mc_generation_rows} strict-MC sampled rows used text generation instead of "
            "first-token choice scoring. This usually means the prompt metadata was missing usable choice labels.",
        )
    if non_strict_choice_rows > 0:
        warn_status(
            "pipeline.py",
            "non_strict_choice_scoring_detected",
            f"{non_strict_choice_rows} non-strict rows used choice-probability scoring unexpectedly. "
            "Check `mc_mode`, `task_format`, and prompt metadata.",
        )


def _strict_mc_neutral_rows(
    records: Sequence[Dict[str, Any]],
    *,
    require_choice_probabilities: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if str(record.get("template_type", "") or "") != "neutral":
            continue
        if str(record.get("task_format", "") or "") != "multiple_choice":
            continue
        if str(record.get("mc_mode", "") or "") != MC_MODE_STRICT:
            continue
        if not bool(record.get("usable_for_metrics", False)):
            continue
        if require_choice_probabilities and str(record.get("sampling_mode", "") or "") != "choice_probabilities":
            continue
        rows.append(record)
    return rows


def _strict_mc_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if len(text) == 1 and text.isalpha():
        return text
    if len(text) == 3 and text[0] == "(" and text[2] == ")" and text[1].isalpha():
        return text[1]
    return ""


def _strict_mc_selected_letter(record: Dict[str, Any]) -> str:
    selected = _strict_mc_letter(record.get("response"))
    if selected:
        return selected
    return _strict_mc_letter(record.get("response_raw"))


def _strict_mc_choice_probability_map(record: Dict[str, Any]) -> Dict[str, float]:
    probabilities_raw = record.get("choice_probabilities", {})
    if not isinstance(probabilities_raw, dict):
        return {}

    parsed: Dict[str, float] = {}
    for raw_choice, raw_value in probabilities_raw.items():
        choice = _strict_mc_letter(raw_choice)
        if not choice:
            continue
        try:
            probability = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(probability) or probability < 0.0:
            continue
        parsed[choice] = probability
    return dict(sorted(parsed.items()))


def _effective_option_count(probabilities: Dict[str, float]) -> Optional[float]:
    if not probabilities:
        return None
    total_mass = float(sum(probabilities.values()))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        return None

    entropy = 0.0
    for probability in probabilities.values():
        p = float(probability) / total_mass
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(math.exp(entropy))


def _strict_mc_neutral_selected_label_skew_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    neutral_rows = _strict_mc_neutral_rows(records)
    selected_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    for record in neutral_rows:
        selected_label = _strict_mc_selected_letter(record)
        correct_label = _strict_mc_letter(record.get("correct_letter"))
        if selected_label:
            selected_counts[selected_label] += 1
        if correct_label:
            correct_counts[correct_label] += 1

    selected_total = int(sum(selected_counts.values()))
    correct_total = int(sum(correct_counts.values()))
    labels = sorted(set(selected_counts) | set(correct_counts))
    selected_distribution = {
        label: (0.0 if selected_total <= 0 else float(selected_counts[label]) / float(selected_total))
        for label in labels
    }
    correct_distribution = {
        label: (0.0 if correct_total <= 0 else float(correct_counts[label]) / float(correct_total))
        for label in labels
    }

    dominant_label = ""
    dominant_rate = None
    dominant_excess = None
    if selected_counts:
        dominant_label, dominant_count = sorted(selected_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        dominant_rate = 0.0 if selected_total <= 0 else float(dominant_count) / float(selected_total)
        dominant_excess = (
            None
            if dominant_rate is None
            else float(dominant_rate - correct_distribution.get(dominant_label, 0.0))
        )

    total_variation_distance = None
    if labels and selected_total > 0 and correct_total > 0:
        total_variation_distance = 0.5 * sum(
            abs(selected_distribution.get(label, 0.0) - correct_distribution.get(label, 0.0))
            for label in labels
        )

    warning_triggered = bool(
        (
            dominant_rate is not None
            and dominant_excess is not None
            and dominant_rate > STRICT_MC_DOMINANT_SELECTED_LABEL_RATE_WARN
            and dominant_excess > STRICT_MC_DOMINANT_SELECTED_LABEL_EXCESS_WARN
        )
        or (
            total_variation_distance is not None
            and total_variation_distance > STRICT_MC_SELECTED_LABEL_TV_DISTANCE_WARN
        )
    )

    return {
        "neutral_row_count": int(len(neutral_rows)),
        "selected_label_row_count": selected_total,
        "correct_label_row_count": correct_total,
        "selected_label_distribution": selected_distribution,
        "correct_label_distribution": correct_distribution,
        "dominant_selected_label": dominant_label,
        "dominant_selected_label_rate": dominant_rate,
        "dominant_selected_label_excess": dominant_excess,
        "selected_vs_answer_key_tv_distance": total_variation_distance,
        "thresholds": {
            "dominant_selected_label_rate_warn": STRICT_MC_DOMINANT_SELECTED_LABEL_RATE_WARN,
            "dominant_selected_label_excess_warn": STRICT_MC_DOMINANT_SELECTED_LABEL_EXCESS_WARN,
            "selected_label_tv_distance_warn": STRICT_MC_SELECTED_LABEL_TV_DISTANCE_WARN,
        },
        "warning_triggered": warning_triggered,
    }


def _strict_mc_neutral_selected_label_skew_warning(summary: Dict[str, Any]) -> Optional[str]:
    if not summary or not bool(summary.get("warning_triggered", False)):
        return None

    dominant_label = str(summary.get("dominant_selected_label", "") or "").strip().upper()
    if not dominant_label:
        return None

    correct_distribution = dict(summary.get("correct_label_distribution", {}) or {})
    dominant_rate = summary.get("dominant_selected_label_rate")
    answer_key_rate = correct_distribution.get(dominant_label)
    dominant_excess = summary.get("dominant_selected_label_excess")
    tv_distance = summary.get("selected_vs_answer_key_tv_distance")
    row_count = int(summary.get("selected_label_row_count", 0) or 0)
    return (
        f"neutral strict-MC selected-label distribution is skewed toward {dominant_label}: "
        f"q({dominant_label})={float(dominant_rate or 0.0):.1%} vs answer-key "
        f"r({dominant_label})={float(answer_key_rate or 0.0):.1%} "
        f"(excess={float(dominant_excess or 0.0):.1%}, "
        f"TV={float(tv_distance or 0.0):.1%}) across {row_count} usable neutral rows. "
        f"Thresholds: q_max>{STRICT_MC_DOMINANT_SELECTED_LABEL_RATE_WARN:.0%} and excess>"
        f"{STRICT_MC_DOMINANT_SELECTED_LABEL_EXCESS_WARN:.0%}, or TV>"
        f"{STRICT_MC_SELECTED_LABEL_TV_DISTANCE_WARN:.0%}."
    )


def _strict_mc_neutral_choice_distribution_collapse_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    neutral_rows = _strict_mc_neutral_rows(records, require_choice_probabilities=True)
    effective_option_counts: List[float] = []
    high_confidence_rows = 0
    selected_probability_rows = 0

    for record in neutral_rows:
        probability_map = _strict_mc_choice_probability_map(record)
        if not probability_map:
            continue

        effective_options = _effective_option_count(probability_map)
        if effective_options is not None:
            effective_option_counts.append(float(effective_options))

        try:
            selected_probability = float(record.get("choice_probability_selected"))
        except (TypeError, ValueError):
            selected_probability = math.nan
        if not math.isfinite(selected_probability):
            selected_label = _strict_mc_selected_letter(record)
            selected_probability = probability_map.get(selected_label, math.nan)
        if math.isfinite(selected_probability):
            selected_probability_rows += 1
            if selected_probability >= STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_SELECTED_PROB:
                high_confidence_rows += 1

    median_effective_options = (
        float(np.median(np.asarray(effective_option_counts, dtype=float)))
        if effective_option_counts
        else None
    )
    high_confidence_selected_rate = (
        float(high_confidence_rows) / float(selected_probability_rows)
        if selected_probability_rows > 0
        else None
    )
    warning_triggered = bool(
        (
            median_effective_options is not None
            and median_effective_options < STRICT_MC_COLLAPSE_MEDIAN_EFFECTIVE_OPTIONS_WARN
        )
        or (
            high_confidence_selected_rate is not None
            and high_confidence_selected_rate > STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_RATE_WARN
        )
    )

    return {
        "neutral_choice_probability_row_count": int(len(neutral_rows)),
        "effective_option_row_count": int(len(effective_option_counts)),
        "selected_probability_row_count": int(selected_probability_rows),
        "median_effective_options": median_effective_options,
        "high_confidence_selected_rate": high_confidence_selected_rate,
        "high_confidence_selected_prob_threshold": STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_SELECTED_PROB,
        "thresholds": {
            "median_effective_options_warn": STRICT_MC_COLLAPSE_MEDIAN_EFFECTIVE_OPTIONS_WARN,
            "high_confidence_selected_rate_warn": STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_RATE_WARN,
        },
        "warning_triggered": warning_triggered,
    }


def _strict_mc_neutral_choice_distribution_collapse_warning(summary: Dict[str, Any]) -> Optional[str]:
    if not summary or not bool(summary.get("warning_triggered", False)):
        return None

    row_count = int(summary.get("selected_probability_row_count", 0) or 0)
    median_effective_options = summary.get("median_effective_options")
    high_confidence_selected_rate = summary.get("high_confidence_selected_rate")
    median_text = "n/a" if median_effective_options is None else f"{float(median_effective_options):.2f}"
    high_confidence_text = (
        "n/a"
        if high_confidence_selected_rate is None
        else f"{float(high_confidence_selected_rate):.1%}"
    )
    return (
        "neutral strict-MC choice distribution appears collapsed across "
        f"{row_count} neutral choice-probability rows: "
        f"median(N_eff)={median_text} and "
        f"mean(P(selected)>={STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_SELECTED_PROB:.2f})="
        f"{high_confidence_text}. "
        f"Thresholds: median(N_eff)<{STRICT_MC_COLLAPSE_MEDIAN_EFFECTIVE_OPTIONS_WARN:.2f} "
        f"or high-confidence rate>{STRICT_MC_COLLAPSE_HIGH_CONFIDENCE_RATE_WARN:.0%}."
    )


def _strict_mc_quality_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    strict_records = [
        record
        for record in records
        if str(record.get("task_format", "") or "") == "multiple_choice"
        and str(record.get("mc_mode", "") or "") == MC_MODE_STRICT
    ]
    if not strict_records:
        return {}

    total = len(strict_records)
    committed = sum(1 for record in strict_records if str(record.get("commitment_kind", "") or "") == "letter")
    starts_with_answer = sum(1 for record in strict_records if bool(record.get("starts_with_answer_prefix", False)))
    cap_hits = sum(1 for record in strict_records if bool(record.get("hit_max_new_tokens", False)))
    explicit_parse_failures = sum(
        1
        for record in strict_records
        if bool(record.get("starts_with_answer_prefix", False))
        and str(record.get("commitment_kind", "") or "") in {"", "none"}
    )
    exact_format_rows = sum(
        1 for record in strict_records if bool(record.get("strict_format_exact", False))
    )
    multiple_answer_marker_rows = sum(
        1 for record in strict_records if bool(record.get("multiple_answer_markers", False))
    )
    by_template: Dict[str, Dict[str, float]] = {}
    for template_type in sorted({str(record.get("template_type", "")) for record in strict_records}):
        template_records = [record for record in strict_records if str(record.get("template_type", "")) == template_type]
        template_total = len(template_records)
        by_template[template_type] = {
            "total": template_total,
            "committed_rate": 0.0
            if template_total == 0
            else sum(
                1
                for record in template_records
                if str(record.get("commitment_kind", "") or "") == "letter"
            )
            / template_total,
            "starts_with_answer_rate": 0.0
            if template_total == 0
            else sum(1 for record in template_records if bool(record.get("starts_with_answer_prefix", False)))
            / template_total,
            "cap_hit_rate": 0.0
            if template_total == 0
            else sum(1 for record in template_records if bool(record.get("hit_max_new_tokens", False)))
            / template_total,
            "exact_format_rate": 0.0
            if template_total == 0
            else sum(1 for record in template_records if bool(record.get("strict_format_exact", False)))
            / template_total,
            "multiple_answer_marker_rows": sum(
                1 for record in template_records if bool(record.get("multiple_answer_markers", False))
            ),
        }

    neutral_rate = by_template.get("neutral", {}).get("starts_with_answer_rate")
    max_bias_gap = 0.0
    if neutral_rate is not None:
        for template_type, stats in by_template.items():
            if template_type == "neutral":
                continue
            max_bias_gap = max(max_bias_gap, abs(neutral_rate - stats["starts_with_answer_rate"]))

    neutral_selected_label_skew = _strict_mc_neutral_selected_label_skew_summary(records)
    neutral_choice_distribution = _strict_mc_neutral_choice_distribution_collapse_summary(records)

    return {
        "total": total,
        "commitment_rate": committed / total,
        "starts_with_answer_rate": starts_with_answer / total,
        "cap_hit_rate": cap_hits / total,
        "explicit_parse_failures": explicit_parse_failures,
        "exact_format_rate": exact_format_rows / total,
        "multiple_answer_marker_rows": multiple_answer_marker_rows,
        "max_neutral_bias_answer_gap": max_bias_gap,
        "neutral_selected_label_skew": neutral_selected_label_skew,
        "neutral_choice_distribution_collapse": neutral_choice_distribution,
        "by_template": by_template,
    }


def _log_strict_mc_quality_summary(summary: Dict[str, Any], issues: Optional[Sequence[str]] = None) -> None:
    if not summary:
        return
    summary_logger = ok_status if not list(issues or []) else log_status
    summary_logger(
        "pipeline.py",
        "strict MC quality: "
        f"commitment_rate={summary['commitment_rate']:.1%} "
        f"starts_with_answer_rate={summary['starts_with_answer_rate']:.1%} "
        f"cap_hit_rate={summary['cap_hit_rate']:.1%} "
        f"explicit_parse_failures={summary['explicit_parse_failures']} "
        f"exact_format_rate={summary['exact_format_rate']:.1%} "
        f"multiple_answer_marker_rows={summary['multiple_answer_marker_rows']} "
        f"max_neutral_bias_answer_gap={summary['max_neutral_bias_answer_gap']:.1%}",
    )
    for template_type, stats in sorted(summary.get("by_template", {}).items()):
        log_status(
            "pipeline.py",
            f"strict MC quality template={template_type}: total={int(stats['total'])} "
            f"commitment_rate={stats['committed_rate']:.1%} "
            f"starts_with_answer_rate={stats['starts_with_answer_rate']:.1%} "
            f"cap_hit_rate={stats['cap_hit_rate']:.1%} "
            f"exact_format_rate={stats['exact_format_rate']:.1%} "
            f"multiple_answer_marker_rows={int(stats['multiple_answer_marker_rows'])}",
        )
    selected_label_skew = summary.get("neutral_selected_label_skew", {})
    if selected_label_skew:
        dominant_label = str(selected_label_skew.get("dominant_selected_label", "") or "").strip().upper() or "n/a"
        answer_key_distribution = dict(selected_label_skew.get("correct_label_distribution", {}) or {})
        log_status(
            "pipeline.py",
            "strict MC neutral selected-label skew: "
            f"rows={int(selected_label_skew.get('selected_label_row_count', 0) or 0)} "
            f"dominant_label={dominant_label} "
            f"q_max={float(selected_label_skew.get('dominant_selected_label_rate') or 0.0):.1%} "
            f"r_max={float(answer_key_distribution.get(dominant_label, 0.0) or 0.0):.1%} "
            f"excess={float(selected_label_skew.get('dominant_selected_label_excess') or 0.0):.1%} "
            f"tv={float(selected_label_skew.get('selected_vs_answer_key_tv_distance') or 0.0):.1%}",
        )
    choice_distribution = summary.get("neutral_choice_distribution_collapse", {})
    if choice_distribution:
        median_effective_options = choice_distribution.get("median_effective_options")
        high_confidence_selected_rate = choice_distribution.get("high_confidence_selected_rate")
        log_status(
            "pipeline.py",
            "strict MC neutral choice distribution: "
            f"rows={int(choice_distribution.get('selected_probability_row_count', 0) or 0)} "
            f"median_effective_options="
            f"{'n/a' if median_effective_options is None else f'{float(median_effective_options):.2f}'} "
            f"high_confidence_selected_rate="
            f"{'n/a' if high_confidence_selected_rate is None else f'{float(high_confidence_selected_rate):.1%}'}",
        )


def _strict_mc_quality_issues(summary: Dict[str, Any]) -> List[str]:
    if not summary:
        return []
    issues: List[str] = []
    if summary["explicit_parse_failures"] > STRICT_MC_MAX_EXPLICIT_PARSE_FAILURES:
        issues.append(
            f"explicit_answer_parse_failures={summary['explicit_parse_failures']} "
            f"> {STRICT_MC_MAX_EXPLICIT_PARSE_FAILURES}"
        )
    if summary["commitment_rate"] < STRICT_MC_MIN_COMMITMENT_RATE:
        issues.append(
            f"commitment_rate={summary['commitment_rate']:.1%} "
            f"< {STRICT_MC_MIN_COMMITMENT_RATE:.0%}"
        )
    if summary["starts_with_answer_rate"] < STRICT_MC_MIN_STARTS_WITH_ANSWER_RATE:
        issues.append(
            f"starts_with_answer_rate={summary['starts_with_answer_rate']:.1%} "
            f"< {STRICT_MC_MIN_STARTS_WITH_ANSWER_RATE:.0%}"
        )
    if summary["cap_hit_rate"] > STRICT_MC_MAX_CAP_HIT_RATE:
        issues.append(
            f"cap_hit_rate={summary['cap_hit_rate']:.1%} "
            f"> {STRICT_MC_MAX_CAP_HIT_RATE:.0%}"
        )
    if summary["exact_format_rate"] < STRICT_MC_MIN_EXACT_FORMAT_RATE:
        issues.append(
            f"exact_format_rate={summary['exact_format_rate']:.1%} "
            f"< {STRICT_MC_MIN_EXACT_FORMAT_RATE:.0%}"
        )
    if summary["multiple_answer_marker_rows"] > STRICT_MC_MAX_MULTIPLE_ANSWER_MARKER_ROWS:
        issues.append(
            f"multiple_answer_marker_rows={summary['multiple_answer_marker_rows']} "
            f"> {STRICT_MC_MAX_MULTIPLE_ANSWER_MARKER_ROWS}"
        )
    if summary["max_neutral_bias_answer_gap"] > 0.20:
        issues.append(
            f"neutral_bias_starts_with_answer_gap={summary['max_neutral_bias_answer_gap']:.1%} > 20%"
        )
    return issues


def _strict_mc_neutral_below_chance_warning(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    neutral_rows: List[tuple[float, int]] = []
    for record in records:
        if str(record.get("template_type", "") or "") != "neutral":
            continue
        if str(record.get("task_format", "") or "") != "multiple_choice":
            continue
        if str(record.get("mc_mode", "") or "") != MC_MODE_STRICT:
            continue
        if not bool(record.get("usable_for_metrics", False)):
            continue

        answers_list = record.get("answers_list")
        if isinstance(answers_list, list) and answers_list:
            choice_count = len(answers_list)
        else:
            letters = [char for char in str(record.get("letters", "") or "") if not char.isspace()]
            choice_count = len(letters)
        if choice_count <= 0:
            continue

        try:
            correctness = float(record.get("correctness"))
        except (TypeError, ValueError):
            continue
        neutral_rows.append((correctness, choice_count))

    if not neutral_rows:
        return None

    neutral_accuracy = sum(correctness for correctness, _ in neutral_rows) / len(neutral_rows)
    chance_baseline = sum(1.0 / choice_count for _, choice_count in neutral_rows) / len(neutral_rows)
    if neutral_accuracy >= chance_baseline:
        return None

    choice_counts = sorted({choice_count for _, choice_count in neutral_rows})
    if len(choice_counts) == 1:
        baseline_text = f"1/{choice_counts[0]} = {chance_baseline:.1%}"
    else:
        baseline_text = (
            f"mean(1/num_choices) = {chance_baseline:.1%} "
            f"across choice counts {choice_counts}"
        )

    return (
        f"neutral strict-MC accuracy={neutral_accuracy:.1%} is below the random-choice baseline "
        f"{baseline_text} across {len(neutral_rows)} usable neutral rows."
    )


def _finalize_warning_summary(run_dir: Path) -> Optional[Path]:
    summary_payload = build_warning_summary_payload()
    total_warnings = int(summary_payload.get("total_warnings", 0) or 0)
    if total_warnings <= 0:
        ok_status("pipeline.py", "warnings summary: no warnings emitted")
        return None

    summary_path = preferred_run_artifact_path(run_dir, "warnings_summary")
    write_json_atomic(summary_path, summary_payload)
    log_status(
        "pipeline.py",
        "warnings summary: "
        f"total={total_warnings} "
        f"unique_codes={int(summary_payload.get('unique_warning_codes', 0) or 0)} "
        f"unique_sources={int(summary_payload.get('unique_sources', 0) or 0)} "
        f"path={summary_path}",
    )
    return summary_path


def run_pipeline(args) -> None:
    if args.sample_batch_size < 1:
        raise ValueError(f"--sample_batch_size must be >= 1, got {args.sample_batch_size}")
    if args.sampling_checkpoint_every < 0:
        raise ValueError(
            f"--sampling_checkpoint_every must be >= 0, got {args.sampling_checkpoint_every}"
        )
    if args.smoke_test and args.max_questions is None:
        args.max_questions = args.smoke_questions

    run_dir = make_run_dir(args.out_dir, args.model, args.run_name)
    run_log_path = preferred_run_artifact_path(run_dir, "run_log")
    warning_log_path = preferred_run_artifact_path(run_dir, "warnings_log")
    configure_run_logging(run_log_path, warning_log_path=warning_log_path)
    lock_path = run_lock_path(run_dir)
    stage_bar: Optional[tqdm] = None
    stage_count = 8
    run_started_at_utc = utc_now_iso()
    run_started_perf = perf_counter()
    stage_timing_rows: List[Dict[str, Any]] = []
    current_stage_timing: Optional[Dict[str, Any]] = None

    def runtime_timing_snapshot(status: str) -> Dict[str, Any]:
        now_utc = utc_now_iso()
        now_perf = perf_counter()
        stages = list(stage_timing_rows)
        if current_stage_timing is not None:
            stages.append(
                {
                    "stage_index": int(current_stage_timing["stage_index"]),
                    "stage_name": str(current_stage_timing["stage_name"]),
                    "stage_status": "in_progress",
                    "started_at_utc": current_stage_timing["started_at_utc"],
                    "ended_at_utc": None,
                    "duration_seconds": max(
                        0.0,
                        float(now_perf - float(current_stage_timing["started_perf"])),
                    ),
                }
            )
        return {
            "status": str(status),
            "started_at_utc": run_started_at_utc,
            "snapshot_at_utc": now_utc,
            "total_elapsed_seconds": max(0.0, float(now_perf - run_started_perf)),
            "stage_count": int(stage_count),
            "stages": stages,
        }

    def begin_stage(index: int, message: str) -> None:
        nonlocal current_stage_timing
        current_stage_timing = {
            "stage_index": int(index),
            "stage_name": str(message),
            "started_at_utc": utc_now_iso(),
            "started_perf": perf_counter(),
        }
        if stage_bar is not None:
            stage_bar.set_description(tqdm_desc("pipeline.py", f"stage {index}/{stage_count} {message}"))
        log_status("pipeline.py", f"stage {index}/{stage_count}: {message}")

    def finish_stage() -> None:
        nonlocal current_stage_timing
        if current_stage_timing is not None:
            ended_at_utc = utc_now_iso()
            ended_perf = perf_counter()
            stage_timing_rows.append(
                {
                    "stage_index": int(current_stage_timing["stage_index"]),
                    "stage_name": str(current_stage_timing["stage_name"]),
                    "stage_status": "completed",
                    "started_at_utc": current_stage_timing["started_at_utc"],
                    "ended_at_utc": ended_at_utc,
                    "duration_seconds": max(
                        0.0,
                        float(ended_perf - float(current_stage_timing["started_perf"])),
                    ),
                }
            )
            current_stage_timing = None
        if stage_bar is not None:
            stage_bar.update(1)

    run_status = "failed"
    run_error: Optional[str] = None
    strict_mc_quality_report: Dict[str, Any] = {}
    strict_mc_quality_failures: List[str] = []
    strict_mc_behavior_warnings: List[Dict[str, str]] = []
    sampling_integrity_summary: Dict[str, Any] = {}
    sampling_integrity_summary_path = preferred_run_artifact_path(run_dir, "sampling_integrity_summary")
    probe_candidate_score_rows: List[Dict[str, Any]] = []
    saved_paths: Dict[str, Path] = {}
    try:
        assert_resume_compatible(run_dir, args)
        acquire_run_lock(lock_path, run_dir)
        log_status("pipeline.py", f"run directory ready: {run_dir}")
        log_status("pipeline.py", f"run lock acquired: {lock_path}")
        write_run_status(run_dir, args=args, status="running", lock_path=lock_path)

        stage_bar = tqdm(
            total=stage_count,
            desc=tqdm_desc("pipeline.py", "pipeline stage progress"),
            unit="stage",
        )

        begin_stage(1, "parsed arguments and execution plan")
        _apply_model_backend_overrides(args, resolve_llm_capabilities(args.model))
        for line in _format_parsed_argument_lines(args):
            log_status("pipeline.py", line)
        _warn_strict_mc_temperature_bookkeeping(args)
        planned_bias_types = resolve_bias_types(args.bias_types)
        resolved_ays_mc_datasets = resolve_ays_mc_datasets(args.ays_mc_datasets)
        args.ays_mc_datasets = resolved_ays_mc_datasets
        log_status(
            "pipeline.py",
            f"execution plan: model={args.model} benchmark_source={args.benchmark_source} "
            f"bias_types={planned_bias_types} "
            f"dataset_name={args.dataset_name} "
            f"draws={args.n_draws} temperature={args.temperature} top_p={args.top_p} "
            f"max_new_tokens={args.max_new_tokens} smoke_test={args.smoke_test} "
            f"sampling_only={bool(getattr(args, 'sampling_only', False))}",
        )
        finish_stage()

        begin_stage(2, "dataset loading, grouping, and split planning")
        random.seed(args.seed)
        np.random.seed(args.seed)
        if str(getattr(args, "model_backend", "") or "") != "openai":
            import torch

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        load_env_file(args.env_file)
        hf_cache_dir = resolve_hf_cache_dir(args.hf_cache_dir)
        if hf_cache_dir:
            Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ["HF_HUB_CACHE"] = hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
            log_status("pipeline.py", f"HF cache dir resolved to {hf_cache_dir}")
        else:
            log_status("pipeline.py", "HF cache dir not set; libraries may fallback to ~/.cache")
        args.requested_device = str(args.device)
        if str(getattr(args, "model_backend", "") or "") == "openai" and str(args.device) == "auto":
            device = "cpu"
        else:
            device = resolve_device(args.device)
        args.resolved_device = device
        log_status("pipeline.py", f"resolved device: requested={args.device} actual={device}")

        data_files = ensure_sycophancy_eval_cached(
            data_dir=args.data_dir,
            repo_id=args.sycophancy_repo,
            force_download=args.force_download_sycophancy,
        )
        input_path = data_files[args.input_jsonl]
        log_status(
            "pipeline.py",
            f"loading dataset: input_jsonl={args.input_jsonl} path={input_path} repo={args.sycophancy_repo}",
        )
        rows_raw = read_jsonl(input_path)
        if args.benchmark_source == "ays_mc_single_turn":
            external_rows = load_external_ays_mc_rows(
                data_dir=args.data_dir,
                selected_ays_mc_datasets=resolved_ays_mc_datasets,
                force_download=args.force_download_sycophancy,
            )
            if external_rows:
                rows_raw = [*rows_raw, *external_rows]
                log_status(
                    "pipeline.py",
                    f"loaded external AYS MC rows: added_rows={len(external_rows)} "
                    f"ays_mc_datasets={resolved_ays_mc_datasets}",
                )

        prepared_rows = prepare_benchmark_rows(
            benchmark_source=args.benchmark_source,
            rows=rows_raw,
            input_jsonl=args.input_jsonl,
            selected_bias_types=planned_bias_types,
            selected_ays_mc_datasets=resolved_ays_mc_datasets,
            instruction_policy=args.instruction_policy,
            mc_mode=args.mc_mode,
        )
        if args.benchmark_source == "ays_mc_single_turn":
            log_status(
                "pipeline.py",
                f"materialized AYS MC rows: source_rows={len(rows_raw)} derived_rows={len(prepared_rows)} "
                f"ays_mc_datasets={resolved_ays_mc_datasets} "
                f"instruction_policy={args.instruction_policy} mc_mode={args.mc_mode}",
            )

        rows = deduplicate_rows(prepared_rows)
        available_dataset_names = unique_dataset_names(rows)
        wanted_dataset_name = str(getattr(args, "dataset_name", "all") or "all").strip() or "all"
        if wanted_dataset_name.lower() != "all" and not any(
            dataset.lower() == wanted_dataset_name.lower() for dataset in available_dataset_names
        ):
            raise ValueError(
                f"--dataset_name={wanted_dataset_name!r} did not match any dataset in {input_path}. "
                f"Available datasets: {available_dataset_names}"
            )

        groups = build_question_groups(
            rows,
            selected_bias_types=planned_bias_types,
            selected_dataset_name=wanted_dataset_name,
        )
        log_status(
            "pipeline.py",
            f"dataset stats: raw_rows={len(rows_raw)} prepared_rows={len(prepared_rows)} dedup_rows={len(rows)} "
            f"valid_groups={len(groups)} dataset_name={wanted_dataset_name} "
            f"available_datasets={available_dataset_names} bias_types={planned_bias_types}",
        )

        if not groups:
            raise ValueError(
                f"No complete question groups found for dataset_name={wanted_dataset_name!r} "
                f"with bias_types={planned_bias_types}."
            )

        if args.max_questions is not None:
            rng = random.Random(args.split_seed)
            rng.shuffle(groups)
            groups = groups[: args.max_questions]
            log_status(
                "pipeline.py",
                f"dataset restricted by max_questions={args.max_questions}: groups={len(groups)}",
            )

        _log_group_example(groups, planned_bias_types)
        args.requested_n_draws = int(args.n_draws)
        _log_multiple_choice_mode_summary(
            _multiple_choice_mode_summary(groups, planned_bias_types),
            requested_n_draws=args.requested_n_draws,
        )
        any_choice_scoring, all_choice_scoring = _choice_scoring_coverage(groups, planned_bias_types)
        model_capabilities = resolve_llm_capabilities(args.model)
        args.strict_mc_choice_scoring = bool(any_choice_scoring and model_capabilities.supports_choice_scoring)
        if any_choice_scoring and not model_capabilities.supports_choice_scoring:
            log_status(
                "pipeline.py",
                f"backend={args.model_backend} does not support exact strict-MC choice scoring; "
                "sampling will fall back to text generation for strict-MC prompts.",
            )
        if args.strict_mc_choice_scoring and all_choice_scoring and args.n_draws != 1:
            log_status(
                "pipeline.py",
                f"strict MC choice scoring active for all prompts; overriding n_draws from {args.n_draws} to 1",
            )
            args.n_draws = 1

        _warn_sampling_only_split_expectations(args)
        train_groups, val_groups, test_groups = split_groups_train_val_test(
            groups,
            test_frac=args.test_frac,
            val_frac=args.probe_val_frac,
            seed=args.split_seed,
        )
        log_status(
            "pipeline.py",
            f"dataset split: train_questions={len(train_groups)} "
            f"val_questions={len(val_groups)} test_questions={len(test_groups)} "
            f"split_seed={args.split_seed}",
        )
        finish_stage()

        begin_stage(3, "sampling plan and checkpoint layout")

        expected_train_keys = enumerate_expected_sample_keys(
            train_groups,
            split_name="train",
            bias_types=planned_bias_types,
            n_draws=args.n_draws,
        )
        expected_val_keys = enumerate_expected_sample_keys(
            val_groups,
            split_name="val",
            bias_types=planned_bias_types,
            n_draws=args.n_draws,
        )
        expected_test_keys = enumerate_expected_sample_keys(
            test_groups,
            split_name="test",
            bias_types=planned_bias_types,
            n_draws=args.n_draws,
        )
        expected_all_keys = expected_train_keys | expected_val_keys | expected_test_keys
        expected_total_records = len(expected_all_keys)

        sampling_spec = build_sampling_spec(
            args=args,
            bias_types=planned_bias_types,
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
            expected_train=len(expected_train_keys),
            expected_val=len(expected_val_keys),
            expected_test=len(expected_test_keys),
        )
        sampling_hash = sampling_spec_hash(sampling_spec)
        sampling_records_path = preferred_run_artifact_path(run_dir, "sampling_records")
        sampling_manifest_path = preferred_run_artifact_path(run_dir, "sampling_manifest")
        split_expected_keys = {
            "train": expected_train_keys,
            "val": expected_val_keys,
            "test": expected_test_keys,
        }
        _log_sampling_plan(
            planned_bias_types,
            split_expected_keys,
            checkpoint_every=args.sampling_checkpoint_every,
            sample_batch_size=args.sample_batch_size,
            sampling_hash=sampling_hash[:12],
        )
        finish_stage()

        begin_stage(4, "sampling cache reuse strategy")
        cached_source_run: Optional[Path] = None
        cached_records: List[Dict[str, Any]] = []
        if not args.no_reuse_sampling_cache:
            candidate = load_sampling_cache_candidate(
                out_dir=args.out_dir,
                model_name=args.model,
                sampling_hash=sampling_hash,
                exclude_run_dir=run_dir,
            )
            if candidate is not None:
                cached_source_run = candidate["run_dir"]
                cached_records_raw = read_jsonl(str(candidate["records_path"]))
                cached_records = normalize_sample_records(cached_records_raw, expected_all_keys)
                log_status(
                    "pipeline.py",
                    f"loaded reusable sampling cache from {cached_source_run}: "
                    f"records={len(cached_records)}/{expected_total_records}",
                )
        split_groups_map = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        }
        split_records: Dict[str, List[Dict[str, Any]]] = {
            split_name: sort_sample_records([r for r in cached_records if r.get("split") == split_name])
            for split_name in ("train", "val", "test")
        }
        for split_name in ("train", "val", "test"):
            split_records[split_name] = refresh_sample_records_for_groups(
                split_records[split_name],
                split_groups_map[split_name],
                split_name=split_name,
            )
        split_sampling_stats: Dict[str, Dict[str, int]] = {
            split_name: {
                "split": split_name,
                "expected_records": len(split_expected_keys[split_name]),
                "reused_records": len(split_records[split_name]),
                "generated_records": 0,
                "total_records": len(split_records[split_name]),
            }
            for split_name in ("train", "val", "test")
        }
        _log_reuse_summary(
            split_expected_keys,
            split_records,
            reuse_enabled=not args.no_reuse_sampling_cache,
            cached_source_run=cached_source_run,
        )

        persist_sampling_state(
            stage="sampling_start",
            split_states=split_records,
            split_stats=split_sampling_stats,
            expected_all_keys=expected_all_keys,
            expected_total_records=expected_total_records,
            sampling_records_path=sampling_records_path,
            sampling_manifest_path=sampling_manifest_path,
            sampling_hash=sampling_hash,
            sampling_spec=sampling_spec,
            cached_source_run=cached_source_run,
        )
        finish_stage()

        begin_stage(5, "sampling responses with progress and examples")
        llm = load_llm(
            args.model,
            device=device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=hf_cache_dir,
        )

        if all(
            len(split_records[split_name]) >= len(split_expected_keys[split_name])
            for split_name in ("train", "val", "test")
        ):
            log_status("pipeline.py", "sampling generation skipped: full sampling cache hit")
            for split_name in ("train", "val", "test"):
                _log_sample_preview(split_name, split_records[split_name])
        else:

            def make_progress_cb(split_name: str):
                def _progress_cb(
                    current_records: List[Dict[str, Any]],
                    stats: Dict[str, int],
                ) -> None:
                    split_records[split_name] = current_records
                    split_sampling_stats[split_name] = dict(stats)
                    persist_sampling_state(
                        stage=f"sampling_{split_name}_in_progress",
                        split_states=split_records,
                        split_stats=split_sampling_stats,
                        expected_all_keys=expected_all_keys,
                        expected_total_records=expected_total_records,
                        sampling_records_path=sampling_records_path,
                        sampling_manifest_path=sampling_manifest_path,
                        sampling_hash=sampling_hash,
                        sampling_spec=sampling_spec,
                        cached_source_run=cached_source_run,
                    )
                    expected = len(split_expected_keys[split_name])
                    remaining = max(0, expected - len(current_records))
                    log_status(
                        "pipeline.py",
                        f"sampling progress split={split_name}: total={len(current_records)}/{expected} "
                        f"generated={stats.get('generated_records', 0)} "
                        f"reused={stats.get('reused_records', 0)} remaining={remaining}",
                    )

                return _progress_cb

            for split_name in ("train", "val", "test"):
                split_records[split_name], split_sampling_stats[split_name] = sample_records_for_groups(
                    llm=llm,
                    groups=split_groups_map[split_name],
                    split_name=split_name,
                    bias_types=planned_bias_types,
                    n_draws=args.n_draws,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    sample_batch_size=args.sample_batch_size,
                    existing_records=split_records[split_name],
                    checkpoint_every=args.sampling_checkpoint_every,
                    progress_callback=make_progress_cb(split_name),
                    start_id=_next_record_id(
                        split_records["train"],
                        split_records["val"],
                        split_records["test"],
                    ),
                )
                _log_sample_preview(split_name, split_records[split_name])

        persist_sampling_state(
            stage="sampling_complete",
            split_states=split_records,
            split_stats=split_sampling_stats,
            expected_all_keys=expected_all_keys,
            expected_total_records=expected_total_records,
            sampling_records_path=sampling_records_path,
            sampling_manifest_path=sampling_manifest_path,
            sampling_hash=sampling_hash,
            sampling_spec=sampling_spec,
            cached_source_run=cached_source_run,
        )

        train_records = split_records["train"]
        val_records = split_records["val"]
        test_records = split_records["test"]
        all_records = train_records + val_records + test_records
        if len(all_records) < expected_total_records:
            warn_status(
                "pipeline.py",
                "incomplete_sampling_coverage",
                f"sampled records are incomplete: got={len(all_records)} expected={expected_total_records}",
            )

        add_empirical_t(all_records)
        log_status(
            "pipeline.py",
            f"sampling results: train_records={len(train_records)} val_records={len(val_records)} "
            f"test_records={len(test_records)} generated_train={split_sampling_stats['train'].get('generated_records', 0)} "
            f"generated_val={split_sampling_stats['val'].get('generated_records', 0)} "
            f"generated_test={split_sampling_stats['test'].get('generated_records', 0)}",
        )
        _log_sampling_mode_warnings(all_records)
        finish_stage()

        begin_stage(6, "post-sampling prompt metrics")
        _log_post_sampling_metrics(all_records)
        sampling_integrity_summary = build_sampling_integrity_summary(all_records)
        log_sampling_integrity_summary(sampling_integrity_summary)
        sampling_integrity_summary_path = save_sampling_integrity_summary(
            run_dir=run_dir,
            sampling_integrity_summary=sampling_integrity_summary,
        )
        strict_mc_quality_report = _strict_mc_quality_summary(all_records)
        strict_mc_quality_failures = _strict_mc_quality_issues(strict_mc_quality_report)
        _log_strict_mc_quality_summary(strict_mc_quality_report, issues=strict_mc_quality_failures)
        dominant_selected_label_skew_warning = _strict_mc_neutral_selected_label_skew_warning(
            strict_mc_quality_report.get("neutral_selected_label_skew", {})
        )
        if dominant_selected_label_skew_warning is not None:
            strict_mc_behavior_warnings.append(
                {
                    "code": "dominant_selected_label_skew",
                    "message": dominant_selected_label_skew_warning,
                }
            )
        choice_distribution_collapse_warning = _strict_mc_neutral_choice_distribution_collapse_warning(
            strict_mc_quality_report.get("neutral_choice_distribution_collapse", {})
        )
        if choice_distribution_collapse_warning is not None:
            strict_mc_behavior_warnings.append(
                {
                    "code": "choice_distribution_collapse",
                    "message": choice_distribution_collapse_warning,
                }
            )
        for issue in strict_mc_quality_failures:
            warn_status("pipeline.py", "strict_mc_quality_gate", issue)
        for warning in strict_mc_behavior_warnings:
            warn_status("pipeline.py", warning["code"], warning["message"])
        finish_stage()

        begin_stage(7, "probe selection, training, and scoring")
        probes_meta: Dict[str, Any] = {}
        if strict_mc_quality_report:
            probes_meta["strict_mc_quality"] = {
                "summary": strict_mc_quality_report,
                "issues": strict_mc_quality_failures,
                "warnings": strict_mc_behavior_warnings,
                "status": "failed" if strict_mc_quality_failures else "passed",
            }

        if bool(getattr(args, "sampling_only", False)):
            log_status(
                "pipeline.py",
                "sampling-only mode enabled; skipping probe selection, training, and scoring",
            )
            probes_meta["probe_training_status"] = "skipped_by_sampling_only"
            probes_meta["probe_construction"] = str(args.probe_construction)
            probes_meta["probe_example_weighting"] = str(args.probe_example_weighting)
            probes_meta["probe_candidate_score_rows"] = 0
        else:
            model, tokenizer = llm.get_model_and_tokenizer()

            if args.smoke_test and strict_mc_quality_failures:
                warn_status(
                    "pipeline.py",
                    "probe_training_skipped_due_to_strict_mc_quality",
                    "skipping probe training because the strict MC quality gate failed for this smoke run",
                )
                all_probes_dir = preferred_run_artifact_path(run_dir, "all_probes_dir")
                chosen_probe_dir = preferred_run_artifact_path(run_dir, "chosen_probe_dir")
                all_probes_manifest_path = write_probe_group_manifest(
                    group_dir=all_probes_dir,
                    artifact_group="all_probes",
                    probe_summaries={},
                )
                chosen_probe_manifest_path = write_probe_group_manifest(
                    group_dir=chosen_probe_dir,
                    artifact_group="chosen_probe",
                    probe_summaries={},
                )
                probes_meta["probe_training_status"] = "skipped_due_to_strict_mc_quality"
                probes_meta["all_probes_dir"] = str(all_probes_dir)
                probes_meta["chosen_probe_dir"] = str(chosen_probe_dir)
                probes_meta["all_probes_manifest"] = str(all_probes_manifest_path)
                probes_meta["chosen_probe_manifest"] = str(chosen_probe_manifest_path)
                probes_meta["probe_construction"] = str(args.probe_construction)
                probes_meta["probe_example_weighting"] = str(args.probe_example_weighting)
                probes_meta["probe_candidate_score_rows"] = 0
            else:
                n_layers = int(getattr(model.config, "num_hidden_layers", args.probe_layer_max))
                layer_min = max(1, args.probe_layer_min)
                layer_max = min(args.probe_layer_max, n_layers)
                layer_grid = list(range(layer_min, layer_max + 1))
                log_status("pipeline.py", f"probe layer grid: {layer_min}..{layer_max} (num_layers={len(layer_grid)})")

                feature_source_spec = {
                    "probe_feature_mode": args.probe_feature_mode,
                    "record_field": "response_raw",
                    "fallback_record_field": "response",
                    "token_position": "last_token_of_full_sampled_completion",
                }

                probe_record_sets = build_probe_record_sets(
                    train_records=train_records,
                    val_records=val_records,
                    test_records=test_records,
                    all_records=all_records,
                    bias_types=planned_bias_types,
                    probe_construction=args.probe_construction,
                    probe_example_weighting=args.probe_example_weighting,
                )
                all_probe_group_summary: Dict[str, Dict[str, Any]] = {}
                chosen_probe_group_summary: Dict[str, Dict[str, Any]] = {}

                def _run_probe_family_artifacts(probe_bundle: Dict[str, Any]) -> Dict[str, Any]:
                    log_status(
                        "pipeline.py",
                        f"probe selection {probe_bundle['desc']}: train_records={len(probe_bundle['train_records'])} "
                        f"val_records={len(probe_bundle['val_records'])} test_records={len(probe_bundle['test_records'])} "
                        f"construction={probe_bundle['probe_construction']} "
                        f"weighting={probe_bundle['probe_example_weighting']}",
                    )

                    eval_cache = prepare_probe_eval_cache(
                        model=model,
                        tokenizer=tokenizer,
                        split_records=probe_bundle["split_records"],
                        layer_grid=layer_grid,
                        desc=probe_bundle["desc"],
                    )
                    best_layer, best_auc, auc_per_layer, layer_clfs = select_best_layer_by_auc(
                        model=model,
                        tokenizer=tokenizer,
                        train_records=probe_bundle["train_records"],
                        val_records=probe_bundle["val_records"],
                        layer_grid=layer_grid,
                        seed=args.probe_seed,
                        max_selection_samples=args.probe_selection_max_samples,
                        desc=probe_bundle["desc"],
                    )
                    clf = (
                        train_probe_for_layer(
                            model=model,
                            tokenizer=tokenizer,
                            records=probe_bundle["retrain_records"],
                            layer=best_layer if best_layer is not None else layer_min,
                            seed=args.probe_seed,
                            max_train_samples=args.probe_train_max_samples,
                            desc=probe_bundle["desc"],
                        )
                        if best_layer is not None
                        else None
                    )
                    log_status(
                        "pipeline.py",
                        f"probe retrain {probe_bundle['desc']}: best_layer={best_layer} best_dev_auc={best_auc}",
                    )

                    score_records_with_probe(
                        model=model,
                        tokenizer=tokenizer,
                        records=probe_bundle["score_records"],
                        clf=clf,
                        layer=best_layer,
                        score_key=probe_bundle["score_key"],
                        desc=probe_bundle["desc"],
                    )
                    if probe_bundle["candidate_score_records"]:
                        score_records_with_probe(
                            model=model,
                            tokenizer=tokenizer,
                            records=probe_bundle["candidate_score_records"],
                            clf=clf,
                            layer=best_layer,
                            score_key="probe_score",
                            desc=f"{probe_bundle['desc']} candidate_scores",
                        )
                        probe_candidate_score_rows.extend(probe_bundle["candidate_score_records"])

                    layer_metrics: Dict[int, Dict[str, Any]] = {}
                    for layer_id, clf_layer in layer_clfs.items():
                        if clf_layer is None:
                            continue
                        layer_metrics[int(layer_id)] = evaluate_probe_from_cache(
                            eval_cache,
                            clf_layer,
                            int(layer_id),
                        )
                    chosen_metrics = evaluate_probe_from_cache(eval_cache, clf, best_layer)

                    selection_fit_records = _probe_fit_subset(
                        probe_bundle["train_records"],
                        args.probe_selection_max_samples,
                        args.probe_seed,
                    )
                    selection_val_records = _probe_fit_subset(
                        probe_bundle["val_records"],
                        args.probe_selection_max_samples,
                        args.probe_seed + 1,
                    )
                    chosen_fit_records = _probe_fit_subset(
                        probe_bundle["retrain_records"],
                        args.probe_train_max_samples,
                        args.probe_seed,
                    )

                    family_summary = save_probe_family_artifacts(
                        run_dir=run_dir,
                        probe_name=probe_bundle["meta_key"],
                        template_type=probe_bundle["template_type"],
                        desc=probe_bundle["desc"],
                        feature_source={
                            **feature_source_spec,
                            "probe_construction": probe_bundle["probe_construction"],
                            "probe_example_weighting": probe_bundle["probe_example_weighting"],
                        },
                        split_records=probe_bundle["split_records"],
                        selection_models=layer_clfs,
                        selection_metrics_by_layer=layer_metrics,
                        auc_per_layer=auc_per_layer,
                        best_layer=best_layer,
                        best_dev_auc=best_auc,
                        chosen_model=clf,
                        chosen_metrics=chosen_metrics,
                        selection_fit_records=selection_fit_records,
                        selection_val_records=selection_val_records,
                        chosen_fit_records=chosen_fit_records,
                        selection_fit_max_samples=args.probe_selection_max_samples,
                        chosen_fit_max_samples=args.probe_train_max_samples,
                        probe_seed=args.probe_seed,
                        probe_construction=probe_bundle["probe_construction"],
                        probe_example_weighting=probe_bundle["probe_example_weighting"],
                    )

                    all_probe_group_summary[probe_bundle["meta_key"]] = {
                        "probe_dir": family_summary["all_probes_dir"],
                        "manifest_path": family_summary["all_probes_manifest"],
                        "trained_layers": list(family_summary.get("trained_layers", [])),
                        "best_layer": family_summary["best_layer"],
                        "best_dev_auc": family_summary["best_dev_auc"],
                        "probe_construction": probe_bundle["probe_construction"],
                        "probe_example_weighting": probe_bundle["probe_example_weighting"],
                    }
                    chosen_probe_group_summary[probe_bundle["meta_key"]] = {
                        "probe_dir": family_summary["chosen_probe_dir"],
                        "manifest_path": family_summary["chosen_probe_manifest"],
                        "chosen_layer": family_summary["best_layer"],
                        "best_dev_auc": family_summary["best_dev_auc"],
                        "model_path": family_summary["saved_best_model"],
                        "metrics_path": family_summary["chosen_probe_metrics_path"],
                        "probe_construction": probe_bundle["probe_construction"],
                        "probe_example_weighting": probe_bundle["probe_example_weighting"],
                    }
                    return family_summary

                probes_meta[probe_record_sets["neutral"]["meta_key"]] = _run_probe_family_artifacts(
                    probe_record_sets["neutral"]
                )
                for btype in planned_bias_types:
                    bias_probe = probe_record_sets[btype]
                    probes_meta[bias_probe["meta_key"]] = _run_probe_family_artifacts(bias_probe)

                all_probes_dir = preferred_run_artifact_path(run_dir, "all_probes_dir")
                chosen_probe_dir = preferred_run_artifact_path(run_dir, "chosen_probe_dir")
                all_probes_manifest_path = write_probe_group_manifest(
                    group_dir=all_probes_dir,
                    artifact_group="all_probes",
                    probe_summaries=all_probe_group_summary,
                )
                chosen_probe_manifest_path = write_probe_group_manifest(
                    group_dir=chosen_probe_dir,
                    artifact_group="chosen_probe",
                    probe_summaries=chosen_probe_group_summary,
                )
                probes_meta["all_probes_dir"] = str(all_probes_dir)
                probes_meta["chosen_probe_dir"] = str(chosen_probe_dir)
                probes_meta["all_probes_manifest"] = str(all_probes_manifest_path)
                probes_meta["chosen_probe_manifest"] = str(chosen_probe_manifest_path)
                probes_meta["probe_construction"] = str(args.probe_construction)
                probes_meta["probe_example_weighting"] = str(args.probe_example_weighting)
                probes_meta["probe_candidate_score_rows"] = int(len(probe_candidate_score_rows))
        finish_stage()

        begin_stage(8, "final artifact saving")
        saved_paths = save_run_results(
            args=args,
            run_dir=run_dir,
            lock_path=lock_path,
            sampling_hash=sampling_hash,
            sampling_records_path=sampling_records_path,
            sampling_manifest_path=sampling_manifest_path,
            run_log_path=run_log_path,
            warning_log_path=warning_log_path,
            sampling_integrity_summary_path=sampling_integrity_summary_path,
            all_records=all_records,
            probe_candidate_score_rows=probe_candidate_score_rows,
            bias_types=planned_bias_types,
            probes_meta=probes_meta,
            run_timing=runtime_timing_snapshot("running"),
        )
        neutral_accuracy_warning = _strict_mc_neutral_below_chance_warning(all_records)
        if args.smoke_test and strict_mc_quality_failures:
            if neutral_accuracy_warning is not None:
                warn_status(
                    "pipeline.py",
                    "strict_mc_neutral_below_chance_baseline",
                    neutral_accuracy_warning,
                )
            raise RuntimeError(
                "strict MC quality gate failed: " + "; ".join(strict_mc_quality_failures)
            )
        run_status = "completed"
        log_status("pipeline.py", f"run completed successfully: {run_dir}")
        try:
            summary_path = saved_paths.get("run_summary_path", saved_paths["reports_summary_path"])
            reports_summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            reports_summary_payload = {}
        for line in build_terminal_final_stats_lines(reports_summary_payload):
            ok_status("pipeline.py", line)
        if neutral_accuracy_warning is not None:
            warn_status(
                "pipeline.py",
                "strict_mc_neutral_below_chance_baseline",
                neutral_accuracy_warning,
            )
        finish_stage()
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        log_status("pipeline.py", f"run failed: {run_error}")
        raise
    finally:
        try:
            try:
                write_run_status(run_dir, args=args, status=run_status, lock_path=lock_path, error=run_error)
            finally:
                try:
                    _finalize_warning_summary(run_dir)
                except Exception as warning_summary_exc:
                    log_status(
                        "pipeline.py",
                        f"warning summary finalization failed: {type(warning_summary_exc).__name__}: {warning_summary_exc}",
                    )
            try:
                if saved_paths.get("run_summary_path") is not None:
                    refresh_runtime_summary_artifacts(
                        run_dir=run_dir,
                        runtime_timing=runtime_timing_snapshot(run_status),
                    )
            except Exception as runtime_summary_exc:
                log_status(
                    "pipeline.py",
                    f"runtime summary refresh failed: {type(runtime_summary_exc).__name__}: {runtime_summary_exc}",
                )
        finally:
            if stage_bar is not None:
                stage_bar.close()
            release_run_lock(lock_path)
            clear_run_logging()


__all__ = [
    "load_llm",
    "run_pipeline",
]
