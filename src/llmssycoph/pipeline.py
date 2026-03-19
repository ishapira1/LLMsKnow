from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from tqdm.auto import tqdm

from .cli import load_env_file, resolve_bias_types, resolve_device, resolve_hf_cache_dir
from .constants import (
    MC_MODE_STRICT,
    STRICT_MC_MAX_CAP_HIT_RATE,
    STRICT_MC_MAX_EXPLICIT_PARSE_FAILURES,
    STRICT_MC_MAX_MULTIPLE_ANSWER_MARKER_ROWS,
    STRICT_MC_MIN_EXACT_FORMAT_RATE,
    STRICT_MC_MIN_COMMITMENT_RATE,
    STRICT_MC_MIN_STARTS_WITH_ANSWER_RATE,
)
from .data import (
    build_question_groups,
    deduplicate_rows,
    ensure_sycophancy_eval_cached,
    prepare_benchmark_rows,
    read_jsonl,
    resolve_ays_mc_datasets,
    split_groups_train_val_test,
    unique_dataset_names,
)
from .grading import add_empirical_t, build_probe_record_sets, refresh_sample_records_for_groups
from .logging_utils import clear_run_logging, configure_run_logging, log_status, tqdm_desc, warn_status
from .llm import (
    build_sampling_spec,
    load_llm,
    load_sampling_cache_candidate,
    normalize_sample_records,
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
    write_run_status,
)
from .sampling_integrity import build_sampling_integrity_summary, log_sampling_integrity_summary
from .saving_manager import (
    persist_sampling_state,
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

    example = groups[0]
    log_status(
        "pipeline.py",
        f"dataset example question_id={example['question_id']} question='{_preview_text(example['question'])}' "
        f"correct='{_preview_text(example['correct_answer'])}' "
        f"incorrect='{_preview_text(example['incorrect_answer'])}'",
    )
    for template_type in ["neutral", *bias_types]:
        row = example["rows_by_type"].get(template_type)
        if row is None:
            continue
        prompt_text = _preview_text((row.get("metadata", {}) or {}).get("prompt_template", ""), limit=180)
        if prompt_text:
            log_status(
                "pipeline.py",
                f"dataset example template={template_type} prompt_template='{prompt_text}'",
            )
        prompt_preview = _preview_text(
            "\n".join(
                message.get("content", "")
                for message in row.get("prompt", [])
                if isinstance(message, dict) and isinstance(message.get("content"), str)
            ),
            limit=220,
        )
        if prompt_preview:
            log_status(
                "pipeline.py",
                f"dataset example template={template_type} prompt_preview='{prompt_preview}'",
                )


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
        log_status("pipeline.py", "reuse strategy: disabled by --no_reuse_sampling_cache")
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

    return {
        "total": total,
        "commitment_rate": committed / total,
        "starts_with_answer_rate": starts_with_answer / total,
        "cap_hit_rate": cap_hits / total,
        "explicit_parse_failures": explicit_parse_failures,
        "exact_format_rate": exact_format_rows / total,
        "multiple_answer_marker_rows": multiple_answer_marker_rows,
        "max_neutral_bias_answer_gap": max_bias_gap,
        "by_template": by_template,
    }


def _log_strict_mc_quality_summary(summary: Dict[str, Any]) -> None:
    if not summary:
        return
    log_status(
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


def run_pipeline(args) -> None:
    import torch

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

    def begin_stage(index: int, message: str) -> None:
        if stage_bar is not None:
            stage_bar.set_description(tqdm_desc("pipeline.py", f"stage {index}/{stage_count} {message}"))
        log_status("pipeline.py", f"stage {index}/{stage_count}: {message}")

    def finish_stage() -> None:
        if stage_bar is not None:
            stage_bar.update(1)

    run_status = "failed"
    run_error: Optional[str] = None
    strict_mc_quality_report: Dict[str, Any] = {}
    strict_mc_quality_failures: List[str] = []
    sampling_integrity_summary: Dict[str, Any] = {}
    sampling_integrity_summary_path = preferred_run_artifact_path(run_dir, "sampling_integrity_summary")
    probe_candidate_score_rows: List[Dict[str, Any]] = []
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
        log_status(
            "pipeline.py",
            "parsed arguments: "
            + json.dumps(vars(args), ensure_ascii=False, sort_keys=True, default=str),
        )
        planned_bias_types = resolve_bias_types(args.bias_types)
        resolved_ays_mc_datasets = resolve_ays_mc_datasets(args.ays_mc_datasets)
        args.ays_mc_datasets = resolved_ays_mc_datasets
        log_status(
            "pipeline.py",
            f"execution plan: model={args.model} benchmark_source={args.benchmark_source} "
            f"bias_types={planned_bias_types} "
            f"dataset_name={args.dataset_name} "
            f"draws={args.n_draws} temperature={args.temperature} top_p={args.top_p} "
            f"max_new_tokens={args.max_new_tokens} smoke_test={args.smoke_test}",
        )
        finish_stage()

        begin_stage(2, "dataset loading, grouping, and split planning")
        random.seed(args.seed)
        np.random.seed(args.seed)
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
        device = resolve_device(args.device)
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
        args.strict_mc_choice_scoring = bool(any_choice_scoring)
        if all_choice_scoring and args.n_draws != 1:
            log_status(
                "pipeline.py",
                f"strict MC choice scoring active for all prompts; overriding n_draws from {args.n_draws} to 1",
            )
            args.n_draws = 1

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
        _log_strict_mc_quality_summary(strict_mc_quality_report)
        strict_mc_quality_failures = _strict_mc_quality_issues(strict_mc_quality_report)
        for issue in strict_mc_quality_failures:
            warn_status("pipeline.py", "strict_mc_quality_gate", issue)
        finish_stage()

        begin_stage(7, "probe selection, training, and scoring")
        model, tokenizer = llm.get_model_and_tokenizer()
        probes_meta: Dict[str, Any] = {}
        if strict_mc_quality_report:
            probes_meta["strict_mc_quality"] = {
                "summary": strict_mc_quality_report,
                "issues": strict_mc_quality_failures,
                "status": "failed" if strict_mc_quality_failures else "passed",
            }

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
            probes_meta["probe_candidate_scores_path"] = str(
                preferred_run_artifact_path(run_dir, "probe_candidate_scores")
            )
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
            probes_meta["probe_candidate_scores_path"] = str(
                preferred_run_artifact_path(run_dir, "probe_candidate_scores")
            )
            probes_meta["probe_candidate_score_rows"] = int(len(probe_candidate_score_rows))
        finish_stage()

        begin_stage(8, "final artifact saving")
        save_run_results(
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
        )
        if args.smoke_test and strict_mc_quality_failures:
            raise RuntimeError(
                "strict MC quality gate failed: " + "; ".join(strict_mc_quality_failures)
            )
        run_status = "completed"
        log_status("pipeline.py", f"run completed successfully: {run_dir}")
        finish_stage()
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        log_status("pipeline.py", f"run failed: {run_error}")
        raise
    finally:
        try:
            write_run_status(run_dir, args=args, status=run_status, lock_path=lock_path, error=run_error)
        finally:
            if stage_bar is not None:
                stage_bar.close()
            release_run_lock(lock_path)
            clear_run_logging()


__all__ = [
    "load_llm",
    "run_pipeline",
]
