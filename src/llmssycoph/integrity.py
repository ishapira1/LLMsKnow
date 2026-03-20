from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from .data import prompt_id_for
from .logging_utils import warn_status
from .runtime import model_slug, resolve_run_artifact_path
from .saving_manager import (
    P_CORRECT_COLUMN,
    P_SELECTED_COLUMN,
    build_model_summary_by_bias_df,
    build_model_summary_by_template_df,
)


REQUIRED_ARTIFACT_KEYS = (
    "run_config",
    "status",
    "sampling_manifest",
    "sampling_records",
    "sampling_integrity_summary",
    "sampled_responses",
    "reports_summary",
    "probe_scores_by_prompt",
    "executive_summary",
)

ALLOWED_REQUESTED_DEVICE_VALUES = {"auto", "cpu", "cuda", "mps"}
ALLOWED_RESOLVED_DEVICE_VALUES = {"cpu", "cuda", "mps"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_list_like(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _resolve_device_metadata(run_config: Dict[str, Any]) -> tuple[str, str]:
    requested_device = str(run_config.get("requested_device", "") or "").strip()
    legacy_device = str(run_config.get("device", "") or "").strip()
    resolved_device = str(run_config.get("resolved_device", "") or "").strip()

    if not requested_device:
        requested_device = legacy_device
    if not resolved_device and requested_device and requested_device != "auto":
        resolved_device = requested_device
    return requested_device, resolved_device


def _extract_reports_summary_rows(reports_summary: Any, run_summary: Any) -> List[Dict[str, Any]]:
    for payload in (reports_summary, run_summary):
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            raw_rows = payload.get("summary_rows")
            if isinstance(raw_rows, list):
                return [row for row in raw_rows if isinstance(row, dict)]

    if not isinstance(reports_summary, dict):
        return []

    rows: List[Dict[str, Any]] = []
    overall = reports_summary.get("overall", {})
    if isinstance(overall, dict):
        rows.append(
            {
                "bias_type": "overall",
                "accuracy": overall.get("accuracy"),
                "avg_p_correct": overall.get("avg_p_correct"),
                "avg_p_selected": overall.get("avg_p_selected"),
                "avg_delta_p_biased_minus_neutral": overall.get("avg_delta_p_xprime_minus_x"),
                "harmful_flip_rate": overall.get("harmful_flip_rate"),
                "helpful_flip_rate": overall.get("helpful_flip_rate"),
                "unchanged_correctness_rate": overall.get("unchanged_correctness_rate"),
            }
        )

    template_lookup = {
        str(row.get("template_type", "") or "").strip(): row
        for row in reports_summary.get("accuracy_by_template", [])
        if isinstance(row, dict) and str(row.get("template_type", "") or "").strip()
    }
    if "neutral" in template_lookup:
        neutral_row = template_lookup["neutral"]
        rows.append(
            {
                "bias_type": "neutral",
                "n_prompt_rows": neutral_row.get("n_rows"),
                "n_questions": neutral_row.get("n_questions"),
                "n_usable_prompt_rows": neutral_row.get("n_usable_rows"),
                "usable_rate": neutral_row.get("usable_rate"),
                "ambiguous_rate": neutral_row.get("ambiguous_rate"),
                "accuracy": neutral_row.get("accuracy"),
                "avg_p_correct": neutral_row.get("avg_p_correct"),
                "avg_p_selected": neutral_row.get("avg_p_selected"),
            }
        )

    for bias_row in reports_summary.get("accuracy_by_bias_type", []):
        if not isinstance(bias_row, dict):
            continue
        template_row = template_lookup.get(str(bias_row.get("bias_type", "") or "").strip(), {})
        rows.append(
            {
                "bias_type": bias_row.get("bias_type"),
                "n_prompt_rows": template_row.get("n_rows"),
                "n_questions": template_row.get("n_questions"),
                "n_usable_prompt_rows": template_row.get("n_usable_rows"),
                "usable_rate": template_row.get("usable_rate"),
                "ambiguous_rate": template_row.get("ambiguous_rate"),
                "accuracy": template_row.get("accuracy", bias_row.get("accuracy_xprime")),
                "avg_p_correct": template_row.get("avg_p_correct", bias_row.get("avg_p_xprime")),
                "avg_p_selected": template_row.get("avg_p_selected"),
                "neutral_accuracy": bias_row.get("accuracy_x"),
                "biased_accuracy": bias_row.get("accuracy_xprime"),
                "avg_delta_p_biased_minus_neutral": bias_row.get("avg_delta_p_xprime_minus_x"),
                "harmful_flip_rate": bias_row.get("harmful_flip_rate"),
                "helpful_flip_rate": bias_row.get("helpful_flip_rate"),
                "unchanged_correctness_rate": bias_row.get("unchanged_correctness_rate"),
            }
        )
    return rows


def _check_probability_series(series: pd.Series, column_name: str, issues: List[str]) -> None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return
    bad_mask = (numeric < 0) | (numeric > 1)
    if bad_mask.any():
        issues.append(f"{column_name} contains values outside [0, 1]")


def _sample_probability_series(samples: pd.DataFrame, column_name: str, legacy_column_name: str) -> pd.Series:
    if column_name in samples.columns:
        return samples[column_name]
    if legacy_column_name in samples.columns:
        return samples[legacy_column_name]
    return pd.Series([pd.NA] * len(samples), index=samples.index)


def _reconstruct_pairs_from_samples(samples: pd.DataFrame, bias_types: Sequence[str]) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "question_id",
                "bias_type",
                "draw_idx",
                "C_x_y",
                "C_xprime_yprime",
                "T_x",
                "T_xprime",
                "probe_x",
                "probe_xprime",
                "y_x",
                "y_xprime",
            ]
        )

    neutral = samples[samples["template_type"].astype(str) == "neutral"].copy()
    if neutral.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for bias_type in bias_types:
        biased = samples[samples["template_type"].astype(str) == str(bias_type)].copy()
        if biased.empty:
            continue
        merged = neutral.merge(
            biased,
            on=["split", "question_id", "draw_idx"],
            how="inner",
            suffixes=("_x", "_xprime"),
        )
        if merged.empty:
            continue
        usable_mask = (
            pd.to_numeric(merged.get("usable_for_metrics_x", 0), errors="coerce").fillna(0) > 0
        ) & (
            pd.to_numeric(merged.get("usable_for_metrics_xprime", 0), errors="coerce").fillna(0) > 0
        )
        merged = merged[usable_mask].copy()
        for _, row in merged.iterrows():
            rows.append(
                {
                    "split": str(row["split"]),
                    "question_id": str(row["question_id"]),
                    "bias_type": str(bias_type),
                    "draw_idx": int(row["draw_idx"]),
                    "C_x_y": pd.to_numeric(pd.Series([row.get("correctness_x")]), errors="coerce").iloc[0],
                    "C_xprime_yprime": pd.to_numeric(pd.Series([row.get("correctness_xprime")]), errors="coerce").iloc[0],
                    "T_x": pd.to_numeric(pd.Series([row.get("T_prompt_x")]), errors="coerce").iloc[0],
                    "T_xprime": pd.to_numeric(pd.Series([row.get("T_prompt_xprime")]), errors="coerce").iloc[0],
                    "probe_x": pd.to_numeric(pd.Series([row.get("probe_x_x")]), errors="coerce").iloc[0]
                    if "probe_x_x" in merged.columns
                    else pd.to_numeric(pd.Series([row.get("probe_x")]), errors="coerce").iloc[0],
                    "probe_xprime": pd.to_numeric(pd.Series([row.get("probe_xprime_xprime")]), errors="coerce").iloc[0]
                    if "probe_xprime_xprime" in merged.columns
                    else pd.to_numeric(pd.Series([row.get("probe_xprime")]), errors="coerce").iloc[0],
                    "y_x": str(row.get("response_x", "") or ""),
                    "y_xprime": str(row.get("response_xprime", "") or ""),
                }
            )
    return pd.DataFrame(rows)


def _resolve_run_dir(run_dir: str | None, out_dir: str, model: str, run_name: str) -> Path:
    if run_dir:
        return Path(run_dir)
    return Path(out_dir) / model_slug(model) / run_name


def check_run_integrity(run_dir: Path) -> Dict[str, Any]:
    issues: List[str] = []
    run_dir = run_dir.resolve()

    required_artifact_paths = [resolve_run_artifact_path(run_dir, "run_log")]
    required_artifact_paths.extend(
        resolve_run_artifact_path(run_dir, artifact_key) for artifact_key in REQUIRED_ARTIFACT_KEYS
    )
    for artifact_path in required_artifact_paths:
        if not artifact_path.exists():
            issues.append(f"missing artifact: {artifact_path}")

    if issues:
        raise RuntimeError("\n".join(issues))

    run_config_path = resolve_run_artifact_path(run_dir, "run_config")
    status_path = resolve_run_artifact_path(run_dir, "status")
    manifest_path = resolve_run_artifact_path(run_dir, "sampling_manifest")
    sampling_records_path = resolve_run_artifact_path(run_dir, "sampling_records")
    sampling_integrity_path = resolve_run_artifact_path(run_dir, "sampling_integrity_summary")
    samples_path = resolve_run_artifact_path(run_dir, "sampled_responses")
    reports_summary_path = resolve_run_artifact_path(run_dir, "reports_summary")
    run_summary_path = resolve_run_artifact_path(run_dir, "run_summary")
    probe_scores_by_prompt_path = resolve_run_artifact_path(run_dir, "probe_scores_by_prompt")
    executive_summary_path = resolve_run_artifact_path(run_dir, "executive_summary")

    run_config = _load_json(run_config_path)
    status = _load_json(status_path)
    manifest = _load_json(manifest_path)
    reports_summary = _load_json(reports_summary_path)
    run_summary = _load_json(run_summary_path) if run_summary_path.exists() else {}
    sampling_integrity = _load_json(sampling_integrity_path)
    summary_meta = run_summary if isinstance(run_summary, dict) and run_summary else (
        reports_summary if isinstance(reports_summary, dict) else {}
    )
    sampling_only = _bool_like(run_config.get("sampling_only", False))
    probe_training_status = ""
    if isinstance(summary_meta, dict):
        probe_training_status = str(summary_meta.get("probe_training_status", "") or "")
    if not probe_training_status and sampling_only:
        probe_training_status = "skipped_by_sampling_only"
    probe_stage_skipped = sampling_only or probe_training_status == "skipped_by_sampling_only"
    summary_rows = _extract_reports_summary_rows(reports_summary, run_summary)
    summary_rows_df = pd.DataFrame(summary_rows)

    samples = pd.read_csv(samples_path)
    probe_scores_by_prompt_df = pd.read_csv(probe_scores_by_prompt_path)
    tuples_df = _reconstruct_pairs_from_samples(samples, _parse_list_like(run_config.get("bias_types")))

    sample_required_columns = {"question_id", "prompt_id", "question", "prompt_text", P_CORRECT_COLUMN, P_SELECTED_COLUMN}
    missing_sample_columns = sorted(sample_required_columns.difference(samples.columns))
    if missing_sample_columns:
        issues.append(
            f"sampled_responses.csv is missing required columns: {missing_sample_columns}"
        )
    summary_required_columns = {"bias_type", "accuracy", "avg_p_correct", "avg_p_selected"}
    if summary_rows_df.empty:
        issues.append("reports/summary.json is empty")
    else:
        missing_summary_columns = sorted(summary_required_columns.difference(summary_rows_df.columns))
        if missing_summary_columns:
            issues.append(
                "reports/summary.json is missing required columns: "
                f"{missing_summary_columns}"
            )

    probe_scores_by_prompt_required_columns = {
        "probe_name",
        "question_id",
        "prompt_id",
        "correct_letter",
        "selected_choice",
        "probe_score_correct_choice",
        "probe_score_selected_choice",
        "probe_argmax_choice",
    }
    missing_probe_scores_by_prompt_columns = sorted(
        probe_scores_by_prompt_required_columns.difference(probe_scores_by_prompt_df.columns)
    )
    if missing_probe_scores_by_prompt_columns:
        issues.append(
            "probe_scores_by_prompt.csv is missing required columns: "
            f"{missing_probe_scores_by_prompt_columns}"
        )

    if status.get("status") != "completed":
        issues.append(f"status.json reports status={status.get('status')!r}, expected 'completed'")
    if status.get("run_dir") and str(run_dir) != str(Path(status["run_dir"]).resolve()):
        issues.append("status.json run_dir does not match the checked run directory")
    if run_config.get("run_dir") and str(run_dir) != str(Path(run_config["run_dir"]).resolve()):
        issues.append("run_config.json run_dir does not match the checked run directory")

    if not bool(run_config.get("smoke_test", False)):
        issues.append("run_config.json does not indicate smoke_test=true")
    if str(run_config.get("benchmark_source")) != "ays_mc_single_turn":
        issues.append("benchmark_source is not ays_mc_single_turn")
    if str(run_config.get("input_jsonl")) != "are_you_sure.jsonl":
        issues.append("input_jsonl is not are_you_sure.jsonl")
    if str(run_config.get("dataset_name")) != "aqua_mc":
        issues.append("dataset_name is not aqua_mc")
    if str(run_config.get("mc_mode")) != "strict_mc":
        issues.append("mc_mode is not strict_mc")
    if sampling_only and probe_training_status != "skipped_by_sampling_only":
        issues.append("sampling_only run is missing probe_training_status=skipped_by_sampling_only")
    requested_device, resolved_device = _resolve_device_metadata(run_config)
    if not requested_device:
        issues.append("run_config.json is missing requested_device/device")
    if requested_device and requested_device not in ALLOWED_REQUESTED_DEVICE_VALUES:
        issues.append(f"requested_device has unexpected value {requested_device!r}")
    if requested_device == "auto" and not resolved_device:
        issues.append("run_config.json is missing resolved_device for requested_device='auto'")
    if resolved_device and resolved_device not in ALLOWED_RESOLVED_DEVICE_VALUES:
        issues.append(f"resolved_device has unexpected value {resolved_device!r}")

    bias_types = _parse_list_like(run_config.get("bias_types"))
    expected_templates = {"neutral", *bias_types}
    expected_question_ids = set(manifest.get("sampling_spec", {}).get("train_question_ids", []))
    expected_question_ids.update(manifest.get("sampling_spec", {}).get("val_question_ids", []))
    expected_question_ids.update(manifest.get("sampling_spec", {}).get("test_question_ids", []))

    expected_total_records = int(manifest.get("expected_records", 0) or 0)
    if not bool(manifest.get("is_complete", False)):
        issues.append("sampling_manifest.json reports is_complete=false")
    if int(manifest.get("n_records", -1)) != expected_total_records:
        issues.append(
            f"sampling_manifest.json n_records={manifest.get('n_records')} "
            f"does not match expected_records={expected_total_records}"
        )
    if len(samples) != expected_total_records:
        issues.append(
            f"sampled_responses.csv has {len(samples)} rows, expected {expected_total_records}"
        )

    sampling_records_lines = sum(
        1 for line in sampling_records_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if sampling_records_lines != expected_total_records:
        issues.append(
            f"sampling_records.jsonl has {sampling_records_lines} non-empty lines, expected {expected_total_records}"
        )
    sampling_mode_series = (
        samples["sampling_mode"].fillna("generation").astype(str)
        if "sampling_mode" in samples.columns
        else pd.Series(["generation"] * len(samples))
    )
    if int(sampling_integrity.get("total_records", -1)) != expected_total_records:
        issues.append(
            "sampling_integrity_summary.json total_records does not match sampling_manifest.json expected_records"
        )
    integrity_modes = sampling_integrity.get("by_sampling_mode", {})
    if not isinstance(integrity_modes, dict):
        issues.append("sampling_integrity_summary.json is missing by_sampling_mode")
        integrity_modes = {}
    for sampling_mode, mode_summary in integrity_modes.items():
        if not isinstance(mode_summary, dict):
            issues.append(f"sampling_integrity_summary.json has non-object summary for mode={sampling_mode}")
            continue
        mode_total = int(mode_summary.get("total", -1))
        actual_mode_total = int((sampling_mode_series == str(sampling_mode)).sum())
        if mode_total != actual_mode_total:
            issues.append(
                f"sampling_integrity_summary.json total mismatch for sampling_mode={sampling_mode}: "
                f"summary={mode_total} actual={actual_mode_total}"
            )
        buckets = mode_summary.get("buckets", {})
        if not isinstance(buckets, dict):
            issues.append(f"sampling_integrity_summary.json buckets missing for sampling_mode={sampling_mode}")
            continue
        bucket_total = 0
        for bucket_name, bucket_stats in buckets.items():
            if not isinstance(bucket_stats, dict):
                issues.append(
                    f"sampling_integrity_summary.json bucket {bucket_name!r} for mode={sampling_mode} is not an object"
                )
                continue
            bucket_total += int(bucket_stats.get("count", 0) or 0)
        if bucket_total != mode_total:
            issues.append(
                f"sampling_integrity_summary.json bucket counts do not sum to total for sampling_mode={sampling_mode}"
            )

    sample_key_cols = ["split", "question_id", "template_type", "draw_idx"]
    duplicate_sample_keys = int(samples.duplicated(subset=sample_key_cols).sum())
    if duplicate_sample_keys:
        issues.append(f"sampled_responses.csv has {duplicate_sample_keys} duplicate sample keys")
    duplicate_record_ids = int(samples.duplicated(subset=["record_id"]).sum())
    if duplicate_record_ids:
        issues.append(f"sampled_responses.csv has {duplicate_record_ids} duplicate record_id values")

    present_splits = set(samples["split"].dropna().astype(str))
    manifest_split_stats = manifest.get("split_stats", {})
    expected_splits = {
        split_name
        for split_name, stats in manifest_split_stats.items()
        if int(stats.get("expected_records", 0) or 0) > 0
    }
    if present_splits != expected_splits:
        issues.append(f"split coverage mismatch: present={sorted(present_splits)} expected={sorted(expected_splits)}")
    for split_name in sorted(expected_splits):
        split_expected = int(manifest_split_stats[split_name].get("expected_records", 0) or 0)
        split_actual = int((samples["split"].astype(str) == split_name).sum())
        if split_actual != split_expected:
            issues.append(f"split={split_name} has {split_actual} rows, expected {split_expected}")

    present_templates = set(samples["template_type"].dropna().astype(str))
    if present_templates != expected_templates:
        issues.append(
            f"template coverage mismatch: present={sorted(present_templates)} expected={sorted(expected_templates)}"
        )

    present_question_ids = set(samples["question_id"].dropna().astype(str))
    if present_question_ids != expected_question_ids:
        issues.append(
            "question_id coverage mismatch between sampled_responses.csv and sampling_manifest.json"
        )
    if "prompt_id" in samples.columns:
        expected_prompt_ids = {
            prompt_id_for(question_id, template_type)
            for question_id in expected_question_ids
            for template_type in expected_templates
        }
        present_prompt_ids = set(samples["prompt_id"].dropna().astype(str))
        if present_prompt_ids != expected_prompt_ids:
            issues.append("prompt_id coverage mismatch in sampled_responses.csv")
        prompt_id_per_variant = (
            samples.groupby(["question_id", "template_type"], as_index=False)["prompt_id"]
            .nunique()
            .rename(columns={"prompt_id": "n_prompt_ids"})
        )
        if int((prompt_id_per_variant["n_prompt_ids"] != 1).sum()):
            issues.append("sampled_responses.csv does not keep a 1:1 mapping from (question_id, template_type) to prompt_id")

    configured_dataset = str(run_config.get("dataset_name", "") or "")
    sample_datasets = set(samples["dataset"].dropna().astype(str))
    if configured_dataset and configured_dataset != "all" and sample_datasets != {configured_dataset}:
        issues.append(f"dataset coverage mismatch: present={sorted(sample_datasets)} expected={[configured_dataset]}")

    task_formats = set(samples["task_format"].dropna().astype(str))
    if task_formats != {"multiple_choice"}:
        issues.append(f"task_format mismatch: present={sorted(task_formats)} expected=['multiple_choice']")
    mc_modes = set(samples["mc_mode"].dropna().astype(str))
    if mc_modes != {"strict_mc"}:
        issues.append(f"mc_mode mismatch in sampled_responses.csv: present={sorted(mc_modes)}")
    if task_formats == {"multiple_choice"}:
        mc_confusion_matrix = summary_meta.get("mc_confusion_matrix") if isinstance(summary_meta, dict) else None
        mc_confusion_matrix_path = resolve_run_artifact_path(run_dir, "mc_confusion_matrix")
        if not isinstance(mc_confusion_matrix, dict):
            issues.append("run summary is missing mc_confusion_matrix for multiple_choice runs")
        if not mc_confusion_matrix_path.exists():
            issues.append("reports confusion matrix is missing for multiple_choice runs")
        elif isinstance(mc_confusion_matrix, dict):
            mc_confusion_df = pd.read_csv(mc_confusion_matrix_path)
            expected_labels = [
                str(label)
                for label in mc_confusion_matrix.get("choice_labels", [])
                if str(label)
            ]
            expected_columns = ["predicted_letter", *expected_labels] if expected_labels else ["predicted_letter"]
            if list(mc_confusion_df.columns) != expected_columns:
                issues.append(
                    "reports confusion matrix has unexpected columns: "
                    f"present={list(mc_confusion_df.columns)} expected={expected_columns}"
                )
            expected_rows = mc_confusion_matrix.get("summary_rows", [])
            if isinstance(expected_rows, list) and len(mc_confusion_df) != len(expected_rows):
                issues.append("reports confusion matrix row count does not match run summary payload")
            if expected_labels and all(label in mc_confusion_df.columns for label in expected_labels):
                confusion_total = int(
                    pd.to_numeric(mc_confusion_df[expected_labels].to_numpy().ravel(), errors="coerce").sum()
                )
                if confusion_total != int(mc_confusion_matrix.get("n_confusion_rows", -1)):
                    issues.append("reports confusion matrix total does not match run summary payload")

    draws_per_prompt = int(run_config.get("n_draws", 0) or 0)
    strict_mc_choice_scoring = bool(run_config.get("strict_mc_choice_scoring", False))
    per_prompt_counts = (
        samples.groupby(["split", "question_id", "template_type"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    prompt_modes = (
        samples.groupby(["split", "question_id", "template_type"], as_index=False)
        .agg(
            task_format=("task_format", "first"),
            mc_mode=("mc_mode", "first"),
        )
    )
    per_prompt_counts = per_prompt_counts.merge(
        prompt_modes,
        on=["split", "question_id", "template_type"],
        how="left",
    )
    per_prompt_counts["expected_n_rows"] = draws_per_prompt
    if strict_mc_choice_scoring:
        strict_mask = (
            per_prompt_counts["task_format"].astype(str) == "multiple_choice"
        ) & (
            per_prompt_counts["mc_mode"].astype(str) == "strict_mc"
        )
        per_prompt_counts.loc[strict_mask, "expected_n_rows"] = 1
    bad_prompt_counts = per_prompt_counts[
        per_prompt_counts["n_rows"] != per_prompt_counts["expected_n_rows"]
    ]
    if not bad_prompt_counts.empty:
        issues.append(
            "not every (split, question_id, template_type) has the configured number of draws"
        )

    model_summary_by_template_df = build_model_summary_by_template_df(samples)
    model_summary_by_bias_df = build_model_summary_by_bias_df(tuples_df)

    _check_probability_series(samples["T_prompt"], "sampled_responses.T_prompt", issues)
    _check_probability_series(
        _sample_probability_series(samples, P_CORRECT_COLUMN, "choice_probability_correct"),
        f"sampled_responses.{P_CORRECT_COLUMN}",
        issues,
    )
    _check_probability_series(
        _sample_probability_series(samples, P_SELECTED_COLUMN, "choice_probability_selected"),
        f"sampled_responses.{P_SELECTED_COLUMN}",
        issues,
    )
    if not model_summary_by_template_df.empty:
        _check_probability_series(model_summary_by_template_df["accuracy"], "reports.summary.accuracy_by_template.accuracy", issues)
        _check_probability_series(
            model_summary_by_template_df["avg_p_correct"],
            "reports.summary.accuracy_by_template.avg_p_correct",
            issues,
        )
    if not model_summary_by_bias_df.empty:
        _check_probability_series(model_summary_by_bias_df["accuracy_x"], "reports.summary.accuracy_by_bias_type.accuracy_x", issues)
        _check_probability_series(
            model_summary_by_bias_df["accuracy_xprime"],
            "reports.summary.accuracy_by_bias_type.accuracy_xprime",
            issues,
        )
        _check_probability_series(model_summary_by_bias_df["avg_p_x"], "reports.summary.accuracy_by_bias_type.avg_p_x", issues)
        _check_probability_series(
            model_summary_by_bias_df["avg_p_xprime"],
            "reports.summary.accuracy_by_bias_type.avg_p_xprime",
            issues,
        )

    correctness_numeric = pd.to_numeric(samples["correctness"], errors="coerce").dropna()
    if not correctness_numeric.isin([0, 1]).all():
        issues.append("sampled_responses.correctness contains values other than 0/1/NaN")

    if not tuples_df.empty:
        tuple_key_cols = ["split", "question_id", "bias_type", "draw_idx"]
        duplicate_tuple_keys = int(tuples_df.duplicated(subset=tuple_key_cols).sum())
        if duplicate_tuple_keys:
            issues.append(f"reconstructed pair table has {duplicate_tuple_keys} duplicate tuple keys")
        tuple_biases = set(tuples_df["bias_type"].dropna().astype(str))
        if not tuple_biases.issubset(set(bias_types)):
            issues.append(
                f"reconstructed pair table contains unexpected bias types: {sorted(tuple_biases - set(bias_types))}"
            )

    if not summary_rows_df.empty:
        if "bias_type" in summary_rows_df.columns:
            duplicate_summary_rows = int(summary_rows_df.duplicated(subset=["bias_type"]).sum())
            if duplicate_summary_rows:
                issues.append(f"reports/summary.json has {duplicate_summary_rows} duplicate bias_type rows")
            summary_biases = set(summary_rows_df["bias_type"].dropna().astype(str))
            expected_summary_biases = {"overall", "neutral", *bias_types}
            if summary_biases != expected_summary_biases:
                issues.append("reports/summary.json has unexpected bias coverage")
        for column_name in (
            "accuracy",
            "avg_p_correct",
            "avg_p_selected",
            "neutral_accuracy",
            "biased_accuracy",
            "neutral_avg_p_correct",
            "biased_avg_p_correct",
            "harmful_flip_rate",
            "helpful_flip_rate",
            "unchanged_correctness_rate",
        ):
            if column_name in summary_rows_df.columns:
                _check_probability_series(summary_rows_df[column_name], f"reports.summary.{column_name}", issues)

    strict_quality = summary_meta.get("strict_mc_quality") if isinstance(summary_meta, dict) else None
    if isinstance(strict_quality, dict):
        if strict_quality.get("status") != "passed":
            issues.append(
                f"strict_mc_quality status is {strict_quality.get('status')!r}, expected 'passed'"
            )
        if strict_quality.get("issues"):
            issues.append(f"strict_mc_quality reported issues: {strict_quality.get('issues')}")
        summary = strict_quality.get("summary", {})
        by_template = summary.get("by_template", {}) if isinstance(summary, dict) else {}
        missing_quality_templates = expected_templates.difference(set(by_template.keys()))
        if missing_quality_templates:
            issues.append(
                f"strict_mc_quality summary is missing templates: {sorted(missing_quality_templates)}"
            )

    headline_counts = summary_meta.get("headline_counts", {}) if isinstance(summary_meta, dict) else {}
    if headline_counts:
        if int(headline_counts.get("sample_rows", -1)) != len(samples):
            issues.append("run summary headline_counts.sample_rows does not match sampled_responses.csv")
        if int(headline_counts.get("question_count", -1)) != len(expected_question_ids):
            issues.append("run summary headline_counts.question_count does not match sampling_manifest.json")
        if int(headline_counts.get("probe_score_prompt_rows", -1)) != len(probe_scores_by_prompt_df):
            issues.append("run summary headline_counts.probe_score_prompt_rows does not match probe_scores_by_prompt.csv")
        if probe_stage_skipped and int(headline_counts.get("probe_family_count", -1)) != 0:
            issues.append("run summary headline_counts.probe_family_count should be 0 when sampling_only=true")

    probe_summary_df = pd.DataFrame(summary_meta.get("probe_score_summaries", [])) if isinstance(summary_meta, dict) else pd.DataFrame()
    expected_probe_names = {"probe_no_bias", *[f"probe_bias_{bias_type}" for bias_type in bias_types]}
    if probe_stage_skipped:
        if not probe_scores_by_prompt_df.empty:
            issues.append("probe_scores_by_prompt.csv should be empty when sampling_only=true")
        if not probe_summary_df.empty:
            issues.append("run summary probe_score_summaries should be empty when sampling_only=true")
    else:
        if not probe_summary_df.empty:
            present_probe_names = set(probe_summary_df["probe_name"].dropna().astype(str))
            if present_probe_names != expected_probe_names:
                issues.append("run summary probe_score_summaries has unexpected probe coverage")
        elif expected_probe_names and isinstance(summary_meta, dict) and "probe_score_summaries" in summary_meta:
            issues.append("run summary probe_score_summaries is empty")

    executive_summary_text = executive_summary_path.read_text(encoding="utf-8").strip()
    if not executive_summary_text:
        issues.append("executive_summary.md is empty")
    if "Executive Summary" not in executive_summary_text:
        issues.append("executive_summary.md is missing the expected title")

    all_probes_dir = resolve_run_artifact_path(run_dir, "all_probes_dir")
    chosen_probe_dir = resolve_run_artifact_path(run_dir, "chosen_probe_dir")
    if probe_stage_skipped:
        if all_probes_dir.exists():
            issues.append("all_probes directory should be absent when sampling_only=true")
        if chosen_probe_dir.exists():
            issues.append("chosen_probe directory should be absent when sampling_only=true")
    else:
        if all_probes_dir.exists():
            all_manifest_path = all_probes_dir / "manifest.json"
            if not all_manifest_path.exists():
                issues.append("all_probes/manifest.json is missing")
        if chosen_probe_dir.exists():
            chosen_manifest_path = chosen_probe_dir / "manifest.json"
            if not chosen_manifest_path.exists():
                issues.append("chosen_probe/manifest.json is missing")

        for probe_name in ["probe_no_bias", *[f"probe_bias_{bias_type}" for bias_type in bias_types]]:
            if all_probes_dir.exists():
                family_manifest = all_probes_dir / probe_name / "manifest.json"
                if not family_manifest.exists():
                    issues.append(f"all_probes manifest is missing for {probe_name}")
            if chosen_probe_dir.exists():
                family_manifest = chosen_probe_dir / probe_name / "manifest.json"
                if not family_manifest.exists():
                    issues.append(f"chosen_probe manifest is missing for {probe_name}")

    if issues:
        raise RuntimeError("\n".join(issues))

    usable_rate = float(pd.to_numeric(samples["usable_for_metrics"], errors="coerce").fillna(0).mean())
    ambiguous_rate = 1.0 - usable_rate
    paired_stats = (
        tuples_df.groupby("bias_type", as_index=False)
        .agg(
            n_pairs=("draw_idx", "size"),
            neutral_accuracy=("C_x_y", "mean"),
            biased_accuracy=("C_xprime_yprime", "mean"),
        )
        .sort_values("bias_type")
    )
    harmful_flip = tuples_df.assign(
        harmful_flip=((tuples_df["C_x_y"] == 1) & (tuples_df["C_xprime_yprime"] == 0)).astype(float)
    )
    harmful_flip_stats = (
        harmful_flip.groupby("bias_type", as_index=False)
        .agg(harmful_flip_rate=("harmful_flip", "mean"))
        .sort_values("bias_type")
    )
    paired_stats = paired_stats.merge(harmful_flip_stats, on="bias_type", how="left")

    template_stats = (
        samples.groupby("template_type", as_index=False)
        .agg(
            n_rows=("record_id", "size"),
            usable_rate=("usable_for_metrics", "mean"),
            cap_hit_rate=("hit_max_new_tokens", "mean"),
            starts_with_answer_rate=("starts_with_answer_prefix", "mean"),
            exact_format_rate=("strict_format_exact", "mean"),
        )
        .sort_values("template_type")
    )

    return {
        "run_dir": str(run_dir),
        "sample_count": int(len(samples)),
        "tuple_count": int(len(tuples_df)),
        "question_count": int(len(expected_question_ids)),
        "requested_device": requested_device,
        "resolved_device": resolved_device or requested_device,
        "reports_summary_path": str(reports_summary_path),
        "bias_types": bias_types,
        "usable_rate": usable_rate,
        "ambiguous_rate": ambiguous_rate,
        "strict_mc_quality": strict_quality["summary"],
        "template_stats": template_stats,
        "paired_stats": paired_stats,
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(f"[integrity] run_dir={report['run_dir']}")
    print(
        "[integrity] artifacts:"
        f" samples={report['sample_count']}"
        f" tuples={report['tuple_count']}"
        f" questions={report['question_count']}"
    )
    print(
        "[integrity] device:"
        f" requested={report['requested_device'] or 'unknown'}"
        f" resolved={report['resolved_device'] or 'unknown'}"
    )
    print(
        "[integrity] usability:"
        f" usable_rate={_format_pct(report['usable_rate'])}"
        f" ambiguous_rate={_format_pct(report['ambiguous_rate'])}"
    )

    strict_quality = report["strict_mc_quality"]
    print(
        "[integrity] strict_mc_quality:"
        f" commitment_rate={_format_pct(strict_quality['commitment_rate'])}"
        f" starts_with_answer_rate={_format_pct(strict_quality['starts_with_answer_rate'])}"
        f" exact_format_rate={_format_pct(strict_quality['exact_format_rate'])}"
        f" cap_hit_rate={_format_pct(strict_quality['cap_hit_rate'])}"
        f" parse_failures={strict_quality['explicit_parse_failures']}"
        f" max_neutral_bias_answer_gap={_format_pct(strict_quality['max_neutral_bias_answer_gap'])}"
    )

    print("[integrity] template_stats:")
    print(report["template_stats"].to_string(index=False))
    print("[integrity] paired_stats:")
    print(report["paired_stats"].to_string(index=False))
    print("[integrity] all checks passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a completed smoke run and print a compact health report.")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit run directory to validate.")
    parser.add_argument("--out_dir", type=str, default="results/sycophancy_bias_probe")
    parser.add_argument("--model", type=str, required=False, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--run_name", type=str, required=False, default="smoke_aqua_mc_mistral7b_auto_q12_l4")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when integrity issues are found instead of emitting warnings.",
    )
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        model=args.model,
        run_name=args.run_name,
    )
    try:
        report = check_run_integrity(run_dir)
    except RuntimeError as exc:
        issues = [line.strip() for line in str(exc).splitlines() if line.strip()]
        warn_status(
            "integrity.py",
            "integrity_check_failed",
            f"integrity check found {len(issues)} issue(s) for run_dir={run_dir}",
        )
        for issue in issues:
            warn_status("integrity.py", "integrity_issue", issue)
        if args.strict:
            return 1
        warn_status(
            "integrity.py",
            "continuing_after_integrity_warnings",
            "continuing because --strict was not set",
        )
        return 0
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
